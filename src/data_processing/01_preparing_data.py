import argparse
import csv
import io
import os
import sys
import time
import subprocess
import signal
from typing import List, Tuple, Optional
import chess
import chess.pgn
import chess.engine
import multiprocessing as mp
from queue import Empty

# ===============================
# Configuration
# ===============================

DEFAULT_OPEN_TARGET = 300000
DEFAULT_MID_TARGET = 300000
DEFAULT_END_TARGET = 400000

SHARD_SIZE = 100000
BATCH_EVAL_SIZE = 100
ENGINE_INSTANCES = 4
ENGINE_TIME_SEC = 0.012
ENGINE_HASH_MB = 64
ENGINE_THREADS_PER_INSTANCE = 0.1
MATE_CP = 2000
CP_CLAMP = 2000
ENDGAME_FOCUS_TRIGGER_GAP = 0.08
ENDGAME_FOCUS_PLY_MOD = 2
MAX_POSITIONS_PER_GAME = 8

CSV_HEADER = [
    "fen", "eval_cp", "eval_norm", "phase_bucket", "ply",
    "move_number", "material_white", "material_black", "piece_count_total",
    "endgame_rule_flag", "mate_flag", "time_ms", "depth"
]

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

# ===============================
# Classes
# ===============================

class PositionMeta:
    def __init__(self,
                 fen_full: str,
                 fen_dedup: str,
                 phase_bucket: str,
                 endgame_flag: int,
                 ply: int,
                 move_number: int,
                 material_white: int,
                 material_black: int,
                 piece_count_total: int):
        self.fen_full = fen_full
        self.fen_dedup = fen_dedup
        self.phase_bucket = phase_bucket
        self.endgame_flag = endgame_flag
        self.ply = ply
        self.move_number = move_number
        self.material_white = material_white
        self.material_black = material_black
        self.piece_count_total = piece_count_total

class EvalResult:
    def __init__(self,
                 meta: PositionMeta,
                 eval_cp: int,
                 eval_norm: float,
                 mate_flag: int,
                 time_ms: float,
                 depth: int):
        self.meta = meta
        self.eval_cp = eval_cp
        self.eval_norm = eval_norm
        self.mate_flag = mate_flag
        self.time_ms = time_ms
        self.depth = depth

# ===============================
# Additional functions
# ===============================

def compute_material(board: chess.Board) -> Tuple[int, int, int, int, int, int, int]:
    material_white = 0
    material_black = 0

    q = r = b = n = 0

    for piece_type in PIECE_VALUES:
        white_squares = board.pieces(piece_type, chess.WHITE)
        for _sq in white_squares:
            material_white += PIECE_VALUES[piece_type]
            if piece_type == chess.QUEEN: q += 1
            elif piece_type == chess.ROOK: r += 1
            elif piece_type == chess.BISHOP: b += 1
            elif piece_type == chess.KNIGHT: n += 1
        black_squares = board.pieces(piece_type, chess.BLACK)
        for _sq in black_squares:
            material_black += PIECE_VALUES[piece_type]
            if piece_type == chess.QUEEN: q += 1
            elif piece_type == chess.ROOK: r += 1
            elif piece_type == chess.BISHOP: b += 1
            elif piece_type == chess.KNIGHT: n += 1

    pawns_white = len(board.pieces(chess.PAWN, chess.WHITE))
    pawns_black = len(board.pieces(chess.PAWN, chess.BLACK))

    piece_count_total = q + r + b + n + pawns_white + pawns_black
    return material_white, material_black, q, r, b, n, piece_count_total

def classify_phase(board: chess.Board) -> Tuple[str, int, float]:
    mw, mb, Q_total, R_total, B_total, N_total, piece_count_total = compute_material(board)
    phase_raw = 4 * Q_total + 2 * R_total + (B_total + N_total)
    phase_norm = phase_raw / 24.0
    endgame = False
    if Q_total == 0:
        endgame = True
    elif piece_count_total <= 10:
        endgame = True
    else:
        light_total = B_total + N_total
        has_heavy = (Q_total > 0 or R_total > 0)
        if not has_heavy and light_total <= 1:
            endgame = True
    if endgame:
        return "END", 1, phase_norm
    else:
        if phase_norm >= 0.66:
            return "OPEN", 0, phase_norm
        else:
            return "MID", 0, phase_norm

def move_is_capture_or_promotion(board: chess.Board, move: chess.Move) -> bool:
    if board.is_capture(move):
        return True
    if move.promotion is not None:
        return True
    return False

# ===============================
# Worker for stockfish engine
# ===============================

def engine_worker(task_queue: mp.Queue,
                  result_queue: mp.Queue,
                  engine_path: str,
                  time_limit: float,
                  hash_mb: int):

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)

        try:
            engine.configure({
                "Hash": hash_mb,
                "Threads": ENGINE_THREADS_PER_INSTANCE
            })
        except Exception:
            pass

        while True:
            item = task_queue.get()
            if item is None:
                break
            idx, meta = item
            board = chess.Board(meta.fen_full)
            start = time.time()
            try:
                info = engine.analyse(board, limit=chess.engine.Limit(time=time_limit))
            except Exception:
                elapsed_fail = (time.time() - start) * 1000
                result_queue.put((
                    idx,
                    EvalResult(
                        meta=meta,
                        eval_cp=0,
                        eval_norm=0.0,
                        mate_flag=0,
                        time_ms=elapsed_fail,
                        depth=-1
                    )
                ))
                continue
            elapsed = (time.time() - start) * 1000
            score = info.get("score")
            depth = info.get("depth", -1)
            eval_cp = 0
            mate_flag = 0
            if score is not None:
                pov = score.pov(board.turn)
                if pov.is_mate():
                    mate_flag = 1
                    mate_val = pov.mate()
                    if mate_val is not None and mate_val > 0:
                        eval_cp = MATE_CP
                    else:
                        eval_cp = -MATE_CP
                else:
                    raw = pov.score()
                    if raw is None:
                        raw = 0
                    if raw > CP_CLAMP:
                        raw = CP_CLAMP
                    if raw < -CP_CLAMP:
                        raw = -CP_CLAMP
                    eval_cp = raw
            if CP_CLAMP == 0:
                eval_norm = 0.0
            else:
                eval_norm = eval_cp / float(CP_CLAMP)

            result_queue.put((
                idx,
                EvalResult(
                    meta=meta,
                    eval_cp=eval_cp,
                    eval_norm=eval_norm,
                    mate_flag=mate_flag,
                    time_ms=elapsed,
                    depth=depth
                )
            ))

        try:
            engine.quit()
        except Exception:
            pass
    except KeyboardInterrupt:
        pass

# ===============================
# Batch evaluation of positions
# ===============================

def evaluate_positions(engine_path: str,
                       metas: List[PositionMeta],
                       time_limit: float,
                       hash_mb: int,
                       instances: int) -> List[EvalResult]:
    if not metas:
        return []
    task_q = mp.Queue()
    result_q = mp.Queue()
    procs = []
    for _ in range(instances):
        p = mp.Process(target=engine_worker,
                       args=(task_q, result_q, engine_path, time_limit, hash_mb))
        p.start()
        procs.append(p)
    for idx, m in enumerate(metas):
        task_q.put((idx, m))
    for _ in procs:
        task_q.put(None)
    results: List[Optional[EvalResult]] = [None] * len(metas)
    got = 0
    while got < len(metas):
        try:
            idx, res = result_q.get(timeout=5)
            results[idx] = res
            got += 1
        except Empty:
            continue
    for p in procs:
        p.join()

    final = []
    for r in results:
        if r is not None:
            final.append(r)
    return final

# ===============================
# DatasetBuilder class
# ===============================

class DatasetBuilder:
    def __init__(self, args):
        self.args = args

        self.target_open = args.target_open
        self.target_mid = args.target_mid
        self.target_end = args.target_end
        self.total_target = self.target_open + self.target_mid + self.target_end
        self.count_open = 0
        self.count_mid = 0
        self.count_end = 0
        self.total_done = 0
        self.seen_fen = set()
        self.pending: List[PositionMeta] = []
        self.current_shard_records: List[EvalResult] = []
        self.shard_index = 1
        self.shard_file = None
        self.shard_writer = None
        self.start_time = time.time()
        self.endgame_focus = False
        os.makedirs(args.out_dir, exist_ok=True)

    def _need_more(self, bucket: str) -> bool:
        if bucket == "OPEN":
            return self.count_open < self.target_open
        elif bucket == "MID":
            return self.count_mid < self.target_mid
        elif bucket == "END":
            return self.count_end < self.target_end
        else:
            return False

    def _plan_end_frac(self) -> float:
        return self.target_end / self.total_target

    def _current_end_frac(self) -> float:
        if self.total_done == 0:
            return 0.0
        return self.count_end / self.total_done

    def _maybe_toggle_endgame_focus(self):
        planned = self._plan_end_frac()
        current = self._current_end_frac()
        if (planned - current) > ENDGAME_FOCUS_TRIGGER_GAP:
            if not self.endgame_focus:
                print("[INFO] Włączam ENDGAME FOCUS.")
                self.endgame_focus = True
        else:
            if self.endgame_focus:
                print("[INFO] Wyłączam ENDGAME FOCUS.")
                self.endgame_focus = False

    def _open_shard(self):
        if self.shard_writer is None:
            name = f"positions_shard_{self.shard_index:04d}.csv"
            path = os.path.join(self.args.out_dir, name)
            f = open(path, "w", newline="", encoding="utf-8")
            self.shard_file = f
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            self.shard_writer = w
            print(f"[SHARD] Nowy shard: {path}")

    def _write_records(self, records: List[EvalResult]):
        self._open_shard()
        for r in records:
            self.shard_writer.writerow([
                r.meta.fen_full,
                r.eval_cp,
                f"{r.eval_norm:.5f}",
                r.meta.phase_bucket,
                r.meta.ply,
                r.meta.move_number,
                r.meta.material_white,
                r.meta.material_black,
                r.meta.piece_count_total,
                r.meta.endgame_flag,
                r.mate_flag,
                f"{r.time_ms:.2f}",
                r.depth
            ])

    def _flush_if_full(self):
        if len(self.current_shard_records) >= SHARD_SIZE:
            self._write_records(self.current_shard_records)
            self.current_shard_records = []
            if self.shard_file:
                self.shard_file.close()
            print(f"[SHARD] Zapisano shard #{self.shard_index} "
                  f"(open={self.count_open}, mid={self.count_mid}, end={self.count_end}, total={self.total_done})")
            self.shard_index += 1
            self.shard_writer = None
            self.shard_file = None

    def _save_pending_eval_batch(self):
        if len(self.pending) == 0:
            return
        metas = self.pending
        self.pending = []
        results = evaluate_positions(
            engine_path=self.args.stockfish,
            metas=metas,
            time_limit=self.args.engine_time,
            hash_mb=self.args.engine_hash,
            instances=self.args.engine_instances
        )
        self.current_shard_records.extend(results)
        self._flush_if_full()

    def _accept_meta(self, meta: PositionMeta):
        self.pending.append(meta)
        if meta.phase_bucket == "OPEN":
            self.count_open += 1
        elif meta.phase_bucket == "MID":
            self.count_mid += 1
        else:
            self.count_end += 1
        self.total_done += 1

    def _should_stop(self) -> bool:
        return self.total_done >= self.total_target

    def process_game(self, game: chess.pgn.Game):
        board = game.board()
        moves = list(game.mainline_moves())
        ply = 0
        accepted_this_game = 0
        i = 0
        while i < len(moves):
            move = moves[i]
            board.push(move)
            ply += 1
            if accepted_this_game >= MAX_POSITIONS_PER_GAME:
                i += 1
                continue
            is_cap_prom = move_is_capture_or_promotion(board, move)
            interval = ENDGAME_FOCUS_PLY_MOD if self.endgame_focus else 4
            take_it = (ply % interval == 0) or is_cap_prom
            if not take_it:
                i += 1
                continue
            bucket, end_flag, _ = classify_phase(board)
            if not self._need_more(bucket):
                i += 1
                continue
            fen_full = board.fen()
            parts = fen_full.split(" ")
            fen_dedup = " ".join(parts[:4])
            if fen_dedup in self.seen_fen:
                i += 1
                continue
            self.seen_fen.add(fen_dedup)
            mw, mb, _Q, _R, _B, _N, piece_count_total = compute_material(board)
            meta = PositionMeta(
                fen_full=fen_full,
                fen_dedup=fen_dedup,
                phase_bucket=bucket,
                endgame_flag=end_flag,
                ply=ply,
                move_number=board.fullmove_number,
                material_white=mw,
                material_black=mb,
                piece_count_total=piece_count_total
            )
            self._accept_meta(meta)
            accepted_this_game += 1
            if len(self.pending) >= self.args.batch_eval_size:
                self._save_pending_eval_batch()
                self._log_progress()
            if self._should_stop():
                break
            i += 1

    def _log_progress(self):
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            speed = 0
        else:
            speed = self.total_done / elapsed
        if self.total_done > 0:
            open_pct = self.count_open / self.total_done * 100
            mid_pct = self.count_mid / self.total_done * 100
            end_pct = self.count_end / self.total_done * 100
        else:
            open_pct = mid_pct = end_pct = 0.0
        print(f"[PROGRESS] total={self.total_done} "
              f"open={self.count_open}({open_pct:.1f}%) "
              f"mid={self.count_mid}({mid_pct:.1f}%) "
              f"end={self.count_end}({end_pct:.1f}%) "
              f"speed={speed:.1f} pos/s focus_end={self.endgame_focus}")

    def finalize(self):
        # Ostatnia ewaluacja
        self._save_pending_eval_batch()
        if self.current_shard_records:
            self._write_records(self.current_shard_records)
            self.current_shard_records = []
        if self.shard_file:
            self.shard_file.close()
        print("[DONE] Generowanie zakończone.")
        print(f"Final counts: open={self.count_open}, mid={self.count_mid}, end={self.count_end}, total={self.total_done}")

# ===============================
# PGN streaming
# ===============================

def shutil_which(cmd: str) -> Optional[str]:
    from shutil import which
    return which(cmd)

def open_pgn_stream(path: str):
    if path.endswith(".zst"):
        try:
            import zstandard as zstd  # type: ignore
            print("[INFO] Używam biblioteki zstandard.")
            dctx = zstd.ZstdDecompressor()
            fh = open(path, "rb")
            stream = dctx.stream_reader(fh)
            return io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")
        except ImportError:
            print("[INFO] Brak python zstandard -> używam zstd -dc.")
            if not shutil_which("zstd"):
                print("BŁĄD: brak 'zstd' w PATH i brak modułu 'zstandard'.")
                sys.exit(1)
            proc = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE)
            return io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="ignore")
    else:
        return open(path, "r", encoding="utf-8", errors="ignore")

# ===============================
# CLI Argument Parsing
# ===============================

def parse_args():
    p = argparse.ArgumentParser(description="Generator datasetu szachowego z PGN + Stockfish eval.")
    p.add_argument("--pgn",default="C:/Users/PIOTR KAPTUR/Desktop/Nowy folder (4)/lichess_db_standard_rated_2025-07.pgn.zst")
    p.add_argument("--stockfish",default="C:/Users/PIOTR KAPTUR/Desktop/Nowy folder (4)/stockfish/stockfish-windows-x86-64-avx2.exe")
    p.add_argument("--out-dir",default="C:/Users/PIOTR KAPTUR/Desktop/Nowy folder (4)/shards_v1")
    p.add_argument("--target-open", type=int, default=DEFAULT_OPEN_TARGET)
    p.add_argument("--target-mid", type=int, default=DEFAULT_MID_TARGET)
    p.add_argument("--target-end", type=int, default=DEFAULT_END_TARGET)
    p.add_argument("--batch-eval-size", type=int, default=BATCH_EVAL_SIZE)
    p.add_argument("--engine-instances", type=int, default=ENGINE_INSTANCES)
    p.add_argument("--engine-time", type=float, default=ENGINE_TIME_SEC)
    p.add_argument("--engine-hash", type=int, default=ENGINE_HASH_MB)
    p.add_argument("--max-games", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=2000)
    p.add_argument("--no-endgame-focus", action="store_true")
    p.add_argument("--target-total", type=int, default=0,)
    return p.parse_args()

def adjust_targets(args):
    if args.target_total and args.target_total > 0:
        total = args.target_total
        o = int(total * 0.30)
        m = int(total * 0.30)
        e = total - o - m
        args.target_open = o
        args.target_mid = m
        args.target_end = e

# ===============================
# Main
# ===============================

def main():
    args = parse_args()
    adjust_targets(args)
    print("=== Configuration ===")
    print("PGN:", args.pgn)
    print("Stockfish:", args.stockfish)
    print("Output dir:", args.out_dir)
    print(f"Targets: OPEN={args.target_open}, MID={args.target_mid}, END={args.target_end}, SUM={args.target_open + args.target_mid + args.target_end}")
    print(f"Batch eval size={args.batch_eval_size}, Engine instances={args.engine_instances}, Time per pos={args.engine_time}s")
    print("====================")

    builder = DatasetBuilder(args)

    def on_sigint(_sig, _frm):
        print("\n[INTERRUPT] Zatrzymano – kończę i zapisuję...")
        builder.finalize()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)

    stream = open_pgn_stream(args.pgn)
    game_count = 0

    while True:
        if builder._should_stop():
            break
        try:
            game = chess.pgn.read_game(stream)
        except Exception as e:
            print("[WARN] Błąd parsowania partii:", e)
            continue
        if game is None:
            break
        game_count += 1
        builder._maybe_toggle_endgame_focus()
        if args.no_endgame_focus:
            builder.endgame_focus = False
        builder.process_game(game)
        if game_count % args.progress_every == 0:
            builder._log_progress()
        if args.max_games and game_count >= args.max_games:
            print("[INFO] Osiągnięto limit max_games.")
            break

    builder.finalize()
    print("[INFO] Zakończono main().")

if __name__ == "__main__":
    main()