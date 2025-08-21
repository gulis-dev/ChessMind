import os
import glob
import random
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG (hardcoded paths)
# =========================
SHARDS_DIR = "../../data/shards_v1"
FILE_PATTERN = "positions_shard_*.csv"
CHUNK_SIZE = 120000
MAX_FILES = 0              # 0 = all
MAX_EVAL_SAMPLE = 300000   # how many eval_cp to keep for hist
MAX_TIME_SAMPLE = 300000
MAX_DEPTH_SAMPLE = 300000
PHASES = ["OPEN", "MID", "END"]
CLAMP_CP = 2000
OUTPUT_DIR = "../../plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    pattern = os.path.join(SHARDS_DIR, FILE_PATTERN)
    files = sorted(glob.glob(pattern))
    if not files:
        print("No shard files found.")
        return
    if MAX_FILES > 0:
        files = files[:MAX_FILES]

    phase_counts = {p: 0 for p in PHASES}
    eval_sample = []
    time_sample = []
    depth_sample = []
    clamp_count = 0
    total_rows = 0

    # For small extra plot: material difference distribution
    material_diff_sample = []

    print(f"Reading {len(files)} shard files...")

    for idx, path in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {os.path.basename(path)}")
        try:
            for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
                rows = len(chunk)
                total_rows += rows

                # Phase counts
                for p in PHASES:
                    phase_counts[p] += (chunk["phase_bucket"] == p).sum()

                # Clamp
                clamp_count += (chunk["eval_cp"].abs() == CLAMP_CP).sum()

                # Samples (truncate to limit)
                if len(eval_sample) < MAX_EVAL_SAMPLE:
                    need = MAX_EVAL_SAMPLE - len(eval_sample)
                    eval_sample.extend(chunk["eval_cp"].tolist()[:need])

                if len(time_sample) < MAX_TIME_SAMPLE:
                    need = MAX_TIME_SAMPLE - len(time_sample)
                    time_sample.extend(chunk["time_ms"].tolist()[:need])

                if len(depth_sample) < MAX_DEPTH_SAMPLE:
                    need = MAX_DEPTH_SAMPLE - len(depth_sample)
                    depth_sample.extend(chunk["depth"].tolist()[:need])

                # material diff = material_white - material_black
                if "material_white" in chunk.columns and "material_black" in chunk.columns:
                    diffs = (chunk["material_white"] - chunk["material_black"]).tolist()
                    if len(material_diff_sample) < 200_000:
                        need = 200_000 - len(material_diff_sample)
                        material_diff_sample.extend(diffs[:need])

        except Exception as e:
            print("Error reading file:", path, e)

    # Convert samples to numpy arrays
    eval_arr = np.array(eval_sample)
    time_arr = np.array(time_sample)
    depth_arr = np.array(depth_sample)
    depth_arr_clean = depth_arr[depth_arr != -1]

    # =========================
    # Phase distribution bar
    # =========================
    plt.figure(figsize=(5,4))
    counts = [phase_counts[p] for p in PHASES]
    total = sum(counts) if sum(counts) else 1
    plt.bar(PHASES, counts, color=["#4f81bd","#c05020","#9bbb59"])
    for i, c in enumerate(counts):
        pct = c / total * 100
        plt.text(i, c, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.title("Phase distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "phase_distribution.png"))
    plt.close()

    # =========================
    # Eval histogram
    # =========================
    if len(eval_arr) > 0:
        plt.figure(figsize=(6,4))
        bins = 80
        plt.hist(eval_arr, bins=bins, range=(-CLAMP_CP, CLAMP_CP), color="#5555aa", alpha=0.85)
        clamp_ratio = (np.abs(eval_arr) == CLAMP_CP).mean() * 100
        plt.title(f"eval_cp histogram (sample={len(eval_arr)}, clamp%={clamp_ratio:.2f})")
        plt.xlabel("eval_cp (centipawns)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "eval_hist.png"))
        plt.close()

        # Cumulative distribution
        plt.figure(figsize=(6,4))
        sorted_eval = np.sort(eval_arr)
        y = np.linspace(0,1,len(sorted_eval))
        plt.plot(sorted_eval, y, color="#2f6f9f")
        plt.title("Cumulative distribution of eval_cp")
        plt.xlabel("eval_cp")
        plt.ylabel("CDF")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "eval_cdf.png"))
        plt.close()

    # =========================
    # Time distributions
    # =========================
    if len(time_arr) > 0:
        plt.figure(figsize=(6,4))
        plt.hist(time_arr, bins=60, color="#888888")
        plt.title(f"time_ms histogram (sample={len(time_arr)})")
        plt.xlabel("time_ms")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "time_hist.png"))
        plt.close()

        # Log scale version (avoid log(0) by adding small epsilon)
        positive_times = time_arr[time_arr > 0]
        if len(positive_times) > 0:
            plt.figure(figsize=(6,4))
            plt.hist(np.log10(positive_times), bins=60, color="#aa8844")
            plt.title("log10(time_ms) histogram")
            plt.xlabel("log10(time_ms)")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "time_hist_log.png"))
            plt.close()

    # =========================
    # Depth histogram (clean)
    # =========================
    if len(depth_arr_clean) > 0:
        plt.figure(figsize=(6,4))
        plt.hist(depth_arr_clean, bins=40, color="#2d8f4e")
        pct_bad = (len(depth_arr) - len(depth_arr_clean)) / len(depth_arr) * 100 if len(depth_arr)>0 else 0
        plt.title(f"depth histogram (removed -1: {pct_bad:.2f}% bad)")
        plt.xlabel("depth")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "depth_hist.png"))
        plt.close()

    # =========================
    # Material diff histogram
    # =========================
    if material_diff_sample:
        diff_arr = np.array(material_diff_sample)
        plt.figure(figsize=(6,4))
        plt.hist(diff_arr, bins=50, color="#6a3d9a")
        plt.title("Material difference (white - black)")
        plt.xlabel("Material diff")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "material_diff_hist.png"))
        plt.close()

    # =========================
    # Simple text summary file
    # =========================
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Total rows (approx, sum of chunks): {}\n".format(total_rows))
        for p in PHASES:
            f.write(f"Phase {p}: {phase_counts[p]}\n")
        if len(eval_arr) > 0:
            clamp_ratio = (np.abs(eval_arr) == CLAMP_CP).mean() * 100
            f.write(f"Eval sample size: {len(eval_arr)} clamp_ratio%: {clamp_ratio:.2f}\n")
            f.write("Eval stats (sample): mean={:.2f} std={:.2f} p50={:.2f} p90={:.2f} p99={:.2f}\n".format(
                eval_arr.mean(), eval_arr.std(), np.percentile(eval_arr,50),
                np.percentile(eval_arr,90), np.percentile(eval_arr,99)
            ))
        if len(time_arr) > 0:
            f.write("Time_ms stats (sample): mean={:.3f} p50={:.3f} p90={:.3f} p99={:.3f}\n".format(
                time_arr.mean(), np.percentile(time_arr,50),
                np.percentile(time_arr,90), np.percentile(time_arr,99)
            ))
        if len(depth_arr_clean) > 0:
            f.write("Depth stats (clean): mean={:.2f} p50={:.2f} p90={:.2f} p99={:.2f}\n".format(
                depth_arr_clean.mean(), np.percentile(depth_arr_clean,50),
                np.percentile(depth_arr_clean,90), np.percentile(depth_arr_clean,99)
            ))
        f.write("Plots generated in directory: {}\n".format(OUTPUT_DIR))

    print("Done. Plots saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()