import os
import glob
import math
import random
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===============================
# KONFIG
# ===============================
SHARDS_DIR          = "../../data/shards_v1_shuffled"
SHARD_PATTERN       = "positions_shard_*.csv"
MAX_SHARDS          = 0

OUTPUT_DIR          = "outputs_shards"
OUTPUT_MODEL_NAME   = "model_v1.pt"

EPOCHS              = 20
BATCH_SIZE          = 512
LEARNING_RATE       = 1e-3
HIDDEN_UNITS        = 128
CLAMP_CP            = 2000

VAL_FRACTION        = 0.05
VAL_MAX_ROWS        = 100000
RANDOM_SEED         = 73
EARLY_STOP_PATIENCE = 8
PRINT_SAMPLE_PRED   = True
SAVE_FIG_DPI        = 300
SCATTER_MAX_POINTS  = 3000

# ===============================
# SEED
# ===============================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(RANDOM_SEED)

# ===============================
# Features
# ===============================
PIECES = ['P','N','B','R','Q','K','p','n','b','r','q','k']

def fen_to_counts(fen):
    parts = fen.split()
    if not parts:
        return [0]*len(PIECES)
    board = parts[0]
    counts = dict.fromkeys(PIECES, 0)
    for ch in board:
        if ch in counts: counts[ch] += 1
    return [counts[p] for p in PIECES]

def side_to_move_feature(fen):
    parts = fen.split()
    if len(parts) < 2: return 0
    return 1 if parts[1] == 'w' else 0

def simple_phase_one_hot(phase_bucket):
    if phase_bucket == "OPEN": return [1,0,0]
    if phase_bucket == "MID":  return [0,1,0]
    if phase_bucket == "END":  return [0,0,1]
    return [0,0,0]

def build_features(df, use_material=True):
    feats = []
    for _, row in df.iterrows():
        fen = row.get("fen","")
        phase = row.get("phase_bucket","")
        depth = row.get("depth",0)
        time_ms = row.get("time_ms",0)

        piece_counts = fen_to_counts(fen)
        stm          = [side_to_move_feature(fen)]
        phase_oh     = simple_phase_one_hot(phase)
        depth_feat   = [float(depth)]
        time_feat    = [float(time_ms)]

        material_feats = []
        if "material_white" in df.columns and "material_black" in df.columns and use_material:
            mw = row.get("material_white", np.nan)
            mb = row.get("material_black", np.nan)
            if pd.isna(mw) or pd.isna(mb):
                material_feats = [0.0,0.0,0.0]
            else:
                material_feats = [float(mw), float(mb), float(mw)-float(mb)]
        else:
            material_feats = [0.0,0.0,0.0]

        feats.append(piece_counts + stm + phase_oh + depth_feat + time_feat + material_feats)
    return np.array(feats, dtype=np.float32)

# ===============================
# DATASET
# ===============================
class PositionsDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.X = build_features(self.df)
        evals = self.df["eval_cp"].clip(-CLAMP_CP, CLAMP_CP).values.astype(np.float32)
        self.y = evals / CLAMP_CP
    def __len__(self): return len(self.df)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ===============================
# MODEL
# ===============================
class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

# ===============================
# PĘTLE
# ===============================
def train_one_shard(model, shard_df, optimizer, loss_fn, device, batch_size):
    ds = PositionsDataset(shard_df)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.train()
    tot = 0.0
    n = 0
    for X,y in loader:
        X = X.to(device); y = y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred,y)
        loss.backward()
        optimizer.step()
        tot += loss.item() * len(X)
        n += len(X)
    return tot / max(1,n)

def eval_full(model, val_df, loss_fn, device, batch_size):
    ds = PositionsDataset(val_df)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    tot_loss = 0.0
    tot_mae_cp = 0.0
    with torch.no_grad():
        for X,y in loader:
            X = X.to(device); y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred,y)
            tot_loss += loss.item() * len(X)
            mae_cp = (pred - y).abs().mean().item() * CLAMP_CP
            tot_mae_cp += mae_cp * len(X)
    size = len(val_df)
    return tot_loss / size, tot_mae_cp / size

# ===============================
# WYKRESY
# ===============================
def make_plots(train_losses, val_losses, val_maes, val_df, model, device):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1 loss curve
    plt.figure(figsize=(6,4))
    plt.plot(train_losses,label="train_loss")
    plt.plot(val_losses,label="val_loss")
    plt.xlabel("Epoka"); plt.ylabel("MSE (norm)")
    plt.title("Train / Val Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"loss_curve.png"), dpi=SAVE_FIG_DPI)
    plt.close()

    # 2 mae
    plt.figure(figsize=(6,4))
    plt.plot(val_maes,label="val_MAE_cp", color="orange")
    plt.xlabel("Epoka"); plt.ylabel("MAE cp")
    plt.title("Val MAE"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"val_mae_curve.png"), dpi=SAVE_FIG_DPI)
    plt.close()

    # 3 pred vs target
    model.eval()
    with torch.no_grad():
        Xv = build_features(val_df)
        preds_norm = model(torch.tensor(Xv, dtype=torch.float32, device=device)).cpu().numpy()
    preds_cp = preds_norm * CLAMP_CP
    targets_cp = val_df["eval_cp"].clip(-CLAMP_CP, CLAMP_CP).values
    if len(preds_cp) > SCATTER_MAX_POINTS:
        idx = np.random.choice(len(preds_cp), SCATTER_MAX_POINTS, replace=False)
        p = preds_cp[idx]; t = targets_cp[idx]
    else:
        p = preds_cp; t = targets_cp
    plt.figure(figsize=(5,5))
    plt.scatter(t,p,s=8,alpha=0.5)
    lo = min(t.min(), p.min()); hi = max(t.max(), p.max())
    plt.plot([lo,hi],[lo,hi],'r--',linewidth=1)
    plt.xlabel("target cp"); plt.ylabel("pred cp"); plt.title("Pred vs Target (val)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"pred_vs_target.png"), dpi=SAVE_FIG_DPI)
    plt.close()

    # 4 residual hist
    resid = p - t
    plt.figure(figsize=(6,4))
    plt.hist(resid, bins=60, color="steelblue", alpha=0.85)
    plt.xlabel("residual (pred-target) cp"); plt.ylabel("count")
    plt.title("Residuals (val subset)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"residual_hist.png"), dpi=SAVE_FIG_DPI)
    plt.close()

# ===============================
# PODSUMOWANIE
# ===============================
def print_summary(train_losses, val_losses, val_maes):
    best_ep = int(np.argmin(val_losses)) + 1
    print("\n=== PODSUMOWANIE ===")
    print("Epoki:", len(train_losses))
    print("Najlepsza epoka:", best_ep)
    print(f"Best val_loss: {val_losses[best_ep-1]:.6f}")
    print(f"Best val_MAE_cp: {val_maes[best_ep-1]:.2f}")
    print(f"Średni train_loss: {np.mean(train_losses):.6f}")
    print(f"Średni val_loss: {np.mean(val_losses):.6f}")
    print("====================\n")

# ===============================
# WALIDACJA – budujemy w 1. epoce
# ===============================
def sample_val_rows(df, fraction, current_val_rows, max_rows):
    if fraction <= 0:
        return pd.DataFrame()
    # losowa maska
    mask = np.random.rand(len(df)) < fraction
    sample = df[mask]
    # nie przekraczaj limitu
    free = max_rows - current_val_rows
    if free <= 0:
        return pd.DataFrame()
    if len(sample) > free:
        sample = sample.sample(n=free, random_state=RANDOM_SEED)
    return sample

# ===============================
# GŁÓWNY
# ===============================
def main():
    print("== START (shardy) ==")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pattern = os.path.join(SHARDS_DIR, SHARD_PATTERN)
    shard_files = sorted(glob.glob(pattern))
    if MAX_SHARDS > 0:
        shard_files = shard_files[:MAX_SHARDS]
    print(f"Znaleziono shardów: {len(shard_files)}")
    if not shard_files:
        print("Brak plików shardów.")
        return

    first_cols = pd.read_csv(shard_files[0], nrows=1).columns
    if "fen" not in first_cols or "eval_cp" not in first_cols:
        print("W shardach brakuje kolumn fen / eval_cp.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Urządzenie:", device)

    model = None
    optimizer = None
    loss_fn = nn.MSELoss()

    val_df_total = pd.DataFrame()
    train_losses = []
    val_losses = []
    val_maes = []
    best_val_loss = math.inf
    patience_cnt = 0

    total_start = time.time()

    for epoch in range(1, EPOCHS+1):
        print(f"\n=== Epoka {epoch}/{EPOCHS} ===")
        ep_start = time.time()
        random.shuffle(shard_files)

        epoch_train_loss_sum = 0.0
        epoch_train_count = 0

        for shard_path in tqdm(shard_files, desc="Shards", leave=False):
            try:
                shard_df = pd.read_csv(shard_path)
            except Exception as e:
                print("Błąd czytania", shard_path, e)
                continue
            shard_df = shard_df.dropna(subset=["eval_cp"])
            if shard_df.empty:
                continue
            if epoch == 1 and VAL_FRACTION > 0 and len(val_df_total) < VAL_MAX_ROWS:
                val_part = sample_val_rows(shard_df, VAL_FRACTION, len(val_df_total), VAL_MAX_ROWS)
                if not val_part.empty:
                    val_df_total = pd.concat([val_df_total, val_part], ignore_index=True)
                if not val_part.empty:
                    shard_df = shard_df.drop(val_part.index)

            if shard_df.empty:
                continue

            local_ds = PositionsDataset(shard_df)
            if model is None:
                in_dim = local_ds.X.shape[1]
                model = SimpleMLP(in_dim, hidden=HIDDEN_UNITS).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                print("Utworzono model. in_dim =", in_dim)

            local_loader = DataLoader(local_ds, batch_size=BATCH_SIZE, shuffle=True)
            model.train()
            shard_loss_sum = 0.0
            shard_n = 0
            for X,y in local_loader:
                X = X.to(device); y = y.to(device)
                optimizer.zero_grad()
                pred = model(X)
                loss = loss_fn(pred,y)
                loss.backward()
                optimizer.step()
                shard_loss_sum += loss.item() * len(X)
                shard_n += len(X)
            shard_loss = shard_loss_sum / shard_n
            epoch_train_loss_sum += shard_loss_sum
            epoch_train_count += shard_n

        if epoch_train_count == 0:
            print("Brak danych treningowych w tej epoce.")
            break

        train_loss_epoch = epoch_train_loss_sum / epoch_train_count
        train_losses.append(train_loss_epoch)

        if val_df_total.empty:
            print("UWAGA: walidacja pusta – zwiększ VAL_FRACTION lub sprawdź dane.")
            val_loss_epoch = float('nan')
            val_mae_epoch = float('nan')
        else:
            val_loss_epoch, val_mae_epoch = eval_full(model, val_df_total, loss_fn, device, BATCH_SIZE)

        val_losses.append(val_loss_epoch)
        val_maes.append(val_mae_epoch)

        ep_time = time.time() - ep_start
        print(f"[Epoka {epoch}] train_loss={train_loss_epoch:.6f}  val_loss={val_loss_epoch:.6f}  val_MAE_cp={val_mae_epoch:.2f}  (czas {ep_time:.1f}s)  val_rows={len(val_df_total)}")

        if not math.isnan(val_loss_epoch):
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                patience_cnt = 0
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, OUTPUT_MODEL_NAME))
                print("  -> Zapisano nowy najlepszy model.")
            else:
                if EARLY_STOP_PATIENCE > 0:
                    patience_cnt += 1
                    if patience_cnt >= EARLY_STOP_PATIENCE:
                        print("Early stopping (brak poprawy).")
                        break

    total_time = time.time() - total_start
    print(f"\nCałkowity czas treningu: {total_time:.1f}s")

    if model is not None and not val_df_total.empty:
        print("Generowanie wykresów...")
        make_plots(train_losses, val_losses, val_maes, val_df_total, model, device)
        print("Wykresy zapisane w:", OUTPUT_DIR)

    print_summary(train_losses, val_losses, val_maes)
    model_path = os.path.join(OUTPUT_DIR, OUTPUT_MODEL_NAME)
    if PRINT_SAMPLE_PRED and model is not None and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        try:
            sample_df = pd.read_csv(shard_files[0]).dropna(subset=["eval_cp"])
            sample_df = sample_df.head(5)
            Xs = build_features(sample_df)
            with torch.no_grad():
                pred_norm = model(torch.tensor(Xs, dtype=torch.float32, device=device)).cpu().numpy()
            pred_cp = pred_norm * CLAMP_CP
            print("Przykładowe predykcje (cp):", np.round(pred_cp,1))
        except Exception as e:
            print("Błąd przy sample pred:", e)

    print("== KONIEC ==")

if __name__ == "__main__":
    main()