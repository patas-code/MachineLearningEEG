"""
EEG Tabular Data Classification with 1D-CNN
Dataset : AD_all_patients.csv
Features: 16 EEG channels (Fp1, Fp2, F7, ..., P4)
Label   : status  (0 = healthy, 1 = AD)
GPU     : NVIDIA GeForce GTX 1650 Super (CUDA)

NOTE (Windows): DataLoader spawns child processes for num_workers > 0.
On Windows this requires all executable code to live inside
    if __name__ == '__main__':
This file wraps everything in main() called from that guard.
num_workers is set to 0 (main-process loading) which is the safest
default on Windows and still plenty fast for this dataset.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
# CONFIG  (module-level constants are fine)
# ─────────────────────────────────────────────
CSV_PATH     = "AD_all_patients.csv"
TARGET_COL   = "status"
BATCH_SIZE   = 2048
EPOCHS       = 30
LR           = 1e-3
WEIGHT_DECAY = 1e-4
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15
SEED         = 42
SAVE_PATH    = "eeg_cnn_best.pt"
NUM_WORKERS  = 0          # 0 = main-process loading; avoids Windows spawn error


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class EEGDataset(Dataset):
    """
    Converts a (N, C) array to (N, 1, C) tensors so each tabular row
    looks like a single-channel 1-D signal for Conv1d.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class EEG_CNN(nn.Module):
    """
    1D-CNN for tabular EEG classification.

    Input  : (B, 1, 16)
    Block 1: Conv1d 1→32,   k=3, pad=1, BN, ReLU, Dropout
    Block 2: Conv1d 32→64,  k=3, pad=1, BN, ReLU, Dropout
    Block 3: Conv1d 64→128, k=3, pad=1, BN, ReLU
    GAP    : AdaptiveAvgPool1d(1) → (B, 128)
    Head   : Linear 128→64, ReLU, Dropout, Linear 64→2
    """
    def __init__(self, num_features: int = 16, num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        def _block(in_ch, out_ch, drop=dropout):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(drop),
            )

        self.conv1 = _block(1,  32)
        self.conv2 = _block(32, 64)
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        return self.head(x)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def run_epoch(loader, model, criterion, device, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(1) == y_batch).sum().item()
            total      += len(y_batch)

    return total_loss / total, correct / total


def predict(raw_samples: np.ndarray, model: nn.Module,
            scaler: StandardScaler, device: torch.device) -> dict:
    """
    Predict class labels and P(AD) for raw (unscaled) EEG rows.

    Parameters
    ----------
    raw_samples : np.ndarray  shape (N, 16)
    model       : trained EEG_CNN
    scaler      : fitted StandardScaler
    device      : torch.device

    Returns
    -------
    {'labels': int array, 'probabilities': float array P(class=1)}
    """
    model.eval()
    X_s    = scaler.transform(raw_samples.astype(np.float32))
    tensor = torch.tensor(X_s, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        labels = logits.argmax(1).cpu().numpy()
    return {"labels": labels, "probabilities": probs}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── 1. Device ────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n\n[ERROR] CUDA not available — PyTorch cannot see your GPU.\n"
            "You likely installed the CPU-only build.  Fix:\n"
            "  1.  pip uninstall torch torchvision torchaudio -y\n"
            "  2.  Check driver:  nvidia-smi  (note the CUDA Version)\n"
            "  3a. CUDA 12.1+:  pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu121\n"
            "  3b. CUDA 11.8:   pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu118\n"
        )

    device = torch.device("cuda")
    print(f"[Device] {torch.cuda.get_device_name(0)}")
    print(f"         VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ── 2. Load & split data ─────────────────────────────────────────────
    print("\n[Data] Loading CSV …")
    df = pd.read_csv(CSV_PATH)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int64)

    print(f"       Samples  : {len(X):,}")
    print(f"       Features : {feature_cols}")
    print(f"       Classes  : 0={np.sum(y==0):,}  1={np.sum(y==1):,}")

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=SEED)

    print(f"       Train : {len(X_train):,}  |  Val : {len(X_val):,}  |  Test : {len(X_test):,}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ── 3. DataLoaders ───────────────────────────────────────────────────
    # num_workers=0  → no child processes spawned; Windows-safe.
    # pin_memory=True → faster CPU→GPU DMA transfers (valid since CUDA confirmed above).
    train_loader = DataLoader(EEGDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(EEGDataset(X_val,   y_val),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(EEGDataset(X_test,  y_test),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ── 4. Model ─────────────────────────────────────────────────────────
    model = EEG_CNN(num_features=len(feature_cols)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model]\n{model}")
    print(f"        Trainable params: {n_params:,}")

    # ── 5. Loss / Optimiser / Scheduler ──────────────────────────────────
    counts  = np.bincount(y_train)
    weights = torch.tensor(len(y_train) / (len(counts) * counts),
                           dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── 6. Training loop ─────────────────────────────────────────────────
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    print(f"\n[Training] {EPOCHS} epochs\n{'─'*65}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_loader, model, criterion, device,
                                    optimizer=optimizer, train=True)
        vl_loss, vl_acc = run_epoch(val_loader, model, criterion, device,
                                    train=False)
        scheduler.step()

        for key, val in zip(["train_loss","val_loss","train_acc","val_acc"],
                             [tr_loss, vl_loss, tr_acc, vl_acc]):
            history[key].append(val)

        saved = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), SAVE_PATH)
            saved = "  ✔ saved"

        print(f"Epoch {epoch:>3}/{EPOCHS}  "
              f"Train loss={tr_loss:.4f} acc={tr_acc:.4f}  |  "
              f"Val   loss={vl_loss:.4f} acc={vl_acc:.4f}  "
              f"[{time.time()-t0:.1f}s]{saved}")

    # ── 7. Test evaluation ───────────────────────────────────────────────
    print(f"\n[Test] Loading best checkpoint ({SAVE_PATH}) …")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model.eval()

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.to(device))
            all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    print(f"\n{'═'*60}")
    print(f"  Test Accuracy : {(all_preds == all_labels).mean():.4f}")
    print(f"  ROC-AUC Score : {roc_auc_score(all_labels, all_probs):.4f}")
    print(f"{'═'*60}\n")
    print(classification_report(all_labels, all_preds,
                                 target_names=["Healthy (0)", "AD (1)"]))

    # ── 8. Training curves ───────────────────────────────────────────────
    ep = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, m in zip(axes, ["loss", "acc"]):
        ax.plot(ep, history[f"train_{m}"], label="Train")
        ax.plot(ep, history[f"val_{m}"],   label="Val")
        ax.set_title(m.capitalize()); ax.set_xlabel("Epoch"); ax.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150); plt.close()
    print("[Plot] Saved training_curves.png")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(all_labels, all_preds),
                annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0","Pred 1"],
                yticklabels=["True 0","True 1"], ax=ax)
    ax.set_title("Confusion Matrix – Test Set")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150); plt.close()
    print("[Plot] Saved confusion_matrix.png")

    # ── 9. Predictions demo ──────────────────────────────────────────────
    demo_idx = np.random.choice(len(X_test), size=5, replace=False)
    demo_raw = scaler.inverse_transform(X_test[demo_idx])
    result   = predict(demo_raw, model, scaler, device)

    print("\n[Predict] 5 random test samples:")
    print(f"{'Sample':<8} {'True':>6} {'Pred':>6} {'P(AD)':>8}")
    print("─" * 32)
    for i, idx in enumerate(demo_idx):
        print(f"{i:<8} {y_test[idx]:>6} {result['labels'][i]:>6} "
              f"{result['probabilities'][i]:>8.4f}")

    # ── 10. Feature importance — gradient saliency ───────────────────────
    print("\n[Feature Importance] Computing input-gradient saliency …")

    # Overall (all test samples combined)
    model.eval()
    imp_accum = np.zeros(len(feature_cols), dtype=np.float64)
    n_total   = 0

    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device).requires_grad_(True)
        y_batch = y_batch.to(device)
        criterion(model(X_batch), y_batch).backward()
        imp_accum += X_batch.grad.detach().abs().squeeze(1).sum(dim=0).cpu().numpy()
        n_total   += len(y_batch)

    imp      = imp_accum / n_total
    imp_norm = (imp - imp.min()) / (imp.max() - imp.min() + 1e-12)

    order        = np.argsort(imp_norm)[::-1]
    sorted_names = [feature_cols[i] for i in order]
    sorted_imp   = imp_norm[order]

    print(f"\n{'Rank':<6} {'Channel':<8} {'Importance':>12}")
    print("─" * 30)
    for rank, (name, val) in enumerate(zip(sorted_names, sorted_imp), 1):
        print(f"{rank:<6} {name:<8} {val:>10.4f}  {'█' * int(val * 30)}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(sorted_names)))
    bars   = ax.barh(sorted_names[::-1], sorted_imp[::-1],
                     color=colors[::-1], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, sorted_imp[::-1]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8)
    ax.set_xlabel("Normalised Mean |Gradient|", fontsize=11)
    ax.set_title("EEG Channel Feature Importance\n(Gradient Saliency — test set)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150); plt.close()
    print("[Plot] Saved feature_importance.png")

    # Per-class saliency heatmap
    print("[Feature Importance] Computing per-class saliency …")
    cls_imp = {0: np.zeros(len(feature_cols)), 1: np.zeros(len(feature_cols))}
    cls_cnt = {0: 0, 1: 0}

    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device).requires_grad_(True)
        y_batch = y_batch.to(device)
        criterion(model(X_batch), y_batch).backward()
        grads     = X_batch.grad.detach().abs().squeeze(1).cpu().numpy()
        labels_np = y_batch.cpu().numpy()
        for cls in [0, 1]:
            mask = labels_np == cls
            if mask.any():
                cls_imp[cls] += grads[mask].sum(axis=0)
                cls_cnt[cls] += mask.sum()

    heatmap = np.stack([cls_imp[c] / (cls_cnt[c] + 1e-12) for c in [0, 1]])
    heatmap = (heatmap - heatmap.min(axis=1, keepdims=True)) / (
        heatmap.max(axis=1, keepdims=True) - heatmap.min(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=(12, 3))
    sns.heatmap(heatmap, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=feature_cols,
                yticklabels=["Healthy (0)", "AD (1)"],
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "Normalised Saliency"})
    ax.set_title("Per-Class EEG Channel Saliency", fontsize=13, fontweight="bold")
    ax.set_xlabel("EEG Channel")
    plt.tight_layout()
    plt.savefig("feature_importance_per_class.png", dpi=150); plt.close()
    print("[Plot] Saved feature_importance_per_class.png")

    print("\n[Done] All outputs saved.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# On Windows, Python uses "spawn" to create new processes. Every worker
# process imports this module from scratch, so any code at module level
# would run again inside each worker — causing the bootstrapping crash.
# Wrapping execution in  if __name__ == '__main__'  ensures that only
# the original process runs main(); worker re-imports stop here safely.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()