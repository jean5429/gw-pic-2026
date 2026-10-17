"""Quantifica os tres problemas da linha v9 para a errata do Relatorio V2.

Roda com cwd = raiz do projeto (dados em data/ e outputs/gwtc/qgrams) e o codigo da
branch v9 em V9 (worktree).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V9 = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.insert(0, str(V9 / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms  # noqa: E402

from gwproject.config import load_toml  # noqa: E402
from gwproject.gravity_spy import (  # noqa: E402
    GravitySpyImageDataset,
    attach_image_paths,
    index_spectrograms,
    official_splits,
)
from gwproject.qgram_inference import CatalogQgramDataset, load_gravity_spy_checkpoint  # noqa: E402
from gwproject.research.uncertainty import (  # noqa: E402
    energy_score,
    fit_mahalanobis_reference,
    mahalanobis_score,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [11, 23, 42, 71, 101]
ckpt = lambda s: V9 / f"outputs/gravity_spy_harmonized/seed_{s}_cuda_balanced/best_model.pt"  # noqa: E731

metadata = pd.read_csv("data/gravity_spy/trainingset_v1d1_metadata.csv")
metadata = attach_image_paths(
    metadata, index_spectrograms(Path("data/gravity_spy/trainingset_harmonized_20_512"), "1.0")
)
splits = official_splits(metadata)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((140, 170)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def pixel_hash(path):
    return hashlib.md5(np.asarray(Image.open(path).convert("L")).tobytes()).hexdigest()


# Painel vazio = grupo de imagens identicas mais numeroso
all_paths = pd.concat([splits[k].assign(split=k) for k in ("train", "val", "test")], ignore_index=True)
all_paths["hash"] = [pixel_hash(p) for p in all_paths["path"]]
blank_hash = all_paths["hash"].value_counts().index[0]
blank_count = int((all_paths["hash"] == blank_hash).sum())


def run(model, frame, classes):
    cls = {c: i for i, c in enumerate(classes)}
    frame = frame[frame["label"].isin(classes)].copy()
    frame["class_idx"] = frame["label"].map(cls).astype(int)
    loader = DataLoader(GravitySpyImageDataset(frame, transform), batch_size=64, shuffle=False)
    feats, logits = [], []
    fc = model.fc
    model.fc = torch.nn.Identity()
    with torch.no_grad():
        for images, _ in loader:
            f = model(images.to(device)).float()
            feats.append(f.cpu().numpy())
            logits.append(fc(f).float().cpu().numpy())
    model.fc = fc
    return frame, np.concatenate(feats), np.concatenate(logits)


result = {"blank_count": blank_count, "seeds": {}}

# (1) metricas de teste com e sem o painel vazio, 5 seeds
for seed in SEEDS:
    model, classes, _ = load_gravity_spy_checkpoint(ckpt(seed), device)
    test = splits["test"].copy()
    test["hash"] = [pixel_hash(p) for p in test["path"]]
    frame, _, logits = run(model, test, classes)
    y = frame["class_idx"].to_numpy()
    pred = logits.argmax(1)
    blank = (frame["hash"] == blank_hash).to_numpy()
    def m(mask):
        labels = sorted(set(y[mask]))
        return {
            "n": int(mask.sum()),
            "macro_f1": float(f1_score(y[mask], pred[mask], labels=labels, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y[mask], pred[mask])),
            "accuracy": float(accuracy_score(y[mask], pred[mask])),
        }
    blank_preds = pd.Series([classes[i] for i in pred[blank]]).value_counts().to_dict()
    blank_true = pd.Series([classes[i] for i in y[blank]]).value_counts().to_dict()
    result["seeds"][seed] = {"all": m(np.ones_like(blank)), "without_blank": m(~blank),
                             "blank_predictions": blank_preds, "blank_true_labels": blank_true}
    print(seed, result["seeds"][seed]["all"]["macro_f1"], result["seeds"][seed]["without_blank"]["macro_f1"], blank_preds)

# (2) e (3) com o seed 42, tudo no mesmo espaco de embeddings
model, classes, _ = load_gravity_spy_checkpoint(ckpt(42), device)
_, emb_tr, log_tr = run(model, splits["train"], classes)
fr_va, emb_va, log_va = run(model, splits["val"], classes)
fr_te, emb_te, log_te = run(model, splits["test"], classes)
cls = {c: i for i, c in enumerate(classes)}
y_tr = splits["train"][splits["train"]["label"].isin(classes)]["label"].map(cls).to_numpy()

v2 = np.load(V9 / "outputs/research/ood_reference_seed42.npz", allow_pickle=False)
from gwproject.research.uncertainty import MahalanobisReference  # noqa: E402
ref_v2 = MahalanobisReference(class_means=v2["class_means"], precision=v2["precision"],
                              class_labels=v2["class_labels"])
q99_v2 = float(v2["mahalanobis_q99"])
ref_ok = fit_mahalanobis_reference(emb_tr, y_tr)
q99_ok = float(np.quantile(mahalanobis_score(emb_va, ref_ok), 0.99))
e99_ok = float(np.quantile(energy_score(log_va), 0.99))

test_v2 = mahalanobis_score(emb_te, ref_v2)
test_ok = mahalanobis_score(emb_te, ref_ok)
result["ood_test"] = {
    "v2_q99": q99_v2, "v2_test_above_q99": float(np.mean(test_v2 > q99_v2)),
    "fixed_q99": q99_ok, "fixed_test_above_q99": float(np.mean(test_ok > q99_ok)),
    "fixed_energy_q99": e99_ok,
    "fixed_test_energy_above_q99": float(np.mean(energy_score(log_te) > e99_ok)),
}
print(result["ood_test"])

# (3) 150 canais GWTC, renderizacao gravity_spy, embeddings do seed 42
cfg = load_toml(V9 / "configs/research_program.toml")["harmonized_inference"]
approved = pd.read_csv(V9 / "outputs/gwtc/approved_channels.csv")
approved = approved[approved["qgram_path"].notna()].reset_index(drop=True)
cat_tf = transforms.Compose([
    transforms.Resize((140, 170)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
])
ds = CatalogQgramDataset(
    approved, cat_tf,
    lower_percentile=float(cfg["lower_percentile"]), upper_percentile=float(cfg["upper_percentile"]),
    source_duration_seconds=float(cfg["source_outseg_seconds"]),
    target_duration_seconds=float(cfg["target_duration_seconds"]),
    baseline_mode=str(cfg["baseline_mode"]), flip_frequency=bool(cfg["flip_frequency"]),
    render_mode="gravity_spy",
)
feats, logits = [], []
fc = model.fc
model.fc = torch.nn.Identity()
with torch.no_grad():
    for images, _ in DataLoader(ds, batch_size=32, shuffle=False):
        f = model(images.to(device)).float()
        feats.append(f.cpu().numpy())
        logits.append(fc(f).float().cpu().numpy())
model.fc = fc
emb_c, log_c = np.concatenate(feats), np.concatenate(logits)
d_v2 = mahalanobis_score(emb_c, ref_v2)
d_ok = mahalanobis_score(emb_c, ref_ok)
e_c = energy_score(log_c)
gs = pd.read_csv(V9 / "outputs/research/gravity_spy_ensemble_gsrender/per_channel.csv")
result["gwtc"] = {
    "channels": int(len(emb_c)),
    "v2_reported_median_distance": float(gs["mahalanobis_distance"].median()),
    "seed42_v2ref_median_distance": float(np.median(d_v2)),
    "seed42_v2ref_above_q99": int(np.sum(d_v2 > q99_v2)),
    "seed42_fixed_median_distance": float(np.median(d_ok)),
    "seed42_fixed_above_q99": int(np.sum(d_ok > q99_ok)),
    "seed42_fixed_energy_above_q99": int(np.sum(e_c > e99_ok)),
    "seed42_fixed_either_above_q99": int(np.sum((d_ok > q99_ok) | (e_c > e99_ok))),
}
print(result["gwtc"])
OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
