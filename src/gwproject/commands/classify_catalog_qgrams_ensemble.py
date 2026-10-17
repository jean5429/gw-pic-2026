from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gwproject.config import load_toml
from gwproject.gravity_spy import resolve_device
from gwproject.provenance import sha256_file, write_manifest
from gwproject.qgram_inference import (
    RENDER_MODES,
    CatalogQgramDataset,
    load_gravity_spy_checkpoint,
)
from gwproject.research.uncertainty import (
    MahalanobisReference,
    empirical_cdf,
    energy_score,
    ensemble_statistics,
    mahalanobis_score,
)


def _checkpoint_paths(values: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        if value.is_dir():
            paths.extend(sorted(value.rglob("best_model.pt")))
        else:
            paths.append(value)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            if not path.is_file():
                raise FileNotFoundError(path)
            unique.append(path)
            seen.add(resolved)
    if not unique:
        raise ValueError("nenhum checkpoint encontrado")
    return unique


def _load_ood_bundle(path: Path, checkpoints: list[Path]) -> list[dict]:
    data = np.load(path, allow_pickle=False)
    if str(data["format_version"].item()) != "4":
        raise ValueError("formato OOD incompatível; reconstrua a referencia")
    if (
        str(data["reference_fit_split"].item()) != "train"
        or str(data["calibration_split"].item()) != "val"
    ):
        raise ValueError(
            "referencia OOD sem separacao treino-ajuste/validacao-calibracao"
        )

    stored_hashes = data["checkpoint_sha256"].astype(str).tolist()
    index_by_hash = {digest: index for index, digest in enumerate(stored_hashes)}
    if len(index_by_hash) != len(stored_hashes):
        raise ValueError("referencia OOD contem hashes duplicados")

    references: list[dict] = []
    missing: list[str] = []
    for checkpoint in checkpoints:
        digest = sha256_file(checkpoint)
        if digest not in index_by_hash:
            missing.append(str(checkpoint))
            continue
        index = index_by_hash[digest]
        references.append({
            "checkpoint_sha256": digest,
            "reference": MahalanobisReference(
                class_means=data["class_means"][index],
                precision=data["precision"][index],
                class_labels=data["class_labels"],
            ),
            "validation_energy": data["validation_energy"][index],
            "validation_mahalanobis": data["validation_mahalanobis"][index],
            "temperature": float(data["temperatures"][index]),
        })
    if missing:
        raise ValueError(
            "checkpoints sem referencia OOD correspondente: " + ", ".join(missing)
        )
    return references


CANONICAL_OUTPUT_DIR = Path("outputs/research/gwtc4_218/gravity_spy_ensemble")


def mahalanobis_log_excess(distances: np.ndarray, validation_distances: np.ndarray, quantile: float) -> np.ndarray:
    """``log(1 + d) - log(1 + q)``, com ``q`` o quantil da validacao do mesmo checkpoint.

    Percentis empiricos saturam em 1 fora do dominio; este escore continuo, ja usado no
    experimento pareado de intensidade, mostra se uma entrada se aproxima do limiar.
    """
    threshold = float(np.quantile(np.asarray(validation_distances, dtype=float), quantile))
    return np.log1p(np.asarray(distances, dtype=float)) - np.log1p(threshold)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inferencia ensemble e open-world sobre Q-grams do catalogo."
    )
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--approved", type=Path, default=Path("outputs/gwtc4_218/approved_channels.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/research_program.toml"))
    parser.add_argument("--output-dir", type=Path, default=CANONICAL_OUTPUT_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--ood-reference", type=Path, required=True)
    parser.add_argument(
        "--render-mode",
        choices=RENDER_MODES,
        default=None,
        help="sobrepoe [harmonized_inference].render_mode; exige --output-dir nao canonico",
    )
    args = parser.parse_args()

    checkpoints = _checkpoint_paths(args.checkpoints)
    cfg_all = load_toml(args.config)
    cfg = cfg_all["harmonized_inference"]
    configured_render_mode = str(cfg.get("render_mode", "percentile"))
    render_mode = args.render_mode or configured_render_mode
    if render_mode != configured_render_mode and args.output_dir.resolve() == CANONICAL_OUTPUT_DIR.resolve():
        raise ValueError(
            "--render-mode diferente da configuracao nao pode gravar na saida canonica; "
            "informe um --output-dir proprio"
        )
    cfg["render_mode"] = render_mode

    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    ood_cfg = cfg_all.get("open_world", {})
    minimum_ood_model_fraction = float(ood_cfg.get("minimum_ood_model_fraction", 0.6))
    maximum_mutual_information = float(
        ood_cfg.get("maximum_mutual_information", 0.10)
    )
    energy_quantiles = [float(value) for value in ood_cfg["energy_quantiles"]]
    mahalanobis_quantiles = [
        float(value) for value in ood_cfg["mahalanobis_quantiles"]
    ]
    if (
        len(energy_quantiles) != 2
        or len(mahalanobis_quantiles) != 2
        or not 0 < energy_quantiles[0] < energy_quantiles[1] < 1
        or not 0 < mahalanobis_quantiles[0] < mahalanobis_quantiles[1] < 1
    ):
        raise ValueError("quantis OOD devem conter dois valores crescentes em (0, 1)")
    if not 0 < minimum_ood_model_fraction <= 1:
        raise ValueError("minimum_ood_model_fraction deve pertencer a (0, 1]")

    device = resolve_device(args.device)
    loaded = [load_gravity_spy_checkpoint(path, device) for path in checkpoints]
    classes = loaded[0][1]
    for _, model_classes, _ in loaded[1:]:
        if model_classes != classes:
            raise ValueError("checkpoints com taxonomias diferentes")
    models = [item[0] for item in loaded]

    approved = pd.read_csv(args.approved)
    approved = approved[approved["qgram_path"].notna()].reset_index(drop=True)
    transform = transforms.Compose([
        transforms.Resize((int(cfg["resize_height"]), int(cfg["resize_width"]))),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    loader = DataLoader(
        CatalogQgramDataset(
            approved,
            transform,
            lower_percentile=float(cfg["lower_percentile"]),
            upper_percentile=float(cfg["upper_percentile"]),
            source_duration_seconds=float(cfg["source_outseg_seconds"]),
            target_duration_seconds=float(cfg["target_duration_seconds"]),
            baseline_mode=str(cfg["baseline_mode"]),
            flip_frequency=bool(cfg["flip_frequency"]),
            render_mode=render_mode,
        ),
        batch_size=int(args.batch_size or cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["num_workers"]) > 0,
    )
    for model in models:
        model.eval()
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last)

    ood_references = _load_ood_bundle(args.ood_reference, checkpoints)
    for model, reference_data in zip(models, ood_references):
        if not np.isclose(
            float(model.gw_temperature),
            float(reference_data["temperature"]),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("temperatura do checkpoint diverge da referencia OOD")

    rows: list[dict] = []
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)

            logits_models: list[np.ndarray] = []
            embeddings_models: list[np.ndarray] = []
            for model in models:
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    original_fc = model.fc
                    model.fc = torch.nn.Identity()
                    features = model(images)
                    logits = original_fc(features) / float(model.gw_temperature)
                    model.fc = original_fc
                    embeddings_models.append(features.float().cpu().numpy())
                logits_models.append(logits.float().cpu().numpy())

            stack = np.stack(logits_models, axis=0)
            stats = ensemble_statistics(stack)
            probabilities = stats["mean_probability"]

            energy_percentiles = []
            mahalanobis_percentiles = []
            mahalanobis_excesses = []
            for model_index, reference_data in enumerate(ood_references):
                model_energy = energy_score(stack[model_index])
                model_mahalanobis = mahalanobis_score(
                    embeddings_models[model_index], reference_data["reference"]
                )
                energy_percentiles.append(
                    empirical_cdf(model_energy, reference_data["validation_energy"])
                )
                mahalanobis_percentiles.append(
                    empirical_cdf(
                        model_mahalanobis,
                        reference_data["validation_mahalanobis"],
                    )
                )
                mahalanobis_excesses.append(
                    mahalanobis_log_excess(
                        model_mahalanobis,
                        reference_data["validation_mahalanobis"],
                        mahalanobis_quantiles[1],
                    )
                )
            energy_percentiles_array = np.stack(energy_percentiles, axis=0)
            mahalanobis_percentiles_array = np.stack(mahalanobis_percentiles, axis=0)
            mahalanobis_excess_array = np.stack(mahalanobis_excesses, axis=0)

            for batch_index, source_index in enumerate(indices.numpy().tolist()):
                probability = probabilities[batch_index]
                order = np.argsort(probability)[::-1]
                source = approved.iloc[int(source_index)]

                energy_model_percentiles = energy_percentiles_array[:, batch_index]
                mahal_model_percentiles = mahalanobis_percentiles_array[:, batch_index]
                energy_fraction_95 = float(
                    np.mean(energy_model_percentiles >= energy_quantiles[0])
                )
                energy_fraction_99 = float(
                    np.mean(energy_model_percentiles >= energy_quantiles[1])
                )
                mahal_fraction_95 = float(
                    np.mean(mahal_model_percentiles >= mahalanobis_quantiles[0])
                )
                mahal_fraction_99 = float(
                    np.mean(mahal_model_percentiles >= mahalanobis_quantiles[1])
                )

                row = {
                    "event": source["event"],
                    "detector": source["detector"],
                    "qgram_path": source["qgram_path"],
                    "forced_closed_set_class": classes[int(order[0])],
                    "closed_set_ensemble_confidence": float(probability[order[0]]),
                    "forced_second_class": classes[int(order[1])],
                    "forced_second_probability": float(probability[order[1]]),
                    "predictive_entropy": float(stats["predictive_entropy"][batch_index]),
                    "expected_entropy": float(stats["expected_entropy"][batch_index]),
                    "mutual_information": float(stats["mutual_information"][batch_index]),
                    "vote_fraction": float(stats["vote_fraction"][batch_index]),
                    "probability_margin": float(stats["probability_margin"][batch_index]),
                    "energy_percentile_median": float(np.nanmedian(energy_model_percentiles)),
                    "mahalanobis_percentile_median": float(np.nanmedian(mahal_model_percentiles)),
                    "mahalanobis_log_excess_99": float(
                        np.median(mahalanobis_excess_array[:, batch_index])
                    ),
                    "ood_energy_model_fraction_95": energy_fraction_95,
                    "ood_energy_model_fraction_99": energy_fraction_99,
                    "ood_mahalanobis_model_fraction_95": mahal_fraction_95,
                    "ood_mahalanobis_model_fraction_99": mahal_fraction_99,
                    "ood_energy_95": bool(
                        energy_fraction_95 >= minimum_ood_model_fraction
                    ),
                    "ood_energy_99": bool(
                        energy_fraction_99 >= minimum_ood_model_fraction
                    ),
                    "ood_mahalanobis_95": bool(
                        mahal_fraction_95 >= minimum_ood_model_fraction
                    ),
                    "ood_mahalanobis_99": bool(
                        mahal_fraction_99 >= minimum_ood_model_fraction
                    ),
                }
                row["open_world_status"] = (
                    "reject_ood"
                    if row["ood_energy_99"] or row["ood_mahalanobis_99"]
                    else "review"
                    if row["ood_energy_95"]
                    or row["ood_mahalanobis_95"]
                    or row["vote_fraction"] < float(cfg["minimum_vote_fraction"])
                    or row["mutual_information"] > maximum_mutual_information
                    else "in_domain_candidate"
                )
                row["reported_class"] = (
                    row["forced_closed_set_class"]
                    if row["open_world_status"] == "in_domain_candidate"
                    else None
                )
                rows.append(row)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["event", "detector"])
    frame.to_csv(output / "per_channel.csv", index=False)

    event_rows = []
    for event, group in frame.groupby("event"):
        event_rows.append({
            "event": event,
            "n_channels": int(len(group)),
            "detectors": ",".join(group["detector"].astype(str)),
            "forced_closed_set_classes": ",".join(
                group["forced_closed_set_class"].astype(str)
            ),
            "reported_classes": ",".join(
                group["reported_class"].dropna().astype(str)
            ),
            "forced_class_detector_disagreement": bool(
                group["forced_closed_set_class"].nunique() > 1
            ),
            "maximum_ood_severity": (
                "reject_ood"
                if (group["open_world_status"] == "reject_ood").any()
                else "review"
                if (group["open_world_status"] == "review").any()
                else "in_domain_candidate"
            ),
            "mean_vote_fraction": float(group["vote_fraction"].mean()),
            "mean_mutual_information": float(group["mutual_information"].mean()),
        })
    event_frame = pd.DataFrame(event_rows)
    event_frame.to_csv(output / "per_event.csv", index=False)

    summary = {
        "checkpoints": [
            {"path": str(path), "sha256": sha256_file(path)} for path in checkpoints
        ],
        "n_models": len(models),
        "channels": int(len(frame)),
        "events": int(frame["event"].nunique()),
        "forced_closed_set_class_counts": (
            frame["forced_closed_set_class"].value_counts().to_dict()
        ),
        "reported_class_counts": frame["reported_class"].value_counts().to_dict(),
        "open_world_counts": frame["open_world_status"].value_counts().to_dict(),
        "forced_class_detector_disagreement_events": int(
            event_frame["forced_class_detector_disagreement"].sum()
        ),
        "mahalanobis_log_excess_99_median": float(
            frame["mahalanobis_log_excess_99"].median()
        ),
        "render_mode": render_mode,
        "preprocessing": dict(cfg),
        "ood_calibration": {
            "reference": str(args.ood_reference),
            "reference_sha256": sha256_file(args.ood_reference),
            "minimum_ood_model_fraction": minimum_ood_model_fraction,
            "energy_quantiles": energy_quantiles,
            "mahalanobis_quantiles": mahalanobis_quantiles,
            "maximum_mutual_information": maximum_mutual_information,
            "temperature_calibration_applied": True,
            "method": "percentis empiricos independentes por checkpoint",
        },
        "interpretation_warning": (
            "reported_class fica vazio para review ou reject_ood. O rotulo fechado "
            "forcado e preservado apenas como diagnostico. Mesmo in_domain_candidate "
            "nao e prova de classe fisica: OOD e ensemble medem compatibilidade "
            "estatistica, nao a origem causal do transiente."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_manifest(output / "manifest.json", {"config": cfg_all, "summary": summary})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
