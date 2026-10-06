"""Train the frozen train/val baseline. Test images are never loaded here."""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import PIL
import sklearn
import torch
import torchvision
import yaml
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from src.models.densenet121_binary import build_densenet121_binary

MANIFEST = ROOT / "manifests/nih_effusion_vs_nofinding_v1.csv"
IMAGE_DIR = ROOT / "data/processed/nih/images_224x224"
EXPERIMENT_ID = "EXP-DN121-BASE-S42"
EXPERIMENT_DIR = ROOT / "experiments" / EXPERIMENT_ID


def load_configs():
    configs = {}
    for key, filename in (
        ("model", "densenet121_baseline_v1.yaml"),
        ("preprocessing", "preprocessing_v1.yaml"),
        ("augmentation", "augmentation_v1.yaml"),
    ):
        with (ROOT / "configs" / filename).open(encoding="utf-8") as stream:
            configs[key] = yaml.safe_load(stream)
    config = configs["model"]
    assert config["model"] == {"architecture": "densenet121", "pretrained": "ImageNet",
                               "num_classes": 1, "dropout_probability": 0.2}
    assert config["input"] == {"resolution": [224, 224], "channels": 3}
    training = config["training"]
    assert training["seed"] == config["reproducibility"]["deterministic_seed"] == 42
    assert config["reproducibility"]["save_rng_state"] is True
    assert training["optimizer"] == "AdamW" and training["loss"] == "BCEWithLogitsLoss"
    assert training["class_weights"] is None
    assert config["selection"]["best_checkpoint_metric"] == "val_auroc"
    for key in ("early_stopping", "scheduler"):
        assert training[key]["monitor"] == "val_auroc" and training[key]["mode"] == "max"
    assert training["scheduler"]["type"] == "ReduceLROnPlateau"
    return configs


def build_pipelines(configs):
    p, a = configs["preprocessing"], configs["augmentation"]
    assert p["version"] == a["version"] == "1.0"
    assert p["status"] == a["status"] == "frozen"
    expected = {
        "input_resolution": [224, 224], "input_mode": "grayscale", "source_image_mode": "L",
        "resize_required": False, "pixel_dtype": "float32",
        "channel_conversion": "replicate_grayscale_to_3_identical_channels",
        "pixel_scaling": {"source_range": [0, 255], "output_range": [0, 1]},
        "normalization": {"method": "ImageNet", "mean": [0.485, 0.456, 0.406],
                          "std": [0.229, 0.224, 0.225]},
        "deterministic": True, "shared_across_splits": True, "overwrite_source_images": False,
        "pipeline_order": ["training_augmentation_if_train", "replicate_channels", "to_tensor", "normalize"],
    }
    for key, value in expected.items():
        assert p[key] == value, f"Frozen preprocessing changed: {key}"
    for key in ("clahe", "histogram_equalization", "contrast_enhancement", "denoising",
                "sharpening", "intensity_clipping"):
        assert p[key] is False
    assert set(p) == set(expected) | {
        "version", "status", "channel_conversion_purpose", "rationale", "clahe",
        "histogram_equalization", "contrast_enhancement", "denoising", "sharpening", "intensity_clipping",
    }, "Unknown preprocessing setting"
    assert a["training_only"] is True and a["validation_transforms"] == a["test_transforms"] == []
    forbidden = ("horizontal_flip", "vertical_flip", "aggressive_crop", "random_contrast",
                 "random_brightness", "blur", "gaussian_noise", "resolution_degradation",
                 "clahe", "histogram_equalization")
    for key in forbidden:
        assert a[key] is False, f"Forbidden augmentation: {key}"
    assert set(a) == {"version", "status", "training_only", "validation_transforms",
                      "test_transforms", "transforms", "rationale", *forbidden}
    assert a["transforms"] == [{"name": "random_affine", "rotation_degrees": [-7.0, 7.0],
                                "translate_fraction": [0.05, 0.05], "interpolation": "bilinear",
                                "fill": 0, "scale": None, "shear": None}]
    affine = a["transforms"][0]
    primary = [T.Grayscale(num_output_channels=3), T.ToTensor(),
               T.Normalize(p["normalization"]["mean"], p["normalization"]["std"])]
    return {
        "train": T.Compose([T.RandomAffine(
            degrees=affine["rotation_degrees"], translate=affine["translate_fraction"],
            interpolation=InterpolationMode.BILINEAR, fill=affine["fill"],
            scale=affine["scale"], shear=affine["shear"]), *primary]),
        "val": T.Compose(primary), "test": T.Compose(primary),
    }


def load_manifest():
    frame = pd.read_csv(MANIFEST)
    assert len(frame) == 19326 and not frame["Image Index"].duplicated().any()
    assert frame[["Image Index", "Patient ID", "split", "binary_target"]].notna().all().all()
    assert set(frame["split"]) == {"train", "val", "test"}
    assert set(frame["cohort_version"].astype(str)) == {"1.0"}
    assert set(frame["endpoint"]) == {"Effusion_present_vs_No_Finding"}
    patients = {}
    for split, expected in {"train": 12384, "val": 3078, "test": 3864}.items():
        rows = frame[frame["split"].eq(split)]
        assert len(rows) == expected, f"Incorrect {split} size"
        assert rows["binary_target"].value_counts().to_dict() == {0: expected // 2, 1: expected // 2}
        patients[split] = set(rows["Patient ID"])
    assert not patients["train"] & patients["val"]
    assert not patients["train"] & patients["test"]
    assert not patients["val"] & patients["test"]
    assert frame.groupby("Patient ID").size().max() <= 5
    return frame


class CohortDataset(Dataset):
    """Only train/val are accepted; test evaluation belongs to a separate script."""

    def __init__(self, rows, split, transform, image_root=IMAGE_DIR):
        if split not in {"train", "val"}:
            raise ValueError("Test image loading is forbidden for training/model selection")
        assert len(rows) and rows["split"].eq(split).all()
        self.rows = rows.reset_index(drop=True).copy()
        self.transform = transform
        self.image_root = Path(image_root)
        self.loaded_filenames = set()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        name = row["Image Index"]
        assert Path(name).name == name and "/" not in name and "\\" not in name
        with Image.open(self.image_root / name) as image:
            image.load()
            assert image.mode == "L" and image.size == (224, 224)
            tensor = self.transform(image)
        assert tensor.shape == (3, 224, 224) and torch.isfinite(tensor).all()
        self.loaded_filenames.add(name)
        return tensor, torch.tensor([float(row["binary_target"])], dtype=torch.float32)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def evaluate_validation(model, loader, criterion, device):
    model.eval()
    logits, targets = [], []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            assert output.shape == labels.shape and torch.isfinite(output).all()
            total_loss += criterion(output, labels).item() * len(labels)
            logits.append(output.cpu().flatten())
            targets.append(labels.cpu().flatten())
    logits = torch.cat(logits)
    targets = torch.cat(targets).numpy().astype(int)
    probabilities = torch.sigmoid(logits).numpy()
    decisions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, decisions, labels=[0, 1]).ravel()
    metrics = {
        "val_auroc": float(roc_auc_score(targets, probabilities)),
        "val_auprc": float(average_precision_score(targets, probabilities)),
        "val_accuracy": float(accuracy_score(targets, decisions)),
        "val_sensitivity": float(tp / (tp + fn)),
        "val_specificity": float(tn / (tn + fp)),
        "val_f1": float(f1_score(targets, decisions, zero_division=0)),
        "val_bce_nll": total_loss / len(loader.dataset),
    }
    return metrics, logits.numpy(), probabilities


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "loader_generator": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=IMAGE_DIR,
                        help="Frozen cohort image directory (default: local processed images)")
    args = parser.parse_args()
    configs = load_configs()
    config = configs["model"]
    training = config["training"]
    manifest = load_manifest()
    pipelines = build_pipelines(configs)
    seed_everything(training["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    datasets = {split: CohortDataset(manifest[manifest["split"].eq(split)], split,
                                     pipelines[split], image_root=args.image_root)
                for split in ("train", "val")}
    generator = torch.Generator().manual_seed(training["seed"])
    loaders = {split: DataLoader(dataset, batch_size=training["batch_size"],
                                shuffle=(split == "train"), num_workers=0,
                                generator=generator if split == "train" else None)
               for split, dataset in datasets.items()}
    if EXPERIMENT_DIR.exists() and any(EXPERIMENT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing experiment: {EXPERIMENT_DIR}")
    model = build_densenet121_binary(config["model"]["dropout_probability"]).to(device)
    assert all(parameter.requires_grad for parameter in model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                 weight_decay=training["weight_decay"])
    scheduler_config = training["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=scheduler_config["mode"], factor=scheduler_config["factor"],
        patience=scheduler_config["patience"])
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "config_snapshot.yaml").write_text(yaml.safe_dump(configs, sort_keys=False))
    metadata = {
        "experiment_id": EXPERIMENT_ID, "architecture": "densenet121",
        "pretrained_weights": "DenseNet121_Weights.IMAGENET1K_V1", "seed": training["seed"],
        "preprocessing_version": configs["preprocessing"]["version"],
        "augmentation_version": configs["augmentation"]["version"], "cohort_version": "1.0",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "optimizer": "AdamW", "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"], "loss": "BCEWithLogitsLoss",
        "dropout_probability": config["model"]["dropout_probability"],
        "best_epoch": None, "best_val_auroc": None, "device": str(device),
        "auprc_definition": "sklearn average_precision_score (non-interpolated)",
        "descriptive_threshold": 0.5, "test_images_loaded": 0,
        "software_versions": {"python": sys.version, "torch": torch.__version__,
                              "torchvision": torchvision.__version__, "numpy": np.__version__,
                              "pandas": pd.__version__, "sklearn": sklearn.__version__,
                              "Pillow": PIL.__version__, "PyYAML": yaml.__version__},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    best, bad_epochs, history = float("-inf"), 0, []
    early = training["early_stopping"]
    for epoch in range(1, training["epochs"] + 1):
        model.train()
        train_loss = 0.0
        learning_rate = optimizer.param_groups[0]["lr"]
        for images, targets in loaders["train"]:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            assert logits.shape == targets.shape and torch.isfinite(logits).all()
            loss = criterion(logits, targets)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)
        metrics, val_logits, probabilities = evaluate_validation(model, loaders["val"], criterion, device)
        improved = metrics["val_auroc"] > best
        if improved:
            best, bad_epochs = metrics["val_auroc"], 0
            metadata.update(best_epoch=epoch, best_val_auroc=best)
        else:
            bad_epochs += 1
        scheduler.step(metrics["val_auroc"])
        history.append({"epoch": epoch, "learning_rate": learning_rate,
                        "train_bce_nll": train_loss / len(datasets["train"]), **metrics})
        pd.DataFrame(history).to_csv(EXPERIMENT_DIR / "training_history.csv", index=False)
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "scheduler_state_dict": scheduler.state_dict(),
                      "rng_state": rng_state(generator), "configs": configs,
                      "best_val_auroc": best, "best_epoch": metadata["best_epoch"],
                      "bad_epochs": bad_epochs, "history": history}
        torch.save(checkpoint, EXPERIMENT_DIR / "checkpoint_last.pt")
        if improved:
            torch.save(checkpoint, EXPERIMENT_DIR / "checkpoint_best.pt")
            predictions = datasets["val"].rows[["Image Index", "Patient ID", "split", "binary_target"]].copy()
            predictions["raw_logit"] = val_logits
            predictions["probability"] = probabilities
            predictions.to_csv(EXPERIMENT_DIR / "val_predictions.csv", index=False)
            np.save(EXPERIMENT_DIR / "val_logits.npy", val_logits)
        metadata.update(completed_epochs=epoch, updated_at_utc=datetime.now(timezone.utc).isoformat())
        (EXPERIMENT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(history[-1]), flush=True)
        if early["enabled"] and bad_epochs >= early["patience"]:
            print(f"Early stopping at epoch {epoch}; best epoch {metadata['best_epoch']}")
            break
    print(f"Finished {EXPERIMENT_ID}; no test images loaded or test predictions generated.")


if __name__ == "__main__":
    main()
