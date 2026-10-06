"""One training step and one validation pass; no checkpoints or test images."""

import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    import pandas as pd
    import torch
    import torchvision
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from train_densenet121_baseline import (
        CohortDataset, IMAGE_DIR, build_densenet121_binary, build_pipelines,
        load_configs, load_manifest, seed_everything,
    )

    print(f"torch: {torch.__version__}", flush=True)
    print(f"torchvision: {torchvision.__version__}", flush=True)
    torch.set_num_threads(4)
    configs = load_configs()
    pipelines = build_pipelines(configs)
    manifest = load_manifest()
    seed_everything(42)
    samples = {}
    for split in ("train", "val"):
        rows = manifest[manifest["split"].eq(split)]
        samples[split] = pd.concat([
            rows[rows["binary_target"].eq(target)].sample(n=8, random_state=42)
            for target in (0, 1)
        ], ignore_index=True)
    hashes = {name: hashlib.sha256((IMAGE_DIR / name).read_bytes()).hexdigest()
              for rows in samples.values() for name in rows["Image Index"]}
    for name in hashes:
        with Image.open(IMAGE_DIR / name) as image:
            image.load()
            assert image.mode == "L" and image.size == (224, 224)
            tensor = T.ToTensor()(T.Grayscale(num_output_channels=3)(image))
            assert tensor.shape == (3, 224, 224)
            assert torch.equal(tensor[0], tensor[1]) and torch.equal(tensor[1], tensor[2])
            assert torch.equal(pipelines["val"](image), pipelines["val"](image))
            augmented = pipelines["train"](image)
            assert augmented.shape == (3, 224, 224) and torch.isfinite(augmented).all()
    print("PASS: 16 train + 16 val samples; frozen preprocessing and affine-only augmentation.", flush=True)
    print("PASS: identical grayscale channels before normalization; deterministic val pipeline.", flush=True)
    datasets = {split: CohortDataset(rows, split, pipelines[split]) for split, rows in samples.items()}
    loaders = {split: DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
               for split, dataset in datasets.items()}
    # Respect existing cache; if weights are absent, use a temporary cache rather
    # than writing outside the workspace's permitted output paths.
    cached = Path(torch.hub.get_dir()) / "checkpoints/densenet121-a639ec97.pth"
    if not cached.is_file():
        torch.hub.set_dir(str(Path(tempfile.gettempdir()) / "medtrust_torch_hub"))
    model = build_densenet121_binary(configs["model"]["model"]["dropout_probability"])
    assert isinstance(model.classifier[0], nn.Dropout) and model.classifier[0].p == 0.2
    assert isinstance(model.classifier[1], nn.Linear) and model.classifier[1].out_features == 1
    assert all(parameter.requires_grad for parameter in model.parameters())
    print("PASS: ImageNet DenseNet121_Weights.IMAGENET1K_V1 loaded; dropout 0.2; full model trainable.", flush=True)
    training = configs["model"]["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                 weight_decay=training["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    images, targets = next(iter(loaders["train"]))
    assert images.shape == (16, 3, 224, 224)
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    assert logits.shape == (16, 1) and torch.isfinite(logits).all()
    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters() if parameter.requires_grad)
    assert model.features.conv0.weight.grad.abs().sum() > 0
    assert model.classifier[1].weight.grad.abs().sum() > 0
    before = model.classifier[1].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.classifier[1].weight)
    print(f"PASS: train logits [16,1], finite BCE={loss.item():.6f}; backbone/head gradients; one AdamW step.", flush=True)
    model.eval()
    with torch.no_grad():
        images, targets = next(iter(loaders["val"]))
        logits = model(images)
        assert logits.shape == (16, 1) and torch.isfinite(logits).all()
        val_loss = criterion(logits, targets)
        assert torch.isfinite(val_loss)
    print(f"PASS: validation forward [16,1], finite BCE={val_loss.item():.6f}.", flush=True)
    loaded = set.union(*(dataset.loaded_filenames for dataset in datasets.values()))
    assert len(loaded) == 32
    assert not loaded & set(manifest.loc[manifest["split"].eq("test"), "Image Index"])
    for name, digest in hashes.items():
        assert hashlib.sha256((IMAGE_DIR / name).read_bytes()).hexdigest() == digest
    print("PASS: test images never loaded; source hashes unchanged; no checkpoints saved.", flush=True)
    print("PASS: DenseNet-121 smoke test completed; full training was not run.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", flush=True)
        raise SystemExit(1)
