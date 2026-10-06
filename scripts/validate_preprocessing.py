"""Validate frozen torchvision pipelines in memory, without writing images."""

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def main():
    if importlib.util.find_spec("torchvision") is None:
        print("FAIL: torchvision is not installed in the active environment.")
        print("Validation stopped; no packages installed and no images modified.")
        return 1
    try:
        import torch
        import torchvision
        from torchvision import transforms as T
        from torchvision.transforms import InterpolationMode
    except Exception as error:
        print(f"FAIL: torchvision cannot be imported: {type(error).__name__}: {error}")
        return 1

    import pandas as pd
    import yaml
    from PIL import Image

    def read_config(name):
        with (ROOT / "configs" / name).open(encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    preprocessing = read_config("preprocessing_v1.yaml")
    augmentation = read_config("augmentation_v1.yaml")
    for config in (preprocessing, augmentation):
        assert config["version"] == "1.0" and config["status"] == "frozen"
    expected_preprocessing = {
        "input_resolution": [224, 224], "input_mode": "grayscale",
        "source_image_mode": "L", "resize_required": False,
        "channel_conversion": "replicate_grayscale_to_3_identical_channels",
        "pixel_dtype": "float32",
        "pixel_scaling": {"source_range": [0, 255], "output_range": [0, 1]},
        "normalization": {"method": "ImageNet", "mean": [0.485, 0.456, 0.406],
                          "std": [0.229, 0.224, 0.225]},
        "deterministic": True, "shared_across_splits": True,
        "overwrite_source_images": False,
        "pipeline_order": ["training_augmentation_if_train", "replicate_channels",
                           "to_tensor", "normalize"],
    }
    for key, value in expected_preprocessing.items():
        assert preprocessing[key] == value, f"Invalid preprocessing: {key}"
    for key in ("clahe", "histogram_equalization", "contrast_enhancement",
                "denoising", "sharpening", "intensity_clipping"):
        assert preprocessing[key] is False, f"Forbidden preprocessing: {key}"
    assert augmentation["training_only"] is True
    assert augmentation["validation_transforms"] == []
    assert augmentation["test_transforms"] == []
    for key in ("horizontal_flip", "vertical_flip", "aggressive_crop", "random_contrast", "random_brightness",
                "blur", "gaussian_noise", "resolution_degradation", "clahe",
                "histogram_equalization"):
        assert augmentation[key] is False, f"Forbidden augmentation: {key}"
    definitions = augmentation["transforms"]
    assert definitions == [
        {"name": "random_affine", "rotation_degrees": [-7.0, 7.0],
         "translate_fraction": [0.05, 0.05], "interpolation": "bilinear",
         "fill": 0, "scale": None, "shear": None},
    ], "Training transformation allowlist or parameters changed"
    # Reject added config settings rather than silently ignoring transformations.
    assert set(preprocessing) == set(expected_preprocessing) | {
        "version", "status", "channel_conversion_purpose", "rationale", "clahe",
        "histogram_equalization", "contrast_enhancement", "denoising",
        "sharpening", "intensity_clipping",
    }, "Unknown preprocessing settings"
    assert set(augmentation) == {
        "version", "status", "training_only", "validation_transforms", "test_transforms",
        "transforms", "horizontal_flip", "vertical_flip", "aggressive_crop", "random_contrast",
        "random_brightness", "blur", "gaussian_noise", "resolution_degradation",
        "clahe", "histogram_equalization", "rationale",
    }, "Unknown augmentation settings"

    to_tensor = T.Compose([T.Grayscale(num_output_channels=3), T.ToTensor()])
    normalization = preprocessing["normalization"]
    normalize = T.Normalize(normalization["mean"], normalization["std"])
    primary = T.Compose([*to_tensor.transforms, normalize])
    affine = definitions[0]
    training_augmentation = T.Compose([
        T.RandomAffine(degrees=affine["rotation_degrees"],
                       translate=affine["translate_fraction"],
                       interpolation=InterpolationMode.BILINEAR, fill=affine["fill"],
                       scale=affine["scale"], shear=affine["shear"]),
    ])
    assert [type(transform) for transform in training_augmentation.transforms] == [
        T.RandomAffine,
    ]
    # The same deterministic primary pipeline serves all splits; only train
    # prepends only the approved mild affine transformation.
    pipelines = {"train": T.Compose([training_augmentation, primary]),
                 "val": primary, "test": primary}
    assert all(type(transform) in (T.Grayscale, T.ToTensor, T.Normalize)
               for split in ("val", "test") for transform in pipelines[split].transforms)
    manifest = pd.read_csv(ROOT / "manifests/nih_effusion_vs_nofinding_v1.csv")
    assert len(manifest) == 19326 and not manifest["Image Index"].duplicated().any()
    torch.manual_seed(SEED)
    samples = []
    for split in ("train", "val", "test"):
        for target in (0, 1):
            group = manifest[manifest["split"].eq(split) & manifest["binary_target"].eq(target)]
            assert len(group) >= 2, f"Missing sample stratum: {split}/{target}"
            samples.extend(group.sample(n=2, random_state=SEED).to_dict("records"))
    for row in samples:
        name = row["Image Index"]
        assert Path(name).name == name, "Invalid filename"
        path = ROOT / "data/processed/nih/images_224x224" / name
        original = path.read_bytes()
        before = hashlib.sha256(original).digest()
        with Image.open(path) as image:
            image.load()
            assert image.mode == preprocessing["source_image_mode"]
            assert image.size == tuple(preprocessing["input_resolution"])
            tensor = to_tensor(image)
            assert tensor.shape == (3, 224, 224) and tensor.dtype == torch.float32
            assert torch.equal(tensor[0], tensor[1]) and torch.equal(tensor[1], tensor[2])
            assert 0 <= tensor.min() and tensor.max() <= 1
            first = primary(image)
            assert first.shape == (3, 224, 224)
            mean = torch.tensor(normalization["mean"], dtype=tensor.dtype).view(3, 1, 1)
            std = torch.tensor(normalization["std"], dtype=tensor.dtype).view(3, 1, 1)
            assert torch.equal(first, (tensor - mean) / std), "ImageNet normalization not applied"
            assert torch.isfinite(first).all()
            assert torch.equal(first, primary(image)), "Nondeterministic primary pipeline"
            for split in ("val", "test"):
                assert torch.equal(first, pipelines[split](image))
            for _ in range(4):
                augmented = pipelines["train"](image)
                assert augmented.shape == (3, 224, 224)
                assert augmented.dtype == torch.float32 and torch.isfinite(augmented).all()
        assert hashlib.sha256(path.read_bytes()).digest() == before, "Source modified"
    print(f"torchvision available: {torchvision.__version__}")
    print("PASS: frozen config values and transformation allowlist.")
    print("PASS: 12 samples, two per train/val/test and binary_target stratum.")
    print("PASS: L inputs at 224x224; float32 [3,224,224]; identical channels before normalization.")
    print("PASS: finite deterministic ImageNet normalization; val/test contain no augmentation.")
    print("PASS: 48 training augmentation checks; only configured mild affine; no flipping.")
    print("PASS: sampled source SHA256 hashes unchanged; no transformed images saved.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}")
        raise SystemExit(1)
