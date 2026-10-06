# First Kaggle DenseNet-121 run

Experiment: `EXP-DN121-BASE-S42`. Use
`notebooks/kaggle_train_densenet121_v1.ipynb` as the orchestration notebook.
The repository script implements training; the notebook does not implement a
second training loop. Full training is run on Kaggle only.

## Prepare the frozen input dataset

From `~/projects/MedTrust-AI`, package only the existing 19,326 files referenced
by `manifests/nih_effusion_vs_nofinding_v1.csv`. Do not download more images,
regenerate the cohort, resize files, or modify their bytes. For example, run
this packaging command locally when preparing the upload (it is not part of
the notebook):

```bash
source .venv/bin/activate
python - <<'PY'
import csv
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path.cwd()
with (root / 'manifests/nih_effusion_vs_nofinding_v1.csv').open(newline='') as stream:
    names = [row['Image Index'] for row in csv.DictReader(stream)]
assert len(names) == len(set(names)) == 19326
images = root / 'data/processed/nih/images_224x224'
assert all(Path(name).name == name for name in names)
assert all((images / name).is_file() and (images / name).stat().st_size > 0 for name in names)
archive = Path('/tmp/medtrust-nih-224.zip')
assert not archive.exists(), 'Choose a fresh archive path instead of overwriting an existing upload.'
with ZipFile(archive, 'x', compression=ZIP_DEFLATED) as output:
    for name in sorted(names):
        output.write(images / name, arcname=f'images_224x224/{name}')
print(archive)
PY
```

In Kaggle's Datasets area, create a new dataset, upload this archive, and use
`medtrust-nih-224` as the dataset slug where available. A private dataset is
sufficient. Verify that the published dataset exposes the extracted
`images_224x224/` folder with all 19,326 PNGs. The notebook reads extracted
files, not an unopened ZIP. Keep the dataset version fixed for the experiment.
The frozen manifest comes from the pinned Git repository and need not be
uploaded as a separate dataset.

## Attach data and select GPU

Create a Kaggle notebook and import the `.ipynb` file. Use **Add Input** to
attach the frozen image dataset. Inspect the mounted directory in the input
browser and edit the variable near the top if its path differs:

```python
KAGGLE_IMAGE_ROOT = "/kaggle/input/medtrust-nih-224/images_224x224"
KAGGLE_WORK_ROOT = "/kaggle/working/MedTrust-AI"
```

In notebook settings, select a GPU accelerator and enable Internet for cloning
GitHub and loading official ImageNet-pretrained weights. The first cell prints
Python, torch, torchvision, CUDA availability, GPU name, and working directory.
The notebook refuses to start training without CUDA. The script uses one GPU;
it does not use distributed training. If a required dependency is missing,
stop and report it; the notebook does not install packages.

GPU settings are described in [NVIDIA's Kaggle documentation](https://docs.nvidia.com/datascience/deployment/latest/platforms/kaggle/).

## Exact baseline provenance

The first DenseNet baseline training run must use exactly:

```text
0daf10aa0d6a19bd53ecc2eb8d24e6b1b3d19db2
```

The notebook clones `https://github.com/Samaira-0/MedTrust-AI.git` and checks out
this exact committed revision in detached-HEAD mode. The revision includes the
validated `--image-root` implementation. The notebook pins:

```python
EXPECTED_GIT_COMMIT = "0daf10aa0d6a19bd53ecc2eb8d24e6b1b3d19db2"
```

Keep this full Git SHA fixed for the first experiment; do not substitute a
branch name or another revision. The notebook verifies that `git rev-parse HEAD`
equals `EXPECTED_GIT_COMMIT` after checkout and again before training. It stops
on any mismatch or dirty checkout. Model, cohort, frozen configuration, and
selection behavior must remain reproducible.

**No runtime patching of tracked source code is allowed.** The notebook invokes
the repository trainer exactly as committed. Do not apply patches, rewrite
scripts, or change tracked configs in Kaggle. The trainer records the Git commit
and clean status in `metadata.json`; the notebook checks those values after
training. The default image path remains compatible with local use, and Kaggle
supplies the attached dataset path through the committed `--image-root` CLI.

## Run and inspect

Run the cells from top to bottom, or use Run All / Save Version with execution.
The preflight verifies the manifest, counts (19,326 overall; 12,384 train;
3,078 val; 3,864 test), balance, unique filenames, patient separation, patient
cap, frozen configs, and presence/nonzero size of every referenced image.
Checking test file paths does not decode or evaluate test images.

The training cell executes:

```bash
python scripts/train_densenet121_baseline.py --image-root "$KAGGLE_IMAGE_ROOT"
```

The script loads `configs/densenet121_baseline_v1.yaml` itself; there is no
`--config` argument. It trains the full ImageNet DenseNet-121 with dropout 0.2,
BCEWithLogitsLoss, AdamW (learning rate and weight decay both `1e-4`), batch
size 32, seed 42, and at most 20 epochs. Training augmentation is mild affine
only. Validation is deterministic. ReduceLROnPlateau and early stopping use
validation AUROC, with early-stopping patience 5. The best checkpoint is
selected solely by validation AUROC. Threshold 0.5 is descriptive only.

The trainer constructs only train and val datasets. **The frozen test split
remains untouched by training and model selection**: no test image decoding,
predictions, metrics, threshold tuning, early stopping, or scheduler decisions.
Test evaluation will occur in a separate frozen step.

Do not change batch size, epochs, model, preprocessing, augmentation, or
selection settings to overcome resource limits under the same experiment ID.
If the runtime ends or training fails, preserve available outputs and report
the failure. The current trainer saves last-checkpoint/RNG state but does not
implement a resume CLI. Start a fresh session for a new attempt rather than
silently overwriting an existing run. Existing nonempty experiment directories
and dirty reused checkouts are rejected.

## Outputs and download

Outputs appear under:

```text
/kaggle/working/MedTrust-AI/experiments/EXP-DN121-BASE-S42/
    checkpoint_best.pt
    checkpoint_last.pt
    training_history.csv
    config_snapshot.yaml
    metadata.json
    val_predictions.csv
    val_logits.npy
```

Validation prediction files correspond to the best checkpoint, not the last
epoch. After successful training, the notebook prints best epoch, best
validation AUROC, validation AUPRC (average precision) and BCE/NLL loss from
that same epoch, and the directory contents. It verifies committed-source
provenance and creates:

```text
/kaggle/working/EXP-DN121-BASE-S42.zip
```

The ZIP contains the entire experiment output directory.
Use the displayed file link or download the ZIP from the saved notebook's
Output files. Save the executed notebook version so the output persists, and
retain the archive and execution logs with the exact dataset version and Git
commit. Do not publish test predictions in this step.

For output retrieval through an already-configured Kaggle CLI, see the
[official notebook CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).
