# NN-project
Project for the course "Neural Networks. Theory and Practice" in the spring semester of the academic year 2025/26.

## Dataset: ENA24

The project uses the **ENA24-detection** dataset for object detection in camera trap images.

Official source:
- LILA BC dataset page: [https://lila.science/datasets/ena24detection/](https://lila.science/datasets/ena24detection/)

Official files:
- Metadata: [https://storage.googleapis.com/public-datasets-lila/ena24/ena24.json](https://storage.googleapis.com/public-datasets-lila/ena24/ena24.json)
- Public metadata without human images: [https://storage.googleapis.com/public-datasets-lila/ena24/ena24_public.json](https://storage.googleapis.com/public-datasets-lila/ena24/ena24_public.json)
- Images archive: [https://storage.googleapis.com/public-datasets-lila/ena24/ena24.zip](https://storage.googleapis.com/public-datasets-lila/ena24/ena24.zip)

The full dataset is larger than what is needed for early development. For this reason, the project uses a small sample dataset for notebook exploration and initial experiments.

### Sample preparation

The script `scripts/prepare_ena24_sample.py` creates a small development sample in:

```text
data/ena24_sample/
    images/
    annotations.json
```

It supports two workflows:
- **Public lightweight mode**: if `--data_dir` is omitted, the script downloads the public ENA24 metadata and only the selected sample images.
- **Local mode**: if you already downloaded ENA24 manually, the script reads the local files and copies a small random subset.

Run with a local dataset directory:

```bash
python scripts/prepare_ena24_sample.py --data_dir PATH_TO_DATASET
```

Run in lightweight public mode:

```bash
python scripts/prepare_ena24_sample.py
```

### Notes

- This creates a small sample for development and testing.
- The full dataset is **not required initially**.
- If you prefer, you can download ENA24 manually once and then use `--data_dir` to prepare the sample locally.
- If your local ENA24 directory structure differs from the expected layout, update the path resolution logic in `scripts/prepare_ena24_sample.py`.
