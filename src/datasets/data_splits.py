import torch

from src.datasets.ena24_dataset import ENA24DetectionDataset


def prepare_data_splits(
    data_dir,
    train_ratio=0.6,
    val_ratio=0.2,
    seed=42,
    size=None,
):
    torch.manual_seed(seed)

    full_dataset = ENA24DetectionDataset(data_dir=data_dir)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_dataset), generator=generator).tolist()

    if size is not None:
        size = min(size, len(indices))
        indices = indices[:size]

    train_size = int(train_ratio * len(indices))
    val_size = int(val_ratio * len(indices))

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]
    test_samples = [full_dataset.samples[i] for i in test_indices]

    return full_dataset, train_samples, val_samples, test_samples
