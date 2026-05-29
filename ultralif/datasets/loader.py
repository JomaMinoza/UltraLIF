# -*- coding: utf-8 -*-
"""
Dataset loading for all UltraLIF experiments.

Supports static (torchvision) and neuromorphic (tonic) datasets.

Functions:
    get_dataset: Load train/test DataLoaders for a named dataset.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Root data directory (three levels up: datasets/ -> ultralif/ -> package root -> data/)
_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def get_dataset(
    name: str,
    batch_size: int = 128,
    timesteps: int = 30,
    num_workers: int = 1,
    pin_memory: bool = True,
):
    """
    Load a dataset and return optimized DataLoaders.

    Supported datasets:
        Static:        mnist, fashion, cifar10
        Neuromorphic:  nmnist, dvs_gesture, cifar10_dvs, shd, ssc

    Args:
        name: Dataset identifier (see above).
        batch_size: Samples per batch.
        timesteps: Number of time bins (neuromorphic only).
        num_workers: DataLoader worker processes.
        pin_memory: Pin tensors to GPU memory for faster transfer.

    Returns:
        Tuple of (train_loader, test_loader, input_dim, num_classes).

    Raises:
        ValueError: If name is not a recognized dataset.
        ImportError: If tonic is not installed (neuromorphic datasets).

    Example:
        >>> train, test, in_dim, n_cls = get_dataset('mnist')
        >>> train, test, in_dim, n_cls = get_dataset('shd', timesteps=10)
    """
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    # ------------------------------------------------------------------ Static
    if name == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train = datasets.MNIST(_DATA_DIR, train=True, download=True, transform=transform)
        test = datasets.MNIST(_DATA_DIR, train=False, download=True, transform=transform)
        return (
            DataLoader(train, batch_size, shuffle=True, **loader_kwargs),
            DataLoader(test, batch_size, **loader_kwargs),
            784,
            10,
        )

    elif name == "fashion":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
        )
        train = datasets.FashionMNIST(_DATA_DIR, train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(_DATA_DIR, train=False, download=True, transform=transform)
        return (
            DataLoader(train, batch_size, shuffle=True, **loader_kwargs),
            DataLoader(test, batch_size, **loader_kwargs),
            784,
            10,
        )

    elif name == "cifar10":
        tr_t = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
        ])
        te_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
        ])
        train = datasets.CIFAR10(_DATA_DIR, train=True, download=True, transform=tr_t)
        test = datasets.CIFAR10(_DATA_DIR, train=False, download=True, transform=te_t)
        return (
            DataLoader(train, batch_size, shuffle=True, **loader_kwargs),
            DataLoader(test, batch_size, **loader_kwargs),
            3072,
            10,
        )

    # ------------------------------------------------------ Neuromorphic (tonic)
    try:
        import tonic
        import tonic.transforms as T
    except ImportError:
        raise ImportError(
            "tonic is required for neuromorphic datasets.\n"
            "Install with: pip install tonic"
        )

    def neuro_loader(dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size,
            shuffle=shuffle,
            collate_fn=tonic.collation.PadTensors(batch_first=True),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

    if name == "nmnist":
        sensor_size = tonic.datasets.NMNIST.sensor_size
        transform = T.Compose([T.ToFrame(sensor_size=sensor_size, n_time_bins=timesteps), torch.from_numpy])
        train = tonic.datasets.NMNIST(save_to=str(_DATA_DIR), train=True, transform=transform)
        test = tonic.datasets.NMNIST(save_to=str(_DATA_DIR), train=False, transform=transform)
        return neuro_loader(train, shuffle=True), neuro_loader(test), 2 * 34 * 34, 10

    elif name == "dvs_gesture":
        sensor_size = tonic.datasets.DVSGesture.sensor_size
        transform = T.Compose([T.ToFrame(sensor_size=sensor_size, n_time_bins=timesteps), torch.from_numpy])
        train = tonic.datasets.DVSGesture(save_to=str(_DATA_DIR), train=True, transform=transform)
        test = tonic.datasets.DVSGesture(save_to=str(_DATA_DIR), train=False, transform=transform)
        return neuro_loader(train, shuffle=True), neuro_loader(test), 2 * 128 * 128, 11

    elif name == "cifar10_dvs":
        sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
        transform = T.Compose([T.ToFrame(sensor_size=sensor_size, n_time_bins=timesteps), torch.from_numpy])
        full = tonic.datasets.CIFAR10DVS(save_to=str(_DATA_DIR), transform=transform)
        n_train = int(0.8 * len(full))
        train_set, test_set = torch.utils.data.random_split(
            full, [n_train, len(full) - n_train], generator=torch.Generator().manual_seed(42)
        )
        return neuro_loader(train_set, shuffle=True), neuro_loader(test_set), 2 * 128 * 128, 10

    elif name == "shd":
        transform = T.Compose([T.ToFrame(sensor_size=(700, 1, 1), n_time_bins=timesteps), torch.from_numpy])
        train = tonic.datasets.SHD(save_to=str(_DATA_DIR), train=True, transform=transform)
        test = tonic.datasets.SHD(save_to=str(_DATA_DIR), train=False, transform=transform)
        return neuro_loader(train, shuffle=True), neuro_loader(test), 700, 20

    elif name == "ssc":
        transform = T.Compose([T.ToFrame(sensor_size=(700, 1, 1), n_time_bins=timesteps), torch.from_numpy])
        train = tonic.datasets.SSC(save_to=str(_DATA_DIR), split="train", transform=transform)
        test = tonic.datasets.SSC(save_to=str(_DATA_DIR), split="test", transform=transform)
        return neuro_loader(train, shuffle=True), neuro_loader(test), 700, 35

    raise ValueError(
        f"Unknown dataset: {name!r}. "
        "Options: mnist, fashion, cifar10, nmnist, dvs_gesture, cifar10_dvs, shd, ssc"
    )
