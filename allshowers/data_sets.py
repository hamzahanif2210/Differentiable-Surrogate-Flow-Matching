import os
import time
import warnings
from typing import TypedDict

import showerdata
import torch
from torch import Tensor

from allshowers.data_loader import DataLoader, DictDataSet, ModelInputDict
from allshowers.preprocessing import Identity, Transformation, compose

__all__ = ["create_label_list", "to_label_tensor", "get_data_loaders"]


class ShowerDict(TypedDict):
    shower: Tensor
    energy: Tensor
    cellsize: Tensor    # (N, 3) - physical cell dimensions in x, y, z
    material: Tensor    # (N,) - material index: 0=PbWO4, 1=PbF2
    noise: Tensor | None


def batched_histogram(
    data: torch.Tensor, mask: torch.Tensor, num_bins: int = -1
) -> torch.Tensor:
    if num_bins < 0:
        num_bins = int(torch.max(data[mask]).item()) + 1
    histograms = torch.zeros(size=(data.shape[0], num_bins), dtype=torch.int32)
    ones = torch.zeros(size=data.shape, dtype=histograms.dtype)
    ones[mask] = 1
    histograms.scatter_add_(1, data, ones)
    return histograms


@torch.no_grad()
def initialise_trafos(
    energies: Tensor,
    showers: Tensor,
    cellsize: Tensor,
    mask: Tensor,
    samples_energy_trafo: Transformation,
    samples_coordinate_trafo: Transformation,
    cond_trafo: Transformation,
    cellsize_trafo: Transformation,
    *,
    trafos_file: str = "",
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
):
    if trafos_file is None and world_size > 1:
        raise ValueError(
            "If using distributed training, a trafos_file must be provided to save and load the transformations."
        )
    if world_size > 1:
        torch.distributed.barrier(device_ids=[local_rank])
    if rank != 0:
        torch.distributed.barrier(device_ids=[local_rank])
    if os.path.isfile(trafos_file):
        if world_size > 1 and rank == 0:
            torch.distributed.barrier(device_ids=[local_rank])
        parameters = torch.load(trafos_file, weights_only=True)
        samples_energy_trafo.load_state_dict(parameters["samples_energy_trafo"])
        samples_coordinate_trafo.load_state_dict(parameters["samples_coordinate_trafo"])
        cond_trafo.load_state_dict(parameters["cond_trafo"])
        if "cellsize_trafo" in parameters:
            cellsize_trafo.load_state_dict(parameters["cellsize_trafo"])
        print(f"[rank {rank}] Loaded transformations from {trafos_file}")
    else:
        if rank != 0:
            raise RuntimeError(
                "Initialization of transformations is only allowed for rank 0"
            )
        energies_l = energies[:100_000]
        showers_l = showers[:100_000]
        cellsize_l = cellsize[:100_000]
        mask_l = mask[:100_000]
        cond_trafo.fit(energies_l)
        samples_coordinate_trafo.fit(showers_l[:, :, :3], mask_l)   # x, y, z
        samples_energy_trafo.fit(showers_l[:, :, 3], mask_l.squeeze())
        cellsize_trafo.fit(cellsize_l)
        if trafos_file:
            parameters = {
                "samples_energy_trafo": samples_energy_trafo.state_dict(),
                "samples_coordinate_trafo": samples_coordinate_trafo.state_dict(),
                "cond_trafo": cond_trafo.state_dict(),
                "cellsize_trafo": cellsize_trafo.state_dict(),
            }
            torch.save(parameters, trafos_file)
            print(f"[rank {rank}] Saved transformations to {trafos_file}")
        if world_size > 1:
            time.sleep(5)  # make sure file is on network drive
            torch.distributed.barrier(device_ids=[local_rank])


def load_data(
    path: str,
    *,
    start: int = 0,
    stop: int | None = None,
    return_noise: bool = False,
    max_num_points: int | None = None,
) -> ShowerDict:
    showers = showerdata.load(
        path,
        start,
        stop,
        max_points=max_num_points,
    )
    if return_noise:
        noise, _ = showerdata.load_target(path, "target", start=start, stop=stop)
    else:
        noise = None
    if showers.points.shape[2] == 5:
        showers.points = showers.points[:, :, :4]

    # Load cellsize (N, 3): physical cell dimensions in x, y, z.
    # Expected from showers.cellsizes or a 'cellsizes' dataset in the HDF5 file.
    if hasattr(showers, "cellsizes"):
        cellsize = torch.from_numpy(showers.cellsizes)
    else:
        try:
            import h5py
            with h5py.File(path, "r") as f:
                sl = slice(start, stop)
                cellsize = torch.from_numpy(f["cellsizes"][sl].astype("float32"))
        except Exception:
            warnings.warn(
                "cellsizes not found in data file; using zeros. "
                "Provide a 'cellsizes' dataset (shape N×3) in the HDF5 file."
            )
            cellsize = torch.zeros(showers.points.shape[0], 3, dtype=torch.float32)

    # Material codes stored in the pdg field; mapped to 0/1 via to_label_tensor.
    material = to_label_tensor(torch.from_numpy(showers.pdg))

    data = ShowerDict(
        shower=torch.from_numpy(showers.points),
        energy=torch.from_numpy(showers.energies),
        cellsize=cellsize,
        material=material,
        noise=torch.from_numpy(noise) if noise is not None else None,
    )

    return data


@torch.no_grad()
def create_label_list(
    pdg: torch.Tensor,
) -> list[int]:
    unique_pdg = pdg.unique().tolist()
    unique_pdg.sort(key=lambda x: (abs(x), -x))
    return unique_pdg


@torch.no_grad()
def to_label_tensor(
    pdg: torch.Tensor | None,
    label_list: list[int] | None = None,
) -> torch.Tensor | None:
    if pdg is None:
        return None
    if label_list is None:
        label_list = create_label_list(pdg)
    if max(pdg.shape, default=1) != pdg.numel():
        raise ValueError("pdg must be a 1D tensor.")
    pdg = pdg.view(-1)
    label_tensor = torch.zeros(pdg.shape[0], dtype=torch.int64)
    for i, label in enumerate(label_list):
        label_tensor[pdg == label] = i
    return label_tensor


@torch.no_grad()
def load_and_prepare(
    path: str,
    *,
    samples_energy_trafo: Transformation = Identity(),
    samples_coordinate_trafo: Transformation = Identity(),
    cond_trafo: Transformation = Identity(),
    cellsize_trafo: Transformation = Identity(),
    start: int = 0,
    stop: int | None = None,
    return_noise: bool = False,
    max_num_points: int | None = None,
    num_layers: int = -1,
    do_initialise_trafos: bool = True,
    trafos_file: str = "",
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
) -> ModelInputDict:
    data = load_data(
        path,
        start=start,
        stop=stop,
        return_noise=return_noise,
        max_num_points=max_num_points,
    )
    mask = data["shower"][:, :, [3]] > 0   # energy at index 3

    if do_initialise_trafos:
        initialise_trafos(
            data["energy"],
            data["shower"],
            data["cellsize"],
            mask,
            samples_energy_trafo,
            samples_coordinate_trafo,
            cond_trafo,
            cellsize_trafo,
            trafos_file=trafos_file,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )

    # Point features: [x, y, z_actual, energy]  (4-dimensional)
    x = torch.concat(
        [
            samples_coordinate_trafo(data["shower"][:, :, :3]),   # x, y, z
            samples_energy_trafo(data["shower"][:, :, [3]]),       # energy
        ],
        dim=-1,
    )
    x[~mask.repeat(1, 1, 4)] = 0.0

    # Layer: all zeros — z is a continuous feature, no discrete layer index.
    layer = torch.zeros_like(mask, dtype=torch.long)

    # n_cells: total number of valid hits per event  (replaces per-layer histogram)
    n_cells = mask.squeeze(-1).sum(dim=1, keepdim=True).float()   # (batch, 1)

    # Condition: [primaryE (1), cellsize (3), material binary (1)]
    primaryE = cond_trafo(data["energy"])                           # (batch, 1)
    cellsize = cellsize_trafo(data["cellsize"])                     # (batch, 3)
    material = data["material"].float().unsqueeze(-1)               # (batch, 1)
    cond = torch.concat([primaryE, cellsize, material], dim=-1)    # (batch, 5)

    # Label kept for interface compatibility; material is already encoded in cond.
    label = torch.zeros(data["energy"].shape[0], dtype=torch.int64)

    return ModelInputDict(
        x=x,
        cond=cond,
        num_points=n_cells,
        layer=layer,
        mask=mask,
        label=label,
        noise=data["noise"],
    )


def get_data_loaders(
    config_dataset: dict,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    trafos_file: str = "",
) -> tuple[DataLoader, DataLoader, dict[str, Transformation]]:
    config_dataset = config_dataset.copy()
    data_len = showerdata.get_file_shape(config_dataset["path"])[0]
    if "stop" in config_dataset:
        data_len = min(data_len, config_dataset["stop"])
        del config_dataset["stop"]
    if "val_len" in config_dataset:
        val_len = config_dataset.pop("val_len")
        if val_len > data_len // 2:
            warnings.warn(
                f"val_len {val_len} is larger than 50% of data length {data_len // 2},"
                f" reducing to {data_len // 2}.",
                UserWarning,
            )
            val_len = min(val_len, data_len // 2)
    else:
        val_len = data_len // 10
    split = data_len - val_len
    if "samples_energy_trafo" in config_dataset:
        config_dataset["samples_energy_trafo"] = compose(
            config_dataset["samples_energy_trafo"]
        )
    if "samples_coordinate_trafo" in config_dataset:
        config_dataset["samples_coordinate_trafo"] = compose(
            config_dataset["samples_coordinate_trafo"]
        )
    if "cond_trafo" in config_dataset:
        config_dataset["cond_trafo"] = compose(config_dataset["cond_trafo"])
    if "cellsize_trafo" in config_dataset:
        config_dataset["cellsize_trafo"] = compose(config_dataset["cellsize_trafo"])

    start = rank * (split // world_size)
    stop = (rank + 1) * (split // world_size)
    data_train = DictDataSet(
        load_and_prepare(
            **config_dataset,
            start=start,
            stop=stop,
            trafos_file=trafos_file,
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
        )
    )
    loader_train = DataLoader(
        data_set=data_train,
        batch_size=batch_size,
        drop_last=(stop - start) > batch_size,
        shuffle=True,
    )
    if rank == 0:
        data_test = DictDataSet(
            load_and_prepare(
                **config_dataset,
                start=split,
                stop=data_len,
                trafos_file=trafos_file,
                do_initialise_trafos=False,
            )
        )
        loader_test = DataLoader(
            data_set=data_test, batch_size=batch_size, drop_last=False, shuffle=False
        )
    else:
        loader_test = DataLoader(
            data_set=DictDataSet(
                ModelInputDict(
                    x=torch.empty(0, 0, 0),
                    cond=torch.empty(0, 0),
                    num_points=torch.empty(0, 1),
                    layer=torch.empty(0, 0, dtype=torch.int64),
                    mask=torch.empty(0, 0, dtype=torch.bool),
                    label=torch.empty(0, dtype=torch.int64),
                    noise=None,
                )
            ),
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
        )
    trafos = {
        "samples_energy_trafo": config_dataset.get("samples_energy_trafo", Identity()),
        "samples_coordinate_trafo": config_dataset.get(
            "samples_coordinate_trafo", Identity()
        ),
        "cond_trafo": config_dataset.get("cond_trafo", Identity()),
        "cellsize_trafo": config_dataset.get("cellsize_trafo", Identity()),
    }
    return loader_train, loader_test, trafos
