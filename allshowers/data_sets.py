import os
import time
import warnings
from typing import TypedDict

import h5py
import showerdata
import torch
import torch.nn.functional as F
from torch import Tensor

from allshowers.data_loader import DataLoader, DictDataSet, ModelInputDict
from allshowers.preprocessing import Identity, Transformation, compose

__all__ = ["MATERIAL_TYPES", "material_to_onehot", "get_data_loaders"]

MATERIAL_TYPES = ["PbWO4", "PbF2"]
NUM_MATERIALS = len(MATERIAL_TYPES)


class ShowerDict(TypedDict):
    shower: Tensor
    energy: Tensor       # primaryE, shape (n, 1)
    cellsize: Tensor     # shape (n, 3)
    material: list       # list of material name strings, length n
    n_cells: Tensor      # shape (n, 1)
    noise: Tensor | None


def material_to_onehot(material: list) -> Tensor:
    indices = []
    for m in material:
        if isinstance(m, bytes):
            m = m.decode("utf-8")
        m = str(m).strip("\x00 \t\n\r")
        if m not in MATERIAL_TYPES:
            raise ValueError(
                f"Unknown material '{m}'. Expected one of {MATERIAL_TYPES}"
            )
        indices.append(MATERIAL_TYPES.index(m))
    return F.one_hot(
        torch.tensor(indices, dtype=torch.long), num_classes=NUM_MATERIALS
    ).float()


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
    cellsizes: Tensor,
    n_cells: Tensor,
    showers: Tensor,
    mask: Tensor,
    samples_energy_trafo: Transformation,
    samples_coordinate_trafo: Transformation,
    cond_trafo: Transformation,
    cellsize_trafo: Transformation,
    n_cells_trafo: Transformation,
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
        if "n_cells_trafo" in parameters:
            n_cells_trafo.load_state_dict(parameters["n_cells_trafo"])
        print(f"[rank {rank}] Loaded transformations from {trafos_file}")
    else:
        if rank != 0:
            raise RuntimeError(
                "Initialization of transformations is only allowed for rank 0"
            )
        n = 100_000
        energies_l = energies[:n]
        cellsizes_l = cellsizes[:n]
        n_cells_l = n_cells[:n]
        showers_l = showers[:n]
        mask_l = mask[:n]
        cond_trafo.fit(energies_l)
        cellsize_trafo.fit(cellsizes_l)
        n_cells_trafo.fit(n_cells_l)
        samples_coordinate_trafo.fit(showers_l[:, :, :2], mask_l)
        samples_energy_trafo.fit(showers_l[:, :, 3], mask_l.squeeze())
        if trafos_file:
            parameters = {
                "samples_energy_trafo": samples_energy_trafo.state_dict(),
                "samples_coordinate_trafo": samples_coordinate_trafo.state_dict(),
                "cond_trafo": cond_trafo.state_dict(),
                "cellsize_trafo": cellsize_trafo.state_dict(),
                "n_cells_trafo": n_cells_trafo.state_dict(),
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

    n = len(showers.energies)
    with h5py.File(path, "r") as f:
        end = stop if stop is not None else start + n
        if "cellsize" in f:
            cellsize = torch.from_numpy(
                f["cellsize"][start:end].astype("float32")
            ).reshape(n, 3)
        else:
            cellsize = torch.zeros(n, 3)

        if "material" in f:
            raw = f["material"][start:end]
            material = [
                m.decode("utf-8").strip("\x00 ") if isinstance(m, bytes) else str(m).strip("\x00 ")
                for m in raw
            ]
        else:
            material = [MATERIAL_TYPES[0]] * n

        if "n_cells" in f:
            n_cells = torch.from_numpy(
                f["n_cells"][start:end].astype("float32")
            ).reshape(n, 1)
        else:
            n_cells = torch.zeros(n, 1)

    return ShowerDict(
        shower=torch.from_numpy(showers.points),
        energy=torch.from_numpy(showers.energies),
        cellsize=cellsize,
        material=material,
        n_cells=n_cells,
        noise=torch.from_numpy(noise) if noise is not None else None,
    )


@torch.no_grad()
def load_and_prepare(
    path: str,
    *,
    samples_energy_trafo: Transformation = Identity(),
    samples_coordinate_trafo: Transformation = Identity(),
    cond_trafo: Transformation = Identity(),
    cellsize_trafo: Transformation = Identity(),
    n_cells_trafo: Transformation = Identity(),
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
    mask = data["shower"][:, :, [3]] > 0

    if do_initialise_trafos:
        initialise_trafos(
            data["energy"],
            data["cellsize"],
            data["n_cells"],
            data["shower"],
            mask,
            samples_energy_trafo,
            samples_coordinate_trafo,
            cond_trafo,
            cellsize_trafo,
            n_cells_trafo,
            trafos_file=trafos_file,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )

    primary_e = cond_trafo(data["energy"])                     # (n, 1)
    cellsize = cellsize_trafo(data["cellsize"])                 # (n, 3)
    material_onehot = material_to_onehot(data["material"])     # (n, 2)
    n_cells = n_cells_trafo(data["n_cells"])                   # (n, 1)
    cond = torch.concat([primary_e, cellsize, material_onehot, n_cells], dim=-1)  # (n, 7)

    x = torch.concat(
        [
            samples_coordinate_trafo(data["shower"][:, :, :2]),
            samples_energy_trafo(data["shower"][:, :, [3]]),
        ],
        dim=-1,
    )
    x[~mask.repeat(1, 1, 3)] = 0.0
    layer = (data["shower"][:, :, [2]] + 0.1).long()
    num_points = batched_histogram(
        data=layer.squeeze(dim=-1),
        mask=mask.squeeze(dim=-1),
        num_bins=num_layers,
    )

    return ModelInputDict(
        x=x,
        cond=cond,
        num_points=num_points,
        layer=layer,
        mask=mask,
        label=torch.zeros(len(x), dtype=torch.int64),
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
    config_dataset.pop("return_direction", None)
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
    if "n_cells_trafo" in config_dataset:
        config_dataset["n_cells_trafo"] = compose(config_dataset["n_cells_trafo"])

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
                    num_points=torch.empty(0, 0, dtype=torch.int64),
                    layer=torch.empty(0, 0, dtype=torch.int64),
                    mask=torch.empty(0, 0, dtype=torch.bool),
                    label=torch.empty(0, 0, dtype=torch.int64),
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
        "n_cells_trafo": config_dataset.get("n_cells_trafo", Identity()),
    }
    return loader_train, loader_test, trafos
