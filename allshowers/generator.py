import argparse
import os
import platform
import sys
import time
import warnings
from typing import Any

import showerdata
import torch
import yaml
from torch import Tensor, nn

from allshowers import flow_matching as fm
from allshowers import transformer
from allshowers.data_sets import to_label_tensor
from allshowers.preprocessing import compose

start = time.perf_counter()


class Generator(nn.Module):
    def __init__(
        self,
        run_dir: str,
        num_timesteps: int = 200,
        compile: bool = False,
        solver: str = "heun",
        resize_factor: float = 1.0,
    ) -> None:
        super().__init__()

        run_params_file = os.path.join(run_dir, "conf.yaml")
        state_dict_file = os.path.join(run_dir, "weights/best.pt")
        if not os.path.exists(run_params_file):
            state_dict_file = os.path.join(run_dir, "weights/best-all.pt")
        trafo_file = os.path.join(run_dir, "preprocessing/trafos.pt")
        if not os.path.exists(trafo_file):
            trafo_file = os.path.join(run_dir, "preprocessing/trafos-all.pt")
        self.result_dir = run_dir
        self.num_timesteps = num_timesteps
        self.do_compile = compile
        self.resize_factor = resize_factor

        with open(run_params_file) as f:
            run_params = yaml.load(f, Loader=yaml.FullLoader)

        self.__init_model(run_params["model"], state_dict_file, solver=solver)
        self.__init_trafo(run_params["data"], trafo_file)
        self.to(torch.get_default_dtype())
        self.max_points = run_params["data"].get("max_num_points", 6016)
        # Number of point features: x, y, z, energy
        self.num_point_features = run_params["model"]["dim_inputs"][0]

    def __init_model(
        self, params: dict[str, Any], state_file: str, solver: str = "heun"
    ) -> None:
        flow_config = params.pop("flow_config") if "flow_config" in params else {}
        flow_config["solver"] = solver
        network = transformer.Transformer(**params)
        state_dict = torch.load(state_file, map_location="cpu", weights_only=True)
        trained_compiled = any("_orig_mod." in key for key in state_dict)
        if trained_compiled and not self.do_compile:
            for k in list(state_dict.keys()):
                if "_orig_mod." in k:
                    new_k = k.replace("_orig_mod.", "")
                    state_dict[new_k] = state_dict.pop(k)
        elif not trained_compiled and self.do_compile:
            for k in list(state_dict.keys()):
                if "network." in k:
                    new_k = k.replace("network.", "network._orig_mod.")
                    state_dict[new_k] = state_dict.pop(k)
        if self.do_compile:
            network = torch.compile(network)
        self.flow = fm.CNF(network, **flow_config)  # type: ignore
        self.flow.load_state_dict(state_dict)

    def __init_trafo(self, params: dict[str, Any], trafo_file: str) -> None:
        self.samples_energy_trafo = compose(params.get("samples_energy_trafo"))
        self.samples_coordinate_trafo = compose(params.get("samples_coordinate_trafo"))
        self.cond_trafo = compose(params.get("cond_trafo"))
        self.cellsize_trafo = compose(params.get("cellsize_trafo"))

        state = torch.load(trafo_file, map_location="cpu", weights_only=True)
        self.samples_energy_trafo.load_state_dict(state["samples_energy_trafo"])
        self.samples_coordinate_trafo.load_state_dict(state["samples_coordinate_trafo"])
        self.cond_trafo.load_state_dict(state["cond_trafo"])
        if "cellsize_trafo" in state:
            self.cellsize_trafo.load_state_dict(state["cellsize_trafo"])

    def forward(
        self,
        energies: Tensor,       # (batch, 1) primary energy
        n_cells: Tensor,        # (batch, 1) number of hits per event
        cellsize: Tensor,       # (batch, 3) physical cell dimensions
        material: Tensor,       # (batch, 1) material binary 0/1
    ) -> Tensor:
        primaryE = self.cond_trafo(energies * self.resize_factor)   # (batch, 1)
        cellsize_t = self.cellsize_trafo(cellsize)                   # (batch, 3)
        material_f = material.float()                                # (batch, 1)
        condition = torch.cat([primaryE, cellsize_t, material_f], dim=-1)  # (batch, 5)

        batch_size = condition.shape[0]
        layer = torch.zeros((batch_size, self.max_points, 1), dtype=torch.int32)
        mask = torch.zeros((batch_size, self.max_points, 1), dtype=torch.bool)
        for i in range(batch_size):
            total_points = int(n_cells[i].item())
            if total_points > self.max_points:
                warnings.warn(
                    f"n_cells {total_points} exceeds max_points {self.max_points}, truncating"
                )
                total_points = self.max_points
            mask[i, :total_points, 0] = True

        layer = layer.to(condition.device)
        mask = mask.to(condition.device)

        # n_cells must be float (batch, 1) for the num_points embedding
        n_cells_f = n_cells.float().to(condition.device)

        raw_samples = self.flow.sample(
            shape=(batch_size, self.max_points, self.num_point_features),
            num_timesteps=self.num_timesteps,
            cond=condition,
            num_points=n_cells_f,
            layer=layer,
            mask=mask,
            label=None,
        )
        samples = torch.zeros(
            (batch_size, self.max_points, self.num_point_features),
            device=raw_samples.device,
        )
        # Inverse-transform coordinates (x, y, z) and energy separately
        samples[:, :, :3] = self.samples_coordinate_trafo.inverse(raw_samples[:, :, :3])
        samples[:, :, 3] = self.samples_energy_trafo.inverse(
            raw_samples[:, :, [3]]
        ).squeeze(-1)
        samples[~mask.repeat(1, 1, self.num_point_features)] = 0
        return samples


def print_time(text):
    now = time.perf_counter()
    print(f"[{int(now - start):6d}s]: {text}")
    sys.stdout.flush()


def generate(
    generator: Generator,
    energies: Tensor,
    n_cells: Tensor,
    cellsize: Tensor,
    material: Tensor,
    batch_size: int | None = None,
    device: str | torch.device = "cpu",
) -> Tensor:
    if batch_size is None:
        batch_size = energies.shape[0]
    split_energies = torch.split(energies, batch_size, dim=0)
    split_n_cells = torch.split(n_cells, batch_size, dim=0)
    split_cellsize = torch.split(cellsize, batch_size, dim=0)
    split_material = torch.split(material, batch_size, dim=0)

    generator = generator.to(device)
    generator.eval()
    samples = []
    for i, batch in enumerate(
        zip(split_energies, split_n_cells, split_cellsize, split_material)
    ):
        print_time(f"start batch {i:3d}")
        batch = [e.to(device) for e in batch]
        samples_l = generator(*batch).cpu()
        samples.append(samples_l)
    samples = torch.cat(samples)
    print_time("generation done")
    return samples


def get_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generates new samples")
    parser.add_argument(
        "run_dir",
        help="directory that contains the model's weights and where the generated samples should be saved",
    )
    parser.add_argument(
        "cond_file",
        help="file with the conditioning information (energies, n_cells, cellsizes, material)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        default=1,
        type=int,
        help="number of samples to generate. default: 1",
    )
    parser.add_argument(
        "-b", "--batch-size", default=1024, type=int, help="default: 1024"
    )
    parser.add_argument("-t", "--num-threads", default=None, type=int)
    parser.add_argument("-d", "--device", default=None, help="device for computations")
    parser.add_argument(
        "--num-timesteps",
        default=200,
        type=int,
        help="number of timesteps for the ODE solver. default: 200",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        type=str,
        help="data type for the generated samples. default: float32",
    )
    parser.add_argument(
        "-r",
        "--rescale-factor",
        default=1.0,
        type=float,
        help="energy rescale factor applied during generation. default: 1.0",
    )
    parser.add_argument(
        "--solver",
        default="heun",
        type=str,
        help="ODE solver to use during generation. default: heun",
    )
    parser.add_argument(
        "--material-codes",
        default=None,
        nargs="+",
        type=int,
        help="integer material codes used in the data (e.g. PDG-equivalent codes for PbWO4 and PbF2)",
    )
    return parser.parse_args(args)


@torch.inference_mode()
def main(args: list[str] | None = None) -> None:
    parsed_args = get_args(args)
    print_time("start main")
    dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if parsed_args.dtype not in dtypes:
        raise ValueError(f"invalid dtype: {parsed_args.dtype}")
    dtype = dtypes[parsed_args.dtype]
    torch.set_default_dtype(dtype)
    torch.set_float32_matmul_precision("high")
    if parsed_args.num_threads:
        torch.set_num_threads(parsed_args.num_threads)
    print(yaml.dump(vars(parsed_args)), end="")
    if parsed_args.device:
        device = parsed_args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    torch.set_default_device(device)
    if "cuda" in device.lower():
        print("device:", torch.cuda.get_device_name(torch.device(device)))
    elif device.lower() == "cpu":
        print("device:", platform.processor())
    print("num threads:", torch.get_num_threads())
    sys.stdout.flush()

    generator = Generator(
        run_dir=parsed_args.run_dir,
        num_timesteps=parsed_args.num_timesteps,
        compile=("cuda" in device.lower()),
        solver=parsed_args.solver,
        resize_factor=parsed_args.rescale_factor,
    )

    # Load conditioning data from file.
    # Expected observables: incident_energies, n_cells, cellsizes, incident_pdg (material codes).
    cond_data = showerdata.observables.read_observables_from_file(
        parsed_args.cond_file,
        observables=[
            "incident_energies",
            "incident_pdg",
            "n_cells",
            "cellsizes",
        ],
        start=-parsed_args.num_samples,
    )
    energies = torch.from_numpy(cond_data["incident_energies"]).to(dtype, copy=False)
    n_cells = torch.from_numpy(cond_data["n_cells"]).float()
    if n_cells.dim() == 1:
        n_cells = n_cells.unsqueeze(-1)   # (N, 1)
    cellsize = torch.from_numpy(cond_data["cellsizes"]).to(dtype, copy=False)
    pdg_raw = torch.from_numpy(cond_data["incident_pdg"])
    material_labels = to_label_tensor(
        pdg=pdg_raw,
        label_list=parsed_args.material_codes,
    )
    material = material_labels.float().unsqueeze(-1)  # (N, 1)

    generator.eval()
    generator = generator.to(device)

    samples = generate(
        generator,
        energies,
        n_cells,
        cellsize,
        material,
        parsed_args.batch_size,
        device,
    )
    showers = showerdata.Showers(
        points=samples.numpy(),
        energies=energies.numpy(),
        directions=None,
        pdg=pdg_raw.numpy(),
    )

    for i in range(100):
        name = f"samples{i:02d}"
        file_path = os.path.join(parsed_args.run_dir, name + ".h5")
        if not os.path.exists(file_path):
            break
    else:
        raise RuntimeError("no free sample file name found")

    showers.save(file_path)
    with open(os.path.join(parsed_args.run_dir, name + ".yaml"), "w") as f:
        yaml.dump(vars(parsed_args), f)

    print(f"saved to {file_path}")
    print_time("all done")


if __name__ == "__main__":
    main()
