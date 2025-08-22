"""
This script loads hyperparameters from JSON files and trains models on specified datasets using
the `create_dataset_model_and_train` function from `train.py` or its PyTorch equivalent. The results
are saved in the output directories defined in the JSON files.

The `run_experiments` function iterates over model names and dataset names, loading configuration
files from a specified folder, and then calls the appropriate training function based on the
framework (PyTorch or JAX).

Arguments for `run_experiments`:
- `model_names`: List of model architectures to use.
- `dataset_names`: List of datasets to train on.
- `experiment_folder`: Directory containing JSON configuration files.
- `pytorch_experiments`: Boolean indicating whether to use PyTorch (True) or JAX (False).

The script also provides a command-line interface (CLI) for specifying whether to run PyTorch experiments.

Usage:
- Use the `--pytorch_experiments` flag to run experiments with PyTorch; otherwise, JAX is used by default.
"""

import argparse
import json
import diffrax
from train import create_dataset_model_and_train

def run_experiments(model_names, dataset_names, experiment_folder):

    for model_name in model_names:
        for dataset_name in dataset_names:
            with open(
                experiment_folder + f"/{model_name}/{dataset_name}.json", "r"
            ) as file:
                data = json.load(file)

            seeds = data["seeds"]
            data_dir = data["data_dir"]
            output_parent_dir = data["output_parent_dir"]
            lr_scheduler = eval(data["lr_scheduler"])
            num_steps = data["num_steps"]
            print_steps = data["print_steps"]
            batch_size = data["batch_size"]
            metric = data["metric"]
            if model_name == 'LinOSS':
                linoss_discretization = data["linoss_discretization"]
            else:
                linoss_discretization = None
            use_presplit = data["use_presplit"]
            T = data["T"]
            if model_name in ["lru", "S5", "S6", "mamba","LinOSS"]:
                dt0 = None
            else:
                dt0 = float(data["dt0"])
            scale = data["scale"]
            lr = float(data["lr"])
            include_time = data["time"].lower() == "true"
            hidden_dim = int(data["hidden_dim"])
            if model_name in ["log_ncde", "nrde", "ncde"]:
                vf_depth = int(data["vf_depth"])
                vf_width = int(data["vf_width"])
                if model_name in ["log_ncde", "nrde"]:
                    logsig_depth = int(data["depth"])
                    stepsize = int(float(data["stepsize"]))
                else:
                    logsig_depth = 1
                    stepsize = 1
                if model_name == "log_ncde":
                    lambd = float(data["lambd"])
                else:
                    lambd = None
                ssm_dim = None
                num_blocks = None
            else:
                vf_depth = None
                vf_width = None
                logsig_depth = 1
                stepsize = 1
                lambd = None
                ssm_dim = int(data["ssm_dim"])
                num_blocks = int(data["num_blocks"])
            if model_name == "S5" or model_name == "LinOSS":
                ssm_blocks = int(data["ssm_blocks"])
            else:
                ssm_blocks = None
            if dataset_name == "ppg":
                output_step = int(data["output_step"])
            else:
                output_step = 1

            model_args = {
                "num_blocks": num_blocks,
                "hidden_dim": hidden_dim,
                "vf_depth": vf_depth,
                "vf_width": vf_width,
                "ssm_dim": ssm_dim,
                "ssm_blocks": ssm_blocks,
                "dt0": dt0,
                "solver": diffrax.Heun(),
                "stepsize_controller": diffrax.ConstantStepSize(),
                "scale": scale,
                "lambd": lambd,
            }
            run_args = {
                "data_dir": data_dir,
                "use_presplit": use_presplit,
                "dataset_name": dataset_name,
                "output_step": output_step,
                "metric": metric,
                "include_time": include_time,
                "T": T,
                "model_name": model_name,
                "stepsize": stepsize,
                "logsig_depth": logsig_depth,
                "linoss_discretization": linoss_discretization,
                "model_args": model_args,
                "num_steps": num_steps,
                "print_steps": print_steps,
                "lr": lr,
                "lr_scheduler": lr_scheduler,
                "batch_size": batch_size,
                "output_parent_dir": output_parent_dir,
                "id": id,
            }
            run_fn = create_dataset_model_and_train

            for seed in seeds:
                print(f"Running experiment with seed: {seed}")
                run_fn(seed=seed, **run_args)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_name", type=str, default='EigenWorms')
    args = args.parse_args()

    model_names = ["LinOSS"]
    dataset_names = [
        args.dataset_name
    ]
    experiment_folder = "experiment_configs/repeats"

    run_experiments(model_names, dataset_names, experiment_folder)
