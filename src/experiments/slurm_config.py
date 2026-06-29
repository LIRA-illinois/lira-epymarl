from os.path import join

class Computers:
    cluster = ["campus", "delta"]
    lab = ["lab"]

    # A40 has 48 GB VRAM per GPU, A100 has 80 GB VRAM per GPU, H200 has 141 GB VRAM per GPU
    # The H200 costs a lot in terms of my credits to use, and is a monster GPU in general
    # Any combination of these strings is valid too, so "gpuA40x4,gpuA100x4,gpuA100x8" is a valid partition
    delta_partitions = ["gpuA40x4", "gpuA100x4", "gpuA100x8", "H200x8"]

    # IllinoisComputes-GPU has 5 nodes
    ## 4 with 4, 80 GB A100 GPUs, 512 GB RAM, and 128 CPU cores
    ## 1 with 8, 141GB H200 GPUs, 1.5 TB RAM, and 64 CPU cores
    # eng-research-gpu has 5 nodes, each with 8, 24 GB A10 GPUs, 512 GB RAM, and 64 CPU cores
    # csl has 2 nodes, each with 8, 48 GB L40S GPUs, 1 TB RAM, and 128 CPU cores
    # Any combination of these strings is valid too, so ""IllinoisComputes-GPU,eng-research-gpu"" is a valid partition
    campus_partitions = ["IllinoisComputes-GPU", "eng-research-gpu", "csl"]



def get_slurm_config_lines(config: dict) -> list[str]:
    """get list of strings to be put in the sbatch file"""
    output_strs: list[str] = ["#!/bin/bash"]

    for k, v in config.items():
        output_strs.append(f"#SBATCH --{k}={v}")

    return output_strs


def get_slurm_args(
    experiment: str,
    delta_config: dict,
    campus_config: dict,
    cluster: str,
    job_idx: int = 1,
    time: str = "2-00:00:00",
    memory_gb: int = 128,
    nodes: int = 1,
    n_tasks_per_node: int = 1,
    cpus_per_task: int = 64,
    log_dir: str = "",
) -> dict:
    """configure the computational resources to be used for an experiment on a compute cluster
    Notes on Delta system specs: https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/architecture.html

    Parameters
    ----------
    experiment : str
        name of the experiment
    account : str, optional
        account name, by default "bfke-delta-gpu"
    time : str, optional
        max run time of the experiment in d-hh:mm:ss format, by default "2-00:00:00"
    nodes : int, optional
        number of compute nodes to use for this job, by default 1
        Don't mess with multi-node stuff. You'll run separate jobs rather than having a single big job you need to parallelize across nodes.
    memory_gb : int, optional
        requested memory in GB, by default 128. 256 GB available on 4-GPU nodes, 2 TB on 8-GPU nodes.
    n_tasks_per_node : int, optional
        something to do with parallel processing. Just keep at the default value of 1.
    cpus_per_task : int, optional
        number of CPU cores to use for your task, by default 64.
        64 CPU cores available on the 4-GPU nodes, 128 cores on the 8-GPU A100 node, and 96 cores on the 8-GPU H200 node.
    gpus_per_node : int, optional
        number of GPUs to use in your job, by default 1
    gpu_bind : Literal["closest"], optional
        chose the GPU and CPU that are physically closest to speed things up, by default "closest"
    """

    # put vars in a dict b/c some vars in the config have
    # dashes in their names and you can't do that with a namespace
    config: dict = {
        "job-name": f"{experiment}_job_{job_idx}",
        "time": time,
    }

    match cluster:
        case "delta":
            config["partition"] = delta_config["partition"]

            if campus_config["exclude"] != "":
                config["exclude"] = delta_config["exclude"]

            config["account"] = "bfke-delta-gpu"
            config["nodes"] = nodes
            config["gpus-per-node"] = delta_config["gpus_per_node"]
            config["cpus-per-task"] = cpus_per_task
            config["ntasks-per-node"] = n_tasks_per_node
            config["gpu-bind"] = "closest"

        case "campus":
            config["partition"] = campus_config["partition"]

            if campus_config["exclude"] != "":
                config["exclude"] = campus_config["exclude"]

            config["account"] = "huytran1-ic"
            config["nodes"] = nodes

            # ‑‑ntasks=p  Total number of cores for the batch job. p is how many cores (ntasks) per job or per node (ntasks-per-node) to use (1 through 40) [default: 1 core].
            # https://docs.ncsa.illinois.edu/systems/icc/en/latest/user_guide/running_jobs.html
            config["ntasks"] = cpus_per_task

            # this cluster can also take a "gres" (GPU resources) argument instead of gpus-per-node. gres takes the format "gpu:{gpu_type}:{n_gpus}"
            # EX: gpu:A100:2 requests 2 Nvidia A100 GPUs on whatever partition you're submitting your job tomake
            # this is different from Delta where choosing the partition also chooses
            # the types of GPUs that are available for use
            config["gpus-per-node"] = campus_config["gpus_per_node"]

    config["mem"] = f"{memory_gb}G"
    config["output"] = join(log_dir, f"job_{job_idx}_log.out")
    config["error"] = join(log_dir, f"job_{job_idx}_log.err")

    return config


