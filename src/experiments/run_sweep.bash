#!/bin/bash

# get the experiment config parsing the -e and -g arg
while getopts e:g: option; do
    case "${option}" in
        e)exp_name=${OPTARG};;
        g)avail_gpus=${OPTARG};;
    esac
done

# use all available GPUs by default if computer has GPUs and avail_gpus was not specified as an arg
if [[ -z $avail_gpus ]] ; then
    mapfile -t avail_gpus < <( nvidia-smi --query-gpu=index --format=csv,noheader,nounits )
fi

n_gpus=${#avail_gpus[@]}

# print useful info about the experiment
echo "Using $n_gpus GPUs with hardware indices (${avail_gpus[@]})"
echo
echo "VRAM Usage"

mapfile -t gpu_names < <( nvidia-smi --query-gpu=name --format=csv,noheader )
mapfile -t total_vrams < <( nvidia-smi --query-gpu=memory.total --format=csv,noheader )
mapfile -t used_vrams < <( nvidia-smi --query-gpu=memory.used --format=csv,noheader )
mapfile -t avail_vrams < <( nvidia-smi --query-gpu=memory.free --format=csv,noheader )

for ((gpu_idx = 0; gpu_idx < $n_gpus; gpu_idx++)); do
    echo "${avail_gpus[$gpu_idx]} -- ${used_vrams[$gpu_idx]} / ${total_vrams[$gpu_idx]} (${avail_vrams[$gpu_idx]} available) -- ${gpu_names[$gpu_idx]}"
done

total_ram=$(free -h | grep "Mem:" | awk '{print $2}')
used_ram=$(free -h | grep "Mem:" | awk '{print $3}')
avail_ram=$(free -h | grep "Mem:" | awk '{print $7}')
echo
printf "RAM Usage\n$used_ram / $total_ram ($avail_ram available)\n"

run_path="experiments/${exp_name}/wandb_config.yaml"

# check if user wants to run all commands in the runner file
read -rp "Run sweep now? (y/n) " run_now
# get lowercase input
run_now="${run_now,,}"
if [[ "$run_now" == "y" ]]; then
    echo "Running sweep defined in $run_path"

    # Init sweep and store the output in a temporary file
    wandb sweep $run_path > wandb_tmp.txt 2>&1

    # Extract the sweep ID using awk
    sweep_id=$(awk '/wandb: Run sweep agent with: wandb agent (.+)/' wandb_tmp.txt | awk '{print $NF}')
    echo Creating sweep with ID: $sweep_id
    rm wandb_tmp.txt

    # run the sweep using wandb and all avail gpus
    # NOTE: this only runs n_gpus agents in parallel, will NOT run all sweep scenarios in parallel
    for gpu_idx in ${avail_gpus[@]}; do
        # use no nohup to run deteched
        nohup CUDA_VISIBLE_DEVICES=$gpu_idx wandb agent $sweep_id &
    done
else
    echo "Exiting without running sweep"
fi
