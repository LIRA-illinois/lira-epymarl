venv_name=.venv
project_name=lira-epymarl

####################
# Experiment management
####################
# shows GPU status, useful for checking how much VRAM is in use
nvidia:
	watch -n 0.2 nvidia-smi

tb:
	screen -dmS tensorboard_${project_name} bash -c 'source .venv/bin/activate; tensorboard --bind_all --port=6009 --logdir "results/tb_logs/"'

activate_venv:
	bash -c 'source .venv/bin/activate; /bin/bash'

# g is a space-delimited string for GPU indices to use
# make run_experiment e=exp_1 g=0 2 3
# default value for gpus
# run_sweep:
# 	@bash src/experiments/run_sweep.bash -e ${e} -g "${g}"
# 	python src/experiments/run_sweep.py --experiment=${e} --gpus=${gpus}

# 	sweep_id=$$(wandb sweep experiments/${e}/wandb_config.yaml); \
# 	echo ${sweep_id};


# 	mapfile -t avail_gpus < <( nvidia-smi --query-gpu=index --format=csv,noheader,nounits )
# 	echo $$avail_gpus
# 	@for gpu_idx in $(gpu_idxs); do \
# 		echo Starting wandb agent for ${e} on GPU $${gpu_idx}; \
# 		CUDA_VISIBLE_DEVICES=$${gpu_idx} wandb sweep experiments/${e}/wandb_config.yaml; \
# 	done

#echo $$i;

#src/search.config.wandb.dissc.yaml

list_screen_experiments:
	screen -ls | grep "exp" | awk "{print $1}" | cut -d"	" -f 2

# find screen sesions with "exp" in them, use cut to grab the session names, adds a prefix and suffix to quit the session, puts commands in a txt file, and opens the txt file. Does NOT stop the experiments, user must choose which sessions to quit and copy + paste the commands into the terminal.
list_screen_experiments_quit:
	screen -ls | grep "exp" | awk "{print $1}" | cut -d"	" -f 2 | sed 's/^/screen -X -S /; s/$$/ quit/' > screen_cmds.txt
	code screen_cmds.txt
	sleep 0.25
	rm -r screen_cmds.txt

# with GPU
campus-int-gpu:
	srun --time=00:30:00 --account=huytran1-ic --partition="IllinoisComputes-GPU,eng-research-gpu,csl" --nodes=1 --mem=64G --ntasks=64 --gpus-per-node=1 --pty /bin/bash

# define formatting for outputs
job_fmt=-O JobID:9,Name:45,Username:15,State:12,TimeUsed:15,TimeLimit:15,NumNodes:7,tres-per-node:20,ReasonList:20,Partition:60
partition_fmt=-O Partition:25,Time:15,Nodes:10
hardware_fmt=-O Partition:25,Nodes:10,CPUs:15,Memory:15,Gres:20,Time:10,NodeList:15

# check the queue to run your code for a given partition
queue:
	squeue -p gpuA40x4 ${job_fmt}
	printf "\n"
	squeue -p gpuA100x4 ${job_fmt}
	squeue -p IllinoisComputes-GPU ${job_fmt}
	printf "\n"
	squeue -p eng-research-gpu ${job_fmt}
	printf "\n"
	squeue -p csl ${job_fmt}

jobs:
	watch -n 0.5 squeue -u jheglun2 ${job_fmt}

# check which partitions you have access to
partitions:
	sinfo ${partition_fmt}

hardware:
	sinfo -p IllinoisComputes-GPU ${hardware_fmt}
	printf "\n"
	sinfo -p eng-research-gpu ${hardware_fmt}
	printf "\n"
	sinfo -p csl ${hardware_fmt}

stop-job:
	scancel $(id)