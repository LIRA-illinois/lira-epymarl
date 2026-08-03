venv_name=.venv
project_name=lira-epymarl

####################
# Experiment management
####################
# shows GPU status, useful for checking how much VRAM is in use
nvidia:
	watch -n 0.2 nvidia-smi

# legacy sync, kinda slow but almost always works
# wandb sync --include-offline --legacy ./results/wandb/*-run-*

# parallel syncing, very fast but can't set n too high due to API upload limits
# doesn't like to work when a run has crashed
# 	wandb beta sync -n 5 ./results/wandb/*-run-*
sync_results_wandb:
	find ./results/wandb/ -type d -name "*-run-*" > wandb_runs.txt
	xargs --arg-file wandb_runs.txt -n 1 -P 5 wandb sync --include-offline
	rm wandb_runs.txt

# default values for these params
# pass in gpus as a space-delimited string like g="0 1 2"
g ?= 0
c ?= lab
# at ~4 GB per run, 20 runs per job is about 80 GB RAM, which works well for requested RAM of 128 GB while leaving plenty of headroom to avoid throttling the system
m ?= 20
r ?= 2
# debugging
d ?= False
ifeq ($(d), True)
    debug = --debug
	cmd = ipdb3 -c continue
else
    debug =
	cmd = python3
endif
run_experiment:
	bash -c 'source .venv/bin/activate; ${cmd} src/experiments/grid_search_experiment.py -e exp_${e} -c ${c} -g ${g} --max_runs_per_job=${m} --n_runners=${r} ${debug}'

activate_venv:
	bash -c 'source .venv/bin/activate; /bin/bash'

tmux_experiments:
	tmux list-panes -a -F "#{session_name}"


# screen_experiments:
# 	screen -ls | grep "exp" | awk "{print $1}" | cut -d"	" -f 2

# find screen sesions with "exp" in them, use cut to grab the session names, adds a prefix and suffix to quit the session, puts commands in a txt file, and opens the txt file. Does NOT stop the experiments, user must choose which sessions to quit and copy + paste the commands into the terminal.
tmux_experiments_cancel:
	@tmux list-panes -a -F "#{session_name}" | sed 's/^/tmux kill-session -t /; s/$$/ /' > tmp.txt
	@nano tmp.txt
	@sleep 0.25
	@rm -r tmp.txt

# with GPU
campus-int-gpu:
	srun --time=00:30:00 --account=huytran1-ic --partition=IllinoisComputes-GPU,eng-research-gpu,csl --nodes=1 --mem=64G --ntasks=64 --gpus-per-node=1 --pty /bin/bash

# get RAM usage by user
# 	watch "mpstat; echo; bash src/utils/get_ram_usage.bash; nvidia-smi"
compute-usage:
	nvidia-smi; bash src/utils/get_cpu_usage.bash; echo; bash src/utils/get_ram_usage.bash

# define formatting for outputs
job_fmt=-O JobID:9,Name:20,Username:10,State:12,TimeUsed:15,TimeLimit:15,NumNodes:7,tres-per-node:20,ReasonList:20,Partition:60
partition_fmt=-O Partition:25,Time:15,Nodes:10
hardware_fmt=-O Partition:25,Nodes:10,CPUs:15,Memory:15,Gres:20,Time:10,NodeList:15

# check the queue to run your code for a given partition
queue:
	squeue -p gpuA40x4 ${job_fmt}
	printf "\n"
	squeue -p gpuA100x4 ${job_fmt}
	printf "\n"
	squeue -p IllinoisComputes-GPU ${job_fmt}
	printf "\n"
	squeue -p eng-research-gpu ${job_fmt}
	printf "\n"
	squeue -p csl ${job_fmt}

jobs:
	watch -n 0.5 squeue -u jheglun2 ${job_fmt}

jobs_cancel:
	@squeue -u jheglun2 -O JobID | sed 's/^/scancel /; s/$$/ \n /'> tmp.txt
	@nano tmp.txt
	@sleep 0.25
	@rm -r tmp.txt


# check which partitions you have access to
partitions:
	sinfo ${partition_fmt}

hardware:
	sinfo -p gpuA40x4 ${hardware_fmt}
	printf "\n"
	sinfo -p gpuA100x4 ${hardware_fmt}
	printf "\n"
	sinfo -p IllinoisComputes-GPU ${hardware_fmt}
	printf "\n"
	sinfo -p eng-research-gpu ${hardware_fmt}
	printf "\n"
	sinfo -p csl ${hardware_fmt}

stop-job:
	scancel $(id)
