venv_name=.venv
project_name=lira-epymarl

####################
# Experiment management
####################
# shows GPU status, useful for checking how much VRAM is in use
nvidia:
	watch -n 0.2 nvidia-smi

sync_to_wandb:
	bash src/experiments/sync_wandb.bash

tb:
	screen -dmS tensorboard_${project_name} bash -c 'source .venv/bin/activate; tensorboard --bind_all --port=6009 --logdir "results/tb_logs/"'

# default values for these params
# pass in gpus as a space-delimited string like g="0 1 2"
g ?= 0
c ?= delta
m ?= 30
r ?= 2
d ?= False
run_experiment:
	@bash -c 'source .venv/bin/activate; python src/experiments/grid_search_experiment.py -e exp_${e} -c ${c} -g ${g} --max_runs_per_job=${m} --n_runners=${r} --debug=${d}'

activate_venv:
	bash -c 'source .venv/bin/activate; /bin/bash'

screen_experiments:
	screen -ls | grep "exp" | awk "{print $1}" | cut -d"	" -f 2

# find screen sesions with "exp" in them, use cut to grab the session names, adds a prefix and suffix to quit the session, puts commands in a txt file, and opens the txt file. Does NOT stop the experiments, user must choose which sessions to quit and copy + paste the commands into the terminal.
screen_experiments_cancel:
	@screen -ls | grep "exp" | awk "{print $1}" | cut -d"	" -f 2 | sed 's/^/screen -X -S /; s/$$/ quit/' > tmp.txt
	@nano tmp.txt
	@sleep 0.25
	@rm -r tmp.txt

# with GPU
campus-int-gpu:
	srun --time=00:30:00 --account=huytran1-ic --partition=IllinoisComputes-GPU,eng-research-gpu,csl --nodes=1 --mem=64G --ntasks=64 --gpus-per-node=1 --pty /bin/bash

# get RAM usage by user
compute-usage:
	watch "echo CPU Usage; mpstat; echo; bash src/utils/get_ram_usage.bash; nvidia-smi"

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