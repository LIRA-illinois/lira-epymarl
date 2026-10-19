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
	find ./results/wandb/ -type d -name "*-run-*" | xargs -n 1 -P 5 wandb sync

# default values for these params
# pass in gpus as a space-	 string like g="0 1 2"
g ?= 0
c ?= lab
ray_address ?= auto
ray_dashboard ?= --ray-dashboard
ifeq ($(c),ray-cluster)
	ray_options = --ray-address=$(ray_address) $(ray_dashboard)
else
	ray_options =
endif
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
	bash -c 'source .venv/bin/activate; ${cmd} src/experiments/grid_search_experiment.py -e exp_${e} -c ${c} -g ${g} --max_runs_per_job=${m} --n_runners=${r} ${ray_options} ${debug}'

activate_venv:
	bash -c 'source .venv/bin/activate

tmux_experiments:
	tmux list-panes -a -F "#{session_name}"

# Clear W&B offline sync cache and any synced models / data
clean_wandb_cache:
	rm -rf ~/.wandb/cache/*
	rm -rf ~/.cache/wandb

# screen_experiments:
# 	screen -ls | grep "exp" | awk "{print $1}" | cut -d"	" -f 2

tmux_experiments_cancel:
	@tmux list-panes -a -F "#{session_name}" | sed 's/^/tmux kill-session -t /; s/$$/ /' > tmp.txt
	@nano tmp.txt
	@sleep 0.25
	@rm -r tmp.txt

# with GPU
campus-int-gpu:
	srun --time=00:30:00 --account=huytran1-ic --partition=IllinoisComputes-GPU,eng-research-gpu,csl --nodes=1 --mem=64G --ntasks=64 --gpus-per-node=1 --pty /bin/bash

# get RAM usage by u# at ~4 GB per run, 20 runs per job is about 80 GB RAM, which works well for requested RAM of 128 GB while leaving plenty of headroom to avoid throttling the system
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
	printf "\n"# find screen sesions with "exp" in them, use cut to grab the session names, adds a prefix and suffix to quit the session, puts commands in a txt file, and opens the txt file. Does NOT stop the experiments, user must choose which sessions to quit and copy + paste the commands into the terminal.

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

####################
# Ray lab cluster
####################
RAY_BIN ?= .venv/bin/ray
RAY_PYTHON ?= .venv/bin/python
RAY_CLUSTER_CONFIG ?= experiments/ray_cluster.yaml
RAY_REMOTE_BIN ?= /home/jaye/dev/lira-epymarl/.venv/bin/ray
RAY_REMOTE_PROJECT_DIR ?= /home/jaye/dev/lira-epymarl
RAY_REMOTE_BIN_DIR = $(patsubst %/,%,$(dir $(RAY_REMOTE_BIN)))
RAY_HEAD_IP = $(shell $(RAY_PYTHON) -c 'import yaml; config = yaml.safe_load(open("$(RAY_CLUSTER_CONFIG)")); print(config["provider"]["head_ip"])')
RAY_WORKER_IPS = $(shell $(RAY_PYTHON) -c 'import yaml; config = yaml.safe_load(open("$(RAY_CLUSTER_CONFIG)")); print(" ".join(config["provider"]["worker_ips"]))')
RAY_SSH_USER = $(shell $(RAY_PYTHON) -c 'import yaml; config = yaml.safe_load(open("$(RAY_CLUSTER_CONFIG)")); print(config["auth"]["ssh_user"])')
RAY_SSH_KEY = $(shell $(RAY_PYTHON) -c 'import os; import yaml; config = yaml.safe_load(open("$(RAY_CLUSTER_CONFIG)")); print(os.path.expanduser(config["auth"]["ssh_private_key"]))')
RAY_CLUSTER_HOSTS = $(RAY_HEAD_IP) $(RAY_WORKER_IPS)

.PHONY: ray-up ray-status ray-nodes ray-git-status ray-check-hosts ray-remote-path ray-down
ray-up:
	RAY_AUTH_MODE=token RAY_UP_enable_autoscaler_v2=0 $(RAY_BIN) up $(RAY_CLUSTER_CONFIG) -y --no-config-cache -v

ray-status:
	RAY_AUTH_MODE=token $(RAY_BIN) status --address=$(RAY_HEAD_IP):6379

ray-nodes:
	RAY_AUTH_MODE=token $(RAY_BIN) list nodes --address=$(RAY_HEAD_IP):6379 --filter state=ALIVE --format table

ray-git-status:
	@for host in $(RAY_CLUSTER_HOSTS); do \
		echo "===== $(RAY_SSH_USER)@$$host: $(RAY_REMOTE_PROJECT_DIR) ====="; \
		ssh -i "$(RAY_SSH_KEY)" -o BatchMode=yes -o ConnectTimeout=8 "$(RAY_SSH_USER)@$$host" \
			"cd '$(RAY_REMOTE_PROJECT_DIR)' && git status --short --branch --ignore-submodules=none && git submodule foreach --recursive 'git status --short --branch'" || exit 1; \
	done

ray-check-hosts:
	@test -s ~/.ray/auth_token || { echo "Missing local ~/.ray/auth_token"; exit 1; }
	@for host in $(RAY_CLUSTER_HOSTS); do \
		echo "Checking $(RAY_SSH_USER)@$$host"; \
		ssh -i "$(RAY_SSH_KEY)" -o BatchMode=yes -o ConnectTimeout=8 "$(RAY_SSH_USER)@$$host" \
			'test -x "$(RAY_REMOTE_BIN)" && "$(RAY_REMOTE_BIN)" --version && test -s ~/.ray/auth_token' || exit 1; \
	done

ray-remote-path:
	@for host in $(RAY_CLUSTER_HOSTS); do \
		echo "Adding the Ray venv to login PATH on $(RAY_SSH_USER)@$$host"; \
		ssh -i "$(RAY_SSH_KEY)" -o BatchMode=yes -o ConnectTimeout=8 "$(RAY_SSH_USER)@$$host" \
			"touch ~/.bashrc; grep -Fqx 'export PATH=$(RAY_REMOTE_BIN_DIR):\$$PATH' ~/.bashrc || printf '%s\\n' 'export PATH=$(RAY_REMOTE_BIN_DIR):\$$PATH' >> ~/.bashrc" || exit 1; \
	done

ray-down: ray-remote-path
	-RAY_AUTH_MODE=token $(RAY_BIN) down $(RAY_CLUSTER_CONFIG) -y
	@failed=0; for host in $(RAY_CLUSTER_HOSTS); do \
		echo "Stopping leftover Ray processes on $(RAY_SSH_USER)@$$host"; \
		ssh -i "$(RAY_SSH_KEY)" -o BatchMode=yes -o ConnectTimeout=8 "$(RAY_SSH_USER)@$$host" \
			'"$(RAY_REMOTE_BIN)" stop --force' || failed=1; \
	done; exit $$failed
