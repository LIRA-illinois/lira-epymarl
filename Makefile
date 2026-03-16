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


# g is a space-delimited string for GPU indices to use
# make run_experiment e=exp_1 g=0 2 3
# default value for gpus
# run_sweep:
# 	@bash src/experiments/run_sweep.bash -e ${e} -g "${g}"
	python src/experiments/run_sweep.py --experiment=${e} --gpus=${gpus}

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


