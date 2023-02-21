# On Calibrating Diffusion Probabilistic Models

The official code for the paper [On Calibrating Diffusion Probabilistic Models](https://arxiv.org/abs/xx) by Tianyu Pang, Cheng Lu, Chao Du, Min Lin, Shuicheng Yan, and Zhijie Deng.

--------------------
We propose a straightforward method for calibrating diffusion probabilistic models that reduces the values of SM objectives and increases model likelihood lower bounds.

## Usage


### Reproduce CIFAR-10 results on image generation and FID

#### Baseline 
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i baseline
```

#### With calibration
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i our --score_mean 
```

### Reproduce CelebA results on image generation and FID 

#### Baseline
```
CUDA_VISIBLE_DEVICES=0 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=50 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i baseline
```

#### With calibration
```
CUDA_VISIBLE_DEVICES=3 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=50 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i our --score_mean 
```

###  Estimate SDE likelihood
```
# CIFAR-10
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --eta 0 --ni --start_time=1e-4 -i temp --likelihood sde

# CelebA
CUDA_VISIBLE_DEVICES=1 python main.py --config celeba.yml --exp=experiments/celeba --sample --eta 0 --ni --start_time=1e-4 -i temp --likelihood sde
```

### Estimate the average estimated score with EDM
```
cd edm/;

# CIFAR-10
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --master_port 12315 --nproc_per_node=1 generate.py --outdir=generations/cifar10/temp --seeds=0-49999 --subdirs --method our --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl

# ImageNet
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --master_port 12311 --nproc_per_node=1 generate.py --outdir=generations/imagenet/temp --seeds=0-49999 --subdirs --steps=256 --S_churn=40 --S_min=0.05 --S_max=50 --S_noise=1.003 --method our --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl
```

## Thanks

[DPM-solver](https://github.com/LuChengTHU/dpm-solver)

[EDM](https://github.com/NVlabs/edm)