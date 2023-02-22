# On Calibrating Diffusion Probabilistic Models

The official code for the paper [On Calibrating Diffusion Probabilistic Models](https://arxiv.org/abs/2302.10688).

--------------------
We propose a straightforward method for calibrating diffusion probabilistic models that reduces the values of SM objectives and increases model likelihood lower bounds.

## Acknowledgement
The codes are modifed based on the [DPM-solver](https://github.com/LuChengTHU/dpm-solver) and [EDM](https://github.com/NVlabs/edm).


## Reproducing CIFAR-10 results on image generation and FID

The command for computing the FID of **baseline** methods (without calibration):
```python
python main.py --config cifar10.yml \
    --exp=experiments/cifar10 \
    --sample --fid \
    --timesteps=20 \
    --eta 0 --ni \
    --skip_type=logSNR \
    --sample_type=dpm_solver \
    --start_time=1e-4 \
    --dpm_solver_fast -i baseline
```

The command for computing the FID of **our** methods (with calibration):
```python
python main.py --config cifar10.yml \
    --exp=experiments/cifar10 \
    --sample --fid \
    --timesteps=20 \
    --eta 0 --ni \
    --skip_type=logSNR \
    --sample_type=dpm_solver \
    --start_time=1e-4 \
    --dpm_solver_fast -i our --score_mean 
```

## Reproducing CelebA results on image generation and FID 

The command for computing the FID of **baseline** methods (without calibration):
```python
python main.py --config celeba.yml \
    --exp=experiments/celeba \
    --sample --fid \
    --timesteps=50 \
    --eta 0 --ni \
    --skip_type=logSNR \
    --sample_type=dpm_solver \
    --start_time=1e-4 \
    --dpm_solver_fast -i baseline
```

The command for computing the FID of **our** methods (with calibration):
```python
python main.py --config celeba.yml \
    --exp=experiments/celeba \
    --sample --fid \
    --timesteps=50 \
    --eta 0 --ni \
    --skip_type=logSNR \
    --sample_type=dpm_solver \
    --start_time=1e-4 \
    --dpm_solver_fast -i our --score_mean 
```

##  Estimating SDE likelihood
The command for running on **CIFAR-10**:
```python
python main.py --config cifar10.yml \
    --exp=experiments/cifar10 \
    --sample --eta 0 \
    --ni --start_time=1e-4 \
    -i temp --likelihood sde
```

The command for running on **CelebA**:
```python
python main.py --config celeba.yml \
    --exp=experiments/celeba \
    --sample --eta 0 \
    --ni --start_time=1e-4 \
    -i temp --likelihood sde
```

## Estimating the average estimated score with EDM

```python
cd edm/;

# CIFAR-10
python torch.distributed.run --master_port 12315 --nproc_per_node=1 generate.py --outdir=generations/cifar10/temp --seeds=0-49999 --subdirs --method our --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl

# ImageNet
python torch.distributed.run --master_port 12311 --nproc_per_node=1 generate.py --outdir=generations/imagenet/temp --seeds=0-49999 --subdirs --steps=256 --S_churn=40 --S_min=0.05 --S_max=50 --S_noise=1.003 --method our --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl
```
The commands for running on `FFHQ` and `AFHQv2` are similar.
