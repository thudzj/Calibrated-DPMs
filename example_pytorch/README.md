### baseline 
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original
```
INFO - diffusion.py - 2022-10-21 13:32:56,939 - FID: 3.7399477568322936

### sec 1.3
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.3  --tradeoff 0.5
```
INFO - diffusion.py - 2022-10-21 13:59:10,399 - FID: 428.48679453530116

### sec 1.4
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4 --score_mean 
```
INFO - diffusion.py - 2022-10-21 15:29:33,742 - FID: 3.356828692191584

### both
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i both --tradeoff 0.5 --score_mean
```
