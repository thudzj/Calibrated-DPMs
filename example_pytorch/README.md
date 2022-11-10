### baseline 
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original
```
    INFO - diffusion.py - 2022-10-21 13:32:56,939 - FID: 3.7399477568322936
50 steps
    INFO - diffusion.py - 2022-10-21 13:57:51,381 - FID: 3.606656278430762
100 steps
    INFO - diffusion.py - 2022-10-21 14:21:24,784 - FID: 3.5979548931645695
200 steps
    INFO - diffusion.py - 2022-10-21 15:42:58,887 - FID: 3.5917284702100005

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
50 steps
    INFO - diffusion.py - 2022-10-21 13:30:21,138 - FID: 3.3813551795428793
100 steps
    INFO - diffusion.py - 2022-10-21 13:29:58,165 - FID: 3.2917572307267164
200 steps
    INFO - diffusion.py - 2022-10-21 15:32:54,742 - FID: 3.376375179294712
    INFO - diffusion.py - 2022-10-21 20:55:29,118 - FID: 3.376101703370068

### both
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i both --tradeoff 0.5 --score_mean
```



## celba

### baseline
```
CUDA_VISIBLE_DEVICES=0 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original
INFO - diffusion.py - 2022-11-10 11:09:26,018 - FID: 2.7838383327336658


CUDA_VISIBLE_DEVICES=2 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=10 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original-10
INFO - diffusion.py - 2022-11-10 11:47:56,949 - FID: 6.693164847189053
```

```
CUDA_VISIBLE_DEVICES=1 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4 --score_mean 
INFO - diffusion.py - 2022-11-10 13:41:13,447 - FID: 2.30422762078814

CUDA_VISIBLE_DEVICES=3 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=10 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-10 --score_mean 
INFO - diffusion.py - 2022-11-10 13:25:54,331 - FID: 4.586808830224896
```



## lsun_bedroom

### baseline
```
CUDA_VISIBLE_DEVICES=3 python main.py --config bedroom_guided.yml --exp=experiments/lsun_bedroom --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original
```
