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
CUDA_VISIBLE_DEVICES=0 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=50 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original-50
INFO - diffusion.py - 2022-11-10 15:16:39,981 - FID: 2.8079302403918973

CUDA_VISIBLE_DEVICES=0 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original
INFO - diffusion.py - 2022-11-10 11:09:26,018 - FID: 2.7838383327336658

CUDA_VISIBLE_DEVICES=1 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=15 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original-15
INFO - diffusion.py - 2022-11-10 14:18:23,657 - FID: 2.9580638547100193

CUDA_VISIBLE_DEVICES=2 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=12 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original-12
INFO - diffusion.py - 2022-11-10 14:10:58,994 - FID: 4.059128224989905

CUDA_VISIBLE_DEVICES=2 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=10 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original-10
INFO - diffusion.py - 2022-11-10 11:47:56,949 - FID: 6.693164847189053
```

### sec1.4
```
CUDA_VISIBLE_DEVICES=3 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=50 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-50 --score_mean 
INFO - diffusion.py - 2022-11-10 19:56:29,272 - FID: 2.491003563268066

CUDA_VISIBLE_DEVICES=1 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4 --score_mean 
INFO - diffusion.py - 2022-11-10 13:41:13,447 - FID: 2.30422762078814

CUDA_VISIBLE_DEVICES=2 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=15 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-15 --score_mean 
INFO - diffusion.py - 2022-11-10 16:30:00,051 - FID: 2.454641486144908

CUDA_VISIBLE_DEVICES=1 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=12 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-12 --score_mean 
INFO - diffusion.py - 2022-11-10 16:12:21,527 - FID: 3.3343069799816476

CUDA_VISIBLE_DEVICES=3 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=10 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-10 --score_mean 
INFO - diffusion.py - 2022-11-10 13:25:54,331 - FID: 4.586808830224896
```

## ImageNet64

### baseline
```
CUDA_VISIBLE_DEVICES=0 python main.py --config imagenet64.yml --exp=experiments/imagenet64 --sample --fid --timesteps=10 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-3 --dpm_solver_fast -i original-10
INFO - diffusion.py - 2022-11-11 02:55:00,970 - FID: 24.85932038951563

CUDA_VISIBLE_DEVICES=0 python main.py --config imagenet64.yml --exp=experiments/imagenet64 --sample --fid --timesteps=12 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-3 --dpm_solver_fast -i original-12
INFO - diffusion.py - 2022-11-11 03:30:50,157 - FID: 20.108340748848832

CUDA_VISIBLE_DEVICES=2 python main.py --config imagenet64.yml --exp=experiments/imagenet64 --sample --fid --timesteps=15 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-3 --dpm_solver_fast -i original-15
INFO - diffusion.py - 2022-11-11 03:39:56,461 - FID: 19.14349412824413

CUDA_VISIBLE_DEVICES=2 python main.py --config imagenet64.yml --exp=experiments/imagenet64 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-3 --dpm_solver_fast -i original-20
INFO - diffusion.py - 2022-11-11 04:36:32,958 - FID: 18.457345889932753
```

### sec1.4
```
CUDA_VISIBLE_DEVICES=3 python main.py --config imagenet64.yml --exp=experiments/imagenet64 --sample --fid --timesteps=10 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-3 --dpm_solver_fast -i sec1.4-10 --score_mean --subsample 200000
INFO - diffusion.py - 2022-11-12 10:21:46,446 - FID: 25.772323133418922
using the train_batch data: INFO - diffusion.py - 2022-11-12 18:11:38,516 - FID: 25.960652226912202

CUDA_VISIBLE_DEVICES=0 python main.py --config imagenet64.yml --exp=experiments/imagenet64 --sample --fid --timesteps=12 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-3 --dpm_solver_fast -i sec1.4-12 --score_mean --subsample 200000
INFO - diffusion.py - 2022-11-12 10:48:44,226 - FID: 20.355067868866513
using the train_batch data: INFO - diffusion.py - 2022-11-12 18:38:22,633 - FID: 20.445618085258047

CUDA_VISIBLE_DEVICES=1 python main.py --config imagenet64.yml --exp=experiments/imagenet64 --sample --fid --timesteps=15 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-3 --dpm_solver_fast -i sec1.4-15 --score_mean --subsample 200000
INFO - diffusion.py - 2022-11-12 11:44:29,159 - FID: 19.175664565095246
using the train_batch data: INFO - diffusion.py - 2022-11-12 19:32:28,480 - FID: 19.28727475350115

CUDA_VISIBLE_DEVICES=2 python main.py --config imagenet64.yml --exp=experiments/imagenet64 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-3 --dpm_solver_fast -i sec1.4-20 --score_mean --subsample 200000

using the train_batch data: INFO - diffusion.py - 2022-11-12 20:29:11,099 - FID: 18.651559134764966
```

## lsun_bedroom

### baseline
```
CUDA_VISIBLE_DEVICES=3 python main.py --config bedroom_guided.yml --exp=experiments/lsun_bedroom --sample --fid --timesteps=12 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original
```

### sec1.4
```
CUDA_VISIBLE_DEVICES=1 python main.py --config bedroom_guided.yml --exp=experiments/lsun_bedroom --sample --fid --timesteps=12 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4 --score_mean --subsample 100000
```