### baseline 
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original/1; \
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=50 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original50/1; \ 
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=100 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original100/1; \
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=200 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original200/1
```
    INFO - diffusion.py - 2022-10-27 10:52:33,246 - FID: 3.738854208393377
    INFO - diffusion.py - 2022-10-27 11:24:28,068 - FID: 3.601940116438982 
    INFO - diffusion.py - 2022-10-27 12:13:41,155 - FID: 3.5907486241511606
    INFO - diffusion.py - 2022-10-27 13:50:38,872 - FID: 3.589256629416525

```
CUDA_VISIBLE_DEVICES=4 python main.py --config cifar10-500k.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original-500k/1; \
CUDA_VISIBLE_DEVICES=5 python main.py --config cifar10-500k.yml --exp=experiments/cifar10 --sample --fid --timesteps=50 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original50-500k/1; \ 
CUDA_VISIBLE_DEVICES=5 python main.py --config cifar10-500k.yml --exp=experiments/cifar10 --sample --fid --timesteps=100 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original100-500k/1
CUDA_VISIBLE_DEVICES=4 python main.py --config cifar10-500k.yml --exp=experiments/cifar10 --sample --fid --timesteps=200 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original200-500k/1
```    
    result miss ... (not important)
    INFO - diffusion.py - 2022-10-27 21:53:00,645 - FID: 3.12143993760742
    INFO - diffusion.py - 2022-10-28 06:01:47,249 - FID: 3.111956577103399
    INFO - diffusion.py - 2022-10-28 11:33:46,220 - FID: 3.107512649877549

### sec 1.3
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.3  --tradeoff 0.5
```

### sec 1.4
```
CUDA_VISIBLE_DEVICES=1 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4 --score_mean; \
CUDA_VISIBLE_DEVICES=1 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=50 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-50 --score_mean; \
CUDA_VISIBLE_DEVICES=1 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=100 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-100 --score_mean; \
CUDA_VISIBLE_DEVICES=1 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=200 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-200 --score_mean
```
    INFO - diffusion.py - 2022-10-27 11:00:57,859 - FID: 3.3067414381310414
    INFO - diffusion.py - 2022-10-27 11:51:49,933 - FID: 3.3732914591901135
    INFO - diffusion.py - 2022-10-27 13:32:23,196 - FID: 3.3731234833595067
    INFO - diffusion.py - 2022-10-27 16:51:36,634 - FID: 3.2697309904325493

```
CUDA_VISIBLE_DEVICES=1 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-ss20000 --score_mean --subsample 20000; \
CUDA_VISIBLE_DEVICES=1 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-ss10000 --score_mean --subsample 10000; \
CUDA_VISIBLE_DEVICES=2 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-ss5000 --score_mean --subsample 5000; \
CUDA_VISIBLE_DEVICES=2 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-ss2000 --score_mean --subsample 2000; \
CUDA_VISIBLE_DEVICES=4 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-ss1000 --score_mean --subsample 1000; \
CUDA_VISIBLE_DEVICES=4 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-ss500 --score_mean --subsample 500; \
CUDA_VISIBLE_DEVICES=5 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-ss200 --score_mean --subsample 200; \
CUDA_VISIBLE_DEVICES=5 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-ss100 --score_mean --subsample 100
```
    INFO - diffusion.py - 2022-10-28 12:14:16,628 - FID: 3.294964272956122
    INFO - diffusion.py - 2022-10-28 12:27:16,768 - FID: 3.359413631639711
    INFO - diffusion.py - 2022-10-28 12:11:17,343 - FID: 3.6422519909853577
    INFO - diffusion.py - 2022-10-28 12:22:37,569 - FID: 4.584509411661145
    INFO - diffusion.py - 2022-10-28 12:12:05,549 - FID: 7.965853716052095
    INFO - diffusion.py - 2022-10-28 12:23:08,809 - FID: 18.212223154875005
    INFO - diffusion.py - 2022-10-28 12:10:36,212 - FID: 71.10145469903227
    INFO - diffusion.py - 2022-10-28 12:21:33,887 - FID: 160.91111787111993

```
CUDA_VISIBLE_DEVICES=2 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-n_est2 --score_mean --n_estimates 2; \
CUDA_VISIBLE_DEVICES=2 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-n_est3 --score_mean --n_estimates 3; \
CUDA_VISIBLE_DEVICES=2 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-n_est4 --score_mean --n_estimates 4; \
CUDA_VISIBLE_DEVICES=2 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-n_est5 --score_mean --n_estimates 5
```
    INFO - diffusion.py - 2022-10-27 18:11:45,540 - FID: 3.354387900159054
    INFO - diffusion.py - 2022-10-27 18:53:02,491 - FID: 3.3084812107937864
    INFO - diffusion.py - 2022-10-27 19:44:06,947 - FID: 3.3083470529578562
    INFO - diffusion.py - 2022-10-27 20:45:31,801 - FID: 3.348334196572523

```
CUDA_VISIBLE_DEVICES=2 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-gen_based --score_mean --which_for_score_mean experiments/cifar10/image_samples/original; \
CUDA_VISIBLE_DEVICES=2 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=50 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-gen_based-50 --score_mean --which_for_score_mean experiments/cifar10/image_samples/original50; \
CUDA_VISIBLE_DEVICES=2 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=100 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-gen_based-100 --score_mean --which_for_score_mean experiments/cifar10/image_samples/original100; \
CUDA_VISIBLE_DEVICES=2 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=200 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-gen_based-200 --score_mean --which_for_score_mean experiments/cifar10/image_samples/original200
```
    INFO - diffusion.py - 2022-10-27 11:21:17,874 - FID: 3.532547063222637
    INFO - diffusion.py - 2022-10-27 12:09:28,164 - FID: 3.505615465916094
    INFO - diffusion.py - 2022-10-27 13:45:38,567 - FID: 3.4959770107498116
    INFO - diffusion.py - 2022-10-27 16:59:27,327 - FID: 3.463634532725905
    
```
CUDA_VISIBLE_DEVICES=7 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-gen_based-100k --score_mean --which_for_score_mean experiments/cifar10/image_samples/original-500k --subsample 100000; \
CUDA_VISIBLE_DEVICES=7 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-gen_based-200k --score_mean --which_for_score_mean experiments/cifar10/image_samples/original-500k --subsample 200000; \
CUDA_VISIBLE_DEVICES=7 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-gen_based-300k --score_mean --which_for_score_mean experiments/cifar10/image_samples/original-500k --subsample 300000; \
CUDA_VISIBLE_DEVICES=7 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-gen_based-400k --score_mean --which_for_score_mean experiments/cifar10/image_samples/original-500k --subsample 400000; \
CUDA_VISIBLE_DEVICES=7 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-gen_based-500k --score_mean --which_for_score_mean experiments/cifar10/image_samples/original-500k --subsample 500000
```
    INFO - diffusion.py - 2022-10-27 20:40:43,869 - FID: 3.656082941244847
    INFO - diffusion.py - 2022-10-27 21:32:08,185 - FID: 3.540064630781899
    INFO - diffusion.py - 2022-10-27 22:43:09,849 - FID: 3.5593301463858324
    INFO - diffusion.py - 2022-10-28 00:15:19,710 - FID: 3.6258597854857157
    INFO - diffusion.py - 2022-10-28 02:07:46,889 - FID: 3.5389399955842578

### both
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i both --tradeoff 0.5 --score_mean
```
