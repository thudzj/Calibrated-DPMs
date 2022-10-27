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
CUDA_VISIBLE_DEVICES=2 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-gen_based --score_mean --which_for_score_mean experiments/cifar10/image_samples/original; \
CUDA_VISIBLE_DEVICES=2 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=50 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-gen_based-50 --score_mean --which_for_score_mean experiments/cifar10/image_samples/original50; \
CUDA_VISIBLE_DEVICES=2 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=100 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-gen_based-100 --score_mean --which_for_score_mean experiments/cifar10/image_samples/original100; \
CUDA_VISIBLE_DEVICES=2 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=200 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-gen_based-200 --score_mean --which_for_score_mean experiments/cifar10/image_samples/original200
```
    INFO - diffusion.py - 2022-10-27 11:21:17,874 - FID: 3.532547063222637
    INFO - diffusion.py - 2022-10-27 12:09:28,164 - FID: 3.505615465916094
    INFO - diffusion.py - 2022-10-27 13:45:38,567 - FID: 3.4959770107498116
    INFO - diffusion.py - 2022-10-27 16:59:27,327 - FID: 3.463634532725905
    

### both
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i both --tradeoff 0.5 --score_mean
```
