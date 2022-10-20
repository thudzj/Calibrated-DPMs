baseline 

CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml -i original --sample --fid --sample_type dpm_solver --timesteps 10

sec 1.3
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml -i sec1.3 --sample --fid --sample_type dpm_solver --timesteps 10 --tradeoff 0.5

sec 1.4
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml -i sec1.4 --sample --fid --sample_type dpm_solver --timesteps 10 --score_mean 

both
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml -i both --sample --fid --sample_type dpm_solver --timesteps 10 --tradeoff 0.5 --score_mean
