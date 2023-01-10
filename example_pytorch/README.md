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

### estimate sde likelihood
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --eta 0 --ni --start_time=1e-4 -i temp --likelihood sde

CUDA_VISIBLE_DEVICES=1 python main.py --config celeba.yml --exp=experiments/celeba --sample --eta 0 --ni --start_time=1e-4 -i temp --likelihood sde

CUDA_VISIBLE_DEVICES=2 python main.py --config imagenet64.yml --exp=experiments/imagenet64 --sample --eta 0 --ni --start_time=1e-3 -i temp --likelihood sde

CUDA_VISIBLE_DEVICES=3 python main.py --config bedroom_guided.yml --exp=experiments/lsun_bedroom --sample --eta 0 --ni --start_time=1e-4 -i temp --likelihood sde

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
subsample all: INFO - diffusion.py - 2022-11-18 19:23:10,769 - FID: 25.198785890239378

CUDA_VISIBLE_DEVICES=0 python main.py --config imagenet64.yml --exp=experiments/imagenet64 --sample --fid --timesteps=12 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-3 --dpm_solver_fast -i sec1.4-12 --score_mean --subsample 200000
INFO - diffusion.py - 2022-11-12 10:48:44,226 - FID: 20.355067868866513
using the train_batch data: INFO - diffusion.py - 2022-11-12 18:38:22,633 - FID: 20.445618085258047
subsample all: INFO - diffusion.py - 2022-11-18 21:45:38,690 - FID: 19.901484729654214

CUDA_VISIBLE_DEVICES=1 python main.py --config imagenet64.yml --exp=experiments/imagenet64 --sample --fid --timesteps=15 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-3 --dpm_solver_fast -i sec1.4-15 --score_mean --subsample 200000
INFO - diffusion.py - 2022-11-12 11:44:29,159 - FID: 19.175664565095246
using the train_batch data: INFO - diffusion.py - 2022-11-12 19:32:28,480 - FID: 19.28727475350115
subsample all: INFO - diffusion.py - 2022-11-19 02:13:55,572 - FID: 19.330613977164433

CUDA_VISIBLE_DEVICES=2 python main.py --config imagenet64.yml --exp=experiments/imagenet64 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-3 --dpm_solver_fast -i sec1.4-20 --score_mean --subsample 200000

using the train_batch data: INFO - diffusion.py - 2022-11-12 20:29:11,099 - FID: 18.651559134764966
```

## lsun_bedroom

### baseline
```
CUDA_VISIBLE_DEVICES=3 python main.py --config bedroom_guided.yml --exp=experiments/lsun_bedroom --sample --fid --timesteps=12 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i original
INFO - diffusion.py - 2022-11-16 01:55:21,777 - FID: 4.141296847025075
```

### sec1.4
```
CUDA_VISIBLE_DEVICES=1 python main.py --config bedroom_guided.yml --exp=experiments/lsun_bedroom --sample --fid --timesteps=12 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4 --score_mean --subsample 100000
INFO - diffusion.py - 2022-11-16 13:59:12,294 - FID: 5.087107856952031
```


## check the score norm ratio

```
CUDA_VISIBLE_DEVICES=0 python main.py --config cifar10.yml --exp=experiments/cifar10 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4 --score_means
    t 1.0 score_norm_mean 55.39272689819336 score_mean_norm 0.27130067348480225 ratio 204.1746826171875
    for genrated data; t 1.0 score_norm_mean 55.418006896972656 score_mean_t_norm 0.27130067348480225 ratio 204.26785278320312
    t 0.9472993016242981 score_norm_mean 55.390708923339844 score_mean_norm 0.26974138617515564 ratio 205.3474578857422
    for genrated data; t 0.9472993016242981 score_norm_mean 55.43699645996094 score_mean_t_norm 0.26974138617515564 ratio 205.51905822753906
    t 0.8915138244628906 score_norm_mean 55.394927978515625 score_mean_norm 0.2691086530685425 ratio 205.84596252441406
    for genrated data; t 0.8915138244628906 score_norm_mean 55.44536590576172 score_mean_t_norm 0.2691086530685425 ratio 206.03338623046875
    t 0.8320419192314148 score_norm_mean 55.386199951171875 score_mean_norm 0.26280683279037476 ratio 210.7487030029297
    for genrated data; t 0.8320419192314148 score_norm_mean 55.477230072021484 score_mean_t_norm 0.26280683279037476 ratio 211.09507751464844
    t 0.768078088760376 score_norm_mean 55.38336944580078 score_mean_norm 0.251809686422348 ratio 219.94137573242188
    for genrated data; t 0.768078088760376 score_norm_mean 55.48709487915039 score_mean_t_norm 0.251809686422348 ratio 220.35330200195312
    t 0.6985347867012024 score_norm_mean 55.35064697265625 score_mean_norm 0.2519877851009369 ratio 219.65606689453125
    for genrated data; t 0.6985347867012024 score_norm_mean 55.466861724853516 score_mean_t_norm 0.2519877851009369 ratio 220.1172637939453
    t 0.6219791173934937 score_norm_mean 55.30058288574219 score_mean_norm 0.24712532758712769 ratio 223.77545166015625
    for genrated data; t 0.6219791173934937 score_norm_mean 55.48015594482422 score_mean_t_norm 0.24712532758712769 ratio 224.50210571289062
    t 0.5367425084114075 score_norm_mean 55.21506881713867 score_mean_norm 0.24989892542362213 ratio 220.94960021972656
    for genrated data; t 0.5367425084114075 score_norm_mean 55.41963577270508 score_mean_t_norm 0.24989892542362213 ratio 221.76820373535156
    t 0.44176021218299866 score_norm_mean 55.0792121887207 score_mean_norm 0.24694305658340454 ratio 223.044189453125
    for genrated data; t 0.44176021218299866 score_norm_mean 55.31357955932617 score_mean_t_norm 0.24694305658340454 ratio 223.99325561523438
    t 0.3392816483974457 score_norm_mean 54.898155212402344 score_mean_norm 0.25330978631973267 ratio 216.723388671875
    for genrated data; t 0.3392816483974457 score_norm_mean 55.192665100097656 score_mean_t_norm 0.25330978631973267 ratio 217.8860321044922
    t 0.23886336386203766 score_norm_mean 54.66322326660156 score_mean_norm 0.24872109293937683 ratio 219.77719116210938
    for genrated data; t 0.23886336386203766 score_norm_mean 54.97321701049805 score_mean_t_norm 0.24872109293937683 ratio 221.02354431152344
    t 0.1548045426607132 score_norm_mean 54.25897216796875 score_mean_norm 0.24373728036880493 ratio 222.61253356933594
    for genrated data; t 0.1548045426607132 score_norm_mean 54.48765563964844 score_mean_t_norm 0.24373728036880493 ratio 223.55076599121094
    t 0.09462288022041321 score_norm_mean 53.57393264770508 score_mean_norm 0.24494238197803497 ratio 218.72055053710938
    for genrated data; t 0.09462288022041321 score_norm_mean 53.79155731201172 score_mean_t_norm 0.24494238197803497 ratio 219.60902404785156
    t 0.05569545924663544 score_norm_mean 52.617164611816406 score_mean_norm 0.23507648706436157 ratio 223.82997131347656
    for genrated data; t 0.05569545924663544 score_norm_mean 52.83177185058594 score_mean_t_norm 0.23507648706436157 ratio 224.74290466308594
    t 0.03175191953778267 score_norm_mean 51.22691345214844 score_mean_norm 0.22870492935180664 ratio 223.98692321777344
    for genrated data; t 0.03175191953778267 score_norm_mean 51.22658920288086 score_mean_t_norm 0.22870492935180664 ratio 223.98550415039062
    t 0.0173982921987772 score_norm_mean 49.34534454345703 score_mean_norm 0.2256598025560379 ratio 218.67140197753906
    for genrated data; t 0.0173982921987772 score_norm_mean 49.20475769042969 score_mean_t_norm 0.2256598025560379 ratio 218.04840087890625
    t 0.008995837531983852 score_norm_mean 46.86515426635742 score_mean_norm 0.211422860622406 ratio 221.66549682617188
    for genrated data; t 0.008995837531983852 score_norm_mean 46.69060516357422 score_mean_t_norm 0.211422860622406 ratio 220.83990478515625
    t 0.004284591414034367 score_norm_mean 43.843994140625 score_mean_norm 0.20189443230628967 ratio 217.1629638671875
    for genrated data; t 0.004284591414034367 score_norm_mean 43.46580505371094 score_mean_t_norm 0.20189443230628967 ratio 215.28976440429688
    t 0.001849339110776782 score_norm_mean 40.527122497558594 score_mean_norm 0.19213886559009552 ratio 210.92620849609375
    for genrated data; t 0.001849339110776782 score_norm_mean 40.00783157348633 score_mean_t_norm 0.19213886559009552 ratio 208.22352600097656
    t 0.0004506578843574971 score_norm_mean 31.700204849243164 score_mean_norm 0.22501231729984283 ratio 140.882080078125
    for genrated data; t 0.0004506578843574971 score_norm_mean 30.572349548339844 score_mean_t_norm 0.22501231729984283 ratio 135.8696746826172
CUDA_VISIBLE_DEVICES=1 python main.py --config bedroom_guided.yml --exp=experiments/lsun_bedroom --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-20 --score_mean --subsample 10000
    t 1.0 score_norm_mean 443.4035949707031 score_mean_norm 4.433732509613037 ratio 100.00684356689453
    for genrated data; t 1.0 score_norm_mean 443.458740234375 score_mean_t_norm 4.433732509613037 ratio 100.01927947998047
    t 0.9472993016242981 score_norm_mean 443.4329833984375 score_mean_norm 4.442672252655029 ratio 99.81221771240234
    for genrated data; t 0.9472993016242981 score_norm_mean 443.5300598144531 score_mean_t_norm 4.442672252655029 ratio 99.83406829833984
    t 0.8915138244628906 score_norm_mean 443.4203796386719 score_mean_norm 4.444706439971924 ratio 99.76370239257812
    for genrated data; t 0.8915138244628906 score_norm_mean 443.5392761230469 score_mean_t_norm 4.444706439971924 ratio 99.79045104980469
    t 0.8320419192314148 score_norm_mean 443.3899841308594 score_mean_norm 1.403130292892456 ratio 316.0005798339844
    t 0.768078088760376 score_norm_mean 443.3462829589844 score_mean_norm 1.405371904373169 ratio 315.4654541015625
    t 0.6985347867012024 score_norm_mean 443.29364013671875 score_mean_norm 1.403102993965149 ratio 315.93804931640625
    t 0.6219791173934937 score_norm_mean 443.21978759765625 score_mean_norm 1.4031082391738892 ratio 315.8842468261719
    t 0.5367425084114075 score_norm_mean 443.0619812011719 score_mean_norm 1.4069222211837769 ratio 314.915771484375
    t 0.44176021218299866 score_norm_mean 442.7196350097656 score_mean_norm 1.4016159772872925 ratio 315.86370849609375
    t 0.3392816483974457 score_norm_mean 442.2220153808594 score_mean_norm 1.399345874786377 ratio 316.0205383300781
    t 0.23886336386203766 score_norm_mean 441.40576171875 score_mean_norm 1.3999086618423462 ratio 315.3103942871094
    t 0.1548045426607132 score_norm_mean 439.94427490234375 score_mean_norm 1.3904343843460083 ratio 316.4078063964844
    t 0.09462288022041321 score_norm_mean 437.8874816894531 score_mean_norm 1.3859484195709229 ratio 315.9479064941406
    t 0.05569545924663544 score_norm_mean 435.2813415527344 score_mean_norm 1.389530062675476 ratio 313.2579650878906
CUDA_VISIBLE_DEVICES=2 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4 --score_mean
    t 1.0 score_norm_mean 110.82925415039062 score_mean_norm 0.278384804725647 ratio 398.1153259277344
    for genrated data; t 1.0 score_norm_mean 110.91140747070312 score_mean_t_norm 0.278384804725647 ratio 398.4104309082031
    t 0.9472993016242981 score_norm_mean 110.82550811767578 score_mean_norm 0.28349974751472473 ratio 390.91925048828125
    for genrated data; t 0.9472993016242981 score_norm_mean 110.92189025878906 score_mean_t_norm 0.28349974751472473 ratio 391.25921630859375
    t 0.8915138244628906 score_norm_mean 110.82233428955078 score_mean_norm 0.2784808576107025 ratio 397.9531555175781
    for genrated data; t 0.8915138244628906 score_norm_mean 110.92904663085938 score_mean_t_norm 0.2784808576107025 ratio 398.3363342285156
    t 0.8320419192314148 score_norm_mean 110.82142639160156 score_mean_norm 0.28152093291282654 ratio 393.65252685546875
    for genrated data; t 0.8320419192314148 score_norm_mean 110.94992065429688 score_mean_t_norm 0.28152093291282654 ratio 394.10894775390625
    t 0.768078088760376 score_norm_mean 110.81732177734375 score_mean_norm 0.27673089504241943 ratio 400.4515686035156
    for genrated data; t 0.768078088760376 score_norm_mean 110.95306396484375 score_mean_t_norm 0.27673089504241943 ratio 400.94207763671875
    t 0.6985347867012024 score_norm_mean 110.78804016113281 score_mean_norm 0.2804817259311676 ratio 394.99200439453125
    for genrated data; t 0.6985347867012024 score_norm_mean 110.9234619140625 score_mean_t_norm 0.2804817259311676 ratio 395.4748229980469
    t 0.6219791173934937 score_norm_mean 110.75575256347656 score_mean_norm 0.2834632098674774 ratio 390.7235412597656
    for genrated data; t 0.6219791173934937 score_norm_mean 110.92813110351562 score_mean_t_norm 0.2834632098674774 ratio 391.3316650390625
    t 0.5367425084114075 score_norm_mean 110.68863677978516 score_mean_norm 0.2883358299732208 ratio 383.8879089355469
    t 0.44176021218299866 score_norm_mean 110.57266998291016 score_mean_norm 0.2864095866680145 ratio 386.0648498535156
    t 0.3392816483974457 score_norm_mean 110.43669891357422 score_mean_norm 0.29596132040023804 ratio 373.1457214355469
    t 0.23886336386203766 score_norm_mean 110.22067260742188 score_mean_norm 0.3025978207588196 ratio 364.2480773925781
    t 0.1548045426607132 score_norm_mean 109.90778350830078 score_mean_norm 0.31470736861228943 ratio 349.238037109375
    t 0.09462288022041321 score_norm_mean 109.34375762939453 score_mean_norm 0.330378919839859 ratio 330.9646911621094
    t 0.05569545924663544 score_norm_mean 108.42180633544922 score_mean_norm 0.35018086433410645 ratio 309.6166076660156
    t 0.03175191953778267 score_norm_mean 106.9240951538086 score_mean_norm 0.3756576478481293 ratio 284.6317443847656
    t 0.0173982921987772 score_norm_mean 104.79335021972656 score_mean_norm 0.4322962462902069 ratio 242.4109649658203
    t 0.008995837531983852 score_norm_mean 102.09196472167969 score_mean_norm 0.55392986536026 ratio 184.3048553466797
    t 0.004284591414034367 score_norm_mean 99.04838562011719 score_mean_norm 0.8025248050689697 ratio 123.42096710205078
    t 0.001849339110776782 score_norm_mean 95.38894653320312 score_mean_norm 1.0879637002944946 ratio 87.67658996582031
    t 0.0004506578843574971 score_norm_mean 78.24842071533203 score_mean_norm 1.5296634435653687 ratio 51.15401077270508
CUDA_VISIBLE_DEVICES=3 python main.py --config imagenet64.yml --exp=experiments/imagenet64 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-3 --dpm_solver_fast -i sec1.4-20 --score_mean --subsample 20000
    t 0.9945999383926392 score_norm_mean 110.80950927734375 score_mean_norm 0.78133225440979 ratio 141.8212432861328
    for genrated data; t 0.9945999383926392 score_norm_mean 110.89753723144531 score_mean_t_norm 0.78133225440979 ratio 141.93389892578125
    t 0.9913787841796875 score_norm_mean 110.81536102294922 score_mean_norm 0.7839657664299011 ratio 141.352294921875
    for genrated data; t 0.9913787841796875 score_norm_mean 110.91912841796875 score_mean_t_norm 0.7839657664299011 ratio 141.4846649169922
    t 0.9862370491027832 score_norm_mean 110.80986022949219 score_mean_norm 0.7869465947151184 ratio 140.8098907470703
    for genrated data; t 0.9862370491027832 score_norm_mean 110.92694854736328 score_mean_t_norm 0.7869465947151184 ratio 140.95867919921875
    t 0.9780318737030029 score_norm_mean 110.81851196289062 score_mean_norm 0.7803673148155212 ratio 142.00814819335938
    for genrated data; t 0.9780318737030029 score_norm_mean 110.9828109741211 score_mean_t_norm 0.7803673148155212 ratio 142.21868896484375
    t 0.9649478197097778 score_norm_mean 110.78204345703125 score_mean_norm 0.7818288803100586 ratio 141.69602966308594
    for genrated data; t 0.9649478197097778 score_norm_mean 110.970458984375 score_mean_t_norm 0.7818288803100586 ratio 141.93701171875
    t 0.9441230297088623 score_norm_mean 110.7484359741211 score_mean_norm 0.7875929474830627 ratio 140.6163330078125
    for genrated data; t 0.9441230297088623 score_norm_mean 110.96013641357422 score_mean_t_norm 0.7875929474830627 ratio 140.8851318359375
    t 0.9111354947090149 score_norm_mean 110.68871307373047 score_mean_norm 0.7809923887252808 ratio 141.72828674316406
    for genrated data; t 0.9111354947090149 score_norm_mean 110.94940948486328 score_mean_t_norm 0.7809923887252808 ratio 142.0620880126953
    t 0.8594980239868164 score_norm_mean 110.53702545166016 score_mean_norm 0.7895810008049011 ratio 139.99453735351562
    for genrated data; t 0.8594980239868164 score_norm_mean 110.8109359741211 score_mean_t_norm 0.7895810008049011 ratio 140.34144592285156
    t 0.7809511423110962 score_norm_mean 110.314208984375 score_mean_norm 0.7776612043380737 ratio 141.85382080078125
    for genrated data; t 0.7809511423110962 score_norm_mean 110.6246337890625 score_mean_t_norm 0.7776612043380737 ratio 142.25299072265625
    t 0.6689164638519287 score_norm_mean 109.95589447021484 score_mean_norm 0.7737168073654175 ratio 142.11387634277344
    for genrated data; t 0.6689164638519287 score_norm_mean 110.22506713867188 score_mean_t_norm 0.7737168073654175 ratio 142.46177673339844
    t 0.5277982354164124 score_norm_mean 109.42994689941406 score_mean_norm 0.77690190076828 ratio 140.85426330566406
    for genrated data; t 0.5277982354164124 score_norm_mean 109.67987823486328 score_mean_t_norm 0.77690190076828 ratio 141.17596435546875
    t 0.3803572952747345 score_norm_mean 108.75397491455078 score_mean_norm 0.7724570035934448 ratio 140.78968811035156
    for genrated data; t 0.3803572952747345 score_norm_mean 108.75320434570312 score_mean_t_norm 0.7724570035934448 ratio 140.78868103027344
    t 0.25443965196609497 score_norm_mean 107.78983306884766 score_mean_norm 0.7639696002006531 ratio 141.09178161621094
    for genrated data; t 0.25443965196609497 score_norm_mean 107.28105163574219 score_mean_t_norm 0.7639696002006531 ratio 140.42581176757812
    t 0.16220487654209137 score_norm_mean 106.34892272949219 score_mean_norm 0.7577733993530273 ratio 140.34396362304688
    for genrated data; t 0.16220487654209137 score_norm_mean 105.51973724365234 score_mean_t_norm 0.7577733993530273 ratio 139.24972534179688
    t 0.10031717270612717 score_norm_mean 104.30731201171875 score_mean_norm 0.7354322671890259 ratio 141.831298828125
    for genrated data; t 0.10031717270612717 score_norm_mean 102.91958618164062 score_mean_t_norm 0.7354322671890259 ratio 139.94435119628906
    t 0.06051749736070633 score_norm_mean 101.49378967285156 score_mean_norm 0.7127761840820312 ratio 142.39222717285156
    for genrated data; t 0.06051749736070633 score_norm_mean 99.36984252929688 score_mean_t_norm 0.7127761840820312 ratio 139.4123992919922
    t 0.035462185740470886 score_norm_mean 97.8561019897461 score_mean_norm 0.6959785223007202 ratio 140.60218811035156
    for genrated data; t 0.035462185740470886 score_norm_mean 95.54957580566406 score_mean_t_norm 0.6959785223007202 ratio 137.28811645507812
    t 0.01994980499148369 score_norm_mean 93.32877349853516 score_mean_norm 0.6652606129646301 ratio 140.28904724121094
    for genrated data; t 0.01994980499148369 score_norm_mean 90.13899230957031 score_mean_t_norm 0.6652606129646301 ratio 135.4942626953125
    t 0.01058944221585989 score_norm_mean 87.75445556640625 score_mean_norm 0.6295714378356934 ratio 139.38760375976562
    for genrated data; t 0.01058944221585989 score_norm_mean 83.06466674804688 score_mean_t_norm 0.6295714378356934 ratio 131.9384307861328
    t 0.003542674705386162 score_norm_mean 77.29254150390625 score_mean_norm 0.5711491107940674 ratio 135.328125
    for genrated data; t 0.003542674705386162 score_norm_mean 72.07686614990234 score_mean_t_norm 0.5711491107940674 ratio 126.19623565673828
```