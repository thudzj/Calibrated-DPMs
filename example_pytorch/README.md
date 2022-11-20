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
    t 1.0 score_norm_mean 55.39274978637695 score_mean_norm 0.27128490805625916 ratio 204.18663024902344
    t 0.9472993016242981 score_norm_mean 55.390724182128906 score_mean_norm 0.26972696185112 ratio 205.35850524902344
    t 0.8915138244628906 score_norm_mean 55.394954681396484 score_mean_norm 0.2690954804420471 ratio 205.8561248779297
    t 0.8320419192314148 score_norm_mean 55.386234283447266 score_mean_norm 0.2627966105937958 ratio 210.7570343017578
    t 0.768078088760376 score_norm_mean 55.38338851928711 score_mean_norm 0.2518022954463959 ratio 219.94790649414062
    t 0.6985347867012024 score_norm_mean 55.350669860839844 score_mean_norm 0.2519852817058563 ratio 219.65834045410156
    t 0.6219791173934937 score_norm_mean 55.30061340332031 score_mean_norm 0.24712510406970978 ratio 223.77578735351562
    t 0.5367425084114075 score_norm_mean 55.21510314941406 score_mean_norm 0.2498999387025833 ratio 220.9488525390625
    t 0.44176021218299866 score_norm_mean 55.0792350769043 score_mean_norm 0.24694347381591797 ratio 223.0438995361328
    t 0.3392816483974457 score_norm_mean 54.89815902709961 score_mean_norm 0.25330880284309387 ratio 216.7242431640625
    t 0.23886336386203766 score_norm_mean 54.66322326660156 score_mean_norm 0.24872027337551117 ratio 219.7779083251953
    t 0.1548045426607132 score_norm_mean 54.258968353271484 score_mean_norm 0.24373413622379303 ratio 222.61538696289062
    t 0.09462288022041321 score_norm_mean 53.57392501831055 score_mean_norm 0.24493654072284698 ratio 218.72573852539062
    t 0.05569545924663544 score_norm_mean 52.617156982421875 score_mean_norm 0.2350705862045288 ratio 223.83555603027344
    t 0.03175191953778267 score_norm_mean 51.22690963745117 score_mean_norm 0.22869795560836792 ratio 223.99374389648438
    t 0.0173982921987772 score_norm_mean 49.34532928466797 score_mean_norm 0.22565482556819916 ratio 218.6761474609375
    t 0.008995837531983852 score_norm_mean 46.865142822265625 score_mean_norm 0.21142055094242096 ratio 221.66786193847656
    t 0.004284591414034367 score_norm_mean 43.843994140625 score_mean_norm 0.20189471542835236 ratio 217.16265869140625
    t 0.001849339110776782 score_norm_mean 40.52711868286133 score_mean_norm 0.19213584065437317 ratio 210.92950439453125
    t 0.0004506578843574971 score_norm_mean 31.700214385986328 score_mean_norm 0.22502508759498596 ratio 140.87413024902344
CUDA_VISIBLE_DEVICES=1 python main.py --config bedroom_guided.yml --exp=experiments/lsun_bedroom --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4-20 --score_mean --subsample 100000
    t 1.0 score_norm_mean 443.396240234375 score_mean_norm 1.4064375162124634 ratio 315.261962890625
    t 0.9472993016242981 score_norm_mean 443.43023681640625 score_mean_norm 1.4033536911010742 ratio 315.97894287109375
CUDA_VISIBLE_DEVICES=2 python main.py --config celeba.yml --exp=experiments/celeba --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-4 --dpm_solver_fast -i sec1.4 --score_mean
    t 1.0 score_norm_mean 110.82925415039062 score_mean_norm 0.27838483452796936 ratio 398.1152648925781
    t 0.9472993016242981 score_norm_mean 110.82550811767578 score_mean_norm 0.28349971771240234 ratio 390.9192810058594
    t 0.8915138244628906 score_norm_mean 110.82233428955078 score_mean_norm 0.2784809470176697 ratio 397.9530334472656
    t 0.8320419192314148 score_norm_mean 110.82142639160156 score_mean_norm 0.28152087330818176 ratio 393.6526184082031
    t 0.768078088760376 score_norm_mean 110.81732177734375 score_mean_norm 0.2767307758331299 ratio 400.4517517089844
    t 0.6985347867012024 score_norm_mean 110.78805541992188 score_mean_norm 0.2804817855358124 ratio 394.9919738769531
    t 0.6219791173934937 score_norm_mean 110.75575256347656 score_mean_norm 0.28346318006515503 ratio 390.7236022949219
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
CUDA_VISIBLE_DEVICES=3 python main.py --config imagenet64.yml --exp=experiments/imagenet64 --sample --fid --timesteps=20 --eta 0 --ni --skip_type=logSNR --sample_type=dpm_solver --start_time=1e-3 --dpm_solver_fast -i sec1.4-20 --score_mean --subsample 200000
    t 0.9945999383926392 score_norm_mean 110.81407928466797 score_mean_norm 0.24464991688728333 ratio 452.9495849609375
    t 0.9913787841796875 score_norm_mean 110.8129653930664 score_mean_norm 0.2485310286283493 ratio 445.87176513671875
    t 0.9862370491027832 score_norm_mean 110.80831909179688 score_mean_norm 0.25148215889930725 ratio 440.6210021972656
    t 0.9780318737030029 score_norm_mean 110.80626678466797 score_mean_norm 0.24695776402950287 ratio 448.6850891113281
    t 0.9649478197097778 score_norm_mean 110.77900695800781 score_mean_norm 0.24660000205039978 ratio 449.2254943847656
    t 0.9441230297088623 score_norm_mean 110.74629974365234 score_mean_norm 0.24762247502803802 ratio 447.2384948730469
    t 0.9111354947090149 score_norm_mean 110.68812561035156 score_mean_norm 0.24804192781448364 ratio 446.2476501464844
    t 0.8594980239868164 score_norm_mean 110.5361099243164 score_mean_norm 0.24932904541492462 ratio 443.3342590332031
    t 0.7809511423110962 score_norm_mean 110.31694793701172 score_mean_norm 0.24668055772781372 ratio 447.2056884765625
    t 0.6689164638519287 score_norm_mean 109.95072937011719 score_mean_norm 0.24829931557178497 ratio 442.8152770996094
    t 0.5277982354164124 score_norm_mean 109.4310302734375 score_mean_norm 0.2479342371225357 ratio 441.3711853027344
    t 0.3803572952747345 score_norm_mean 108.7581787109375 score_mean_norm 0.24296440184116364 ratio 447.6300964355469
    t 0.25443965196609497 score_norm_mean 107.80204010009766 score_mean_norm 0.24373891949653625 ratio 442.2848815917969
    t 0.16220487654209137 score_norm_mean 106.36902618408203 score_mean_norm 0.24401965737342834 ratio 435.90350341796875
    t 0.10031717270612717 score_norm_mean 104.32540893554688 score_mean_norm 0.23870442807674408 ratio 437.0484924316406
    t 0.06051749736070633 score_norm_mean 101.51419067382812 score_mean_norm 0.2331749051809311 ratio 435.3564147949219
    t 0.035462185740470886 score_norm_mean 97.86880493164062 score_mean_norm 0.22929151356220245 ratio 426.8313293457031
    t 0.01994980499148369 score_norm_mean 93.33474731445312 score_mean_norm 0.22176408767700195 ratio 420.8740539550781
    t 0.01058944221585989 score_norm_mean 87.7487564086914 score_mean_norm 0.2178199142217636 ratio 402.8500061035156
    t 0.003542674705386162 score_norm_mean 77.2595443725586 score_mean_norm 0.21784920990467072 ratio 354.6468811035156
```