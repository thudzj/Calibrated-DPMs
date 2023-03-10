import os
import logging
import time
import glob
from tkinter import E

import blobfile as bf

import numpy as np
import tqdm
from dpm_solver.sampler import OurModelWrapper, NoiseScheduleVP
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.improved_ddpm.unet import UNetModel as ImprovedDDPM_Model
from models.guided_diffusion.unet import UNetModel as GuidedDiffusion_Model
from models.guided_diffusion.unet import EncoderUNetModel as GuidedDiffusion_Classifier
from models.guided_diffusion.unet import SuperResModel as GuidedDiffusion_SRModel
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from evaluate.fid_score import calculate_fid_given_paths

import torchvision.utils as tvu

# import jax
# from jax import numpy as jnp
import matplotlib.pyplot as plt
from PIL import Image


def load_data_for_worker(base_samples, batch_size, cond_class):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if cond_class:
            label_arr = obj["arr_1"]
    buffer = []
    label_buffer = []
    while True:
        for i in range(len(image_arr)):
            buffer.append(image_arr[i])
            if cond_class:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = torch.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if cond_class:
                    res["y"] = torch.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def sample(self):
        if self.config.model.model_type == 'improved_ddpm':
            model = ImprovedDDPM_Model(
                in_channels=self.config.model.in_channels,
                model_channels=self.config.model.model_channels,
                out_channels=self.config.model.out_channels,
                num_res_blocks=self.config.model.num_res_blocks,
                attention_resolutions=self.config.model.attention_resolutions,
                dropout=self.config.model.dropout,
                channel_mult=self.config.model.channel_mult,
                conv_resample=self.config.model.conv_resample,
                dims=self.config.model.dims,
                use_checkpoint=self.config.model.use_checkpoint,
                num_heads=self.config.model.num_heads,
                num_heads_upsample=self.config.model.num_heads_upsample,
                use_scale_shift_norm=self.config.model.use_scale_shift_norm
            )
        elif self.config.model.model_type == "guided_diffusion":
            if self.config.model.is_upsampling:
                model = GuidedDiffusion_SRModel(
                    image_size=self.config.model.large_size,
                    in_channels=self.config.model.in_channels,
                    model_channels=self.config.model.model_channels,
                    out_channels=self.config.model.out_channels,
                    num_res_blocks=self.config.model.num_res_blocks,
                    attention_resolutions=self.config.model.attention_resolutions,
                    dropout=self.config.model.dropout,
                    channel_mult=self.config.model.channel_mult,
                    conv_resample=self.config.model.conv_resample,
                    dims=self.config.model.dims,
                    num_classes=self.config.model.num_classes,
                    use_checkpoint=self.config.model.use_checkpoint,
                    use_fp16=self.config.model.use_fp16,
                    num_heads=self.config.model.num_heads,
                    num_head_channels=self.config.model.num_head_channels,
                    num_heads_upsample=self.config.model.num_heads_upsample,
                    use_scale_shift_norm=self.config.model.use_scale_shift_norm,
                    resblock_updown=self.config.model.resblock_updown,
                    use_new_attention_order=self.config.model.use_new_attention_order,
                )
            else:
                model = GuidedDiffusion_Model(
                    image_size=self.config.model.image_size,
                    in_channels=self.config.model.in_channels,
                    model_channels=self.config.model.model_channels,
                    out_channels=self.config.model.out_channels,
                    num_res_blocks=self.config.model.num_res_blocks,
                    attention_resolutions=self.config.model.attention_resolutions,
                    dropout=self.config.model.dropout,
                    channel_mult=self.config.model.channel_mult,
                    conv_resample=self.config.model.conv_resample,
                    dims=self.config.model.dims,
                    num_classes=self.config.model.num_classes,
                    use_checkpoint=self.config.model.use_checkpoint,
                    use_fp16=self.config.model.use_fp16,
                    num_heads=self.config.model.num_heads,
                    num_head_channels=self.config.model.num_head_channels,
                    num_heads_upsample=self.config.model.num_heads_upsample,
                    use_scale_shift_norm=self.config.model.use_scale_shift_norm,
                    resblock_updown=self.config.model.resblock_updown,
                    use_new_attention_order=self.config.model.use_new_attention_order,
                )
        else:
            model = Model(self.config)

        if "ckpt_dir" in self.config.model.__dict__.keys():
            ckpt_dir = os.path.expanduser(self.config.model.ckpt_dir)
            states = torch.load(
                ckpt_dir,
                map_location=self.config.device,
            )
            model = model.to(self.device)
            if self.config.model.model_type == 'improved_ddpm' or self.config.model.model_type == 'guided_diffusion':
                model.load_state_dict(states, strict=True)
                if self.config.model.use_fp16:
                    model.convert_to_fp16()
                model = torch.nn.DataParallel(model)
            else:
                model = torch.nn.DataParallel(model)
                model.load_state_dict(states[0], strict=True)

            if self.config.model.ema: # for celeba 64x64 in DDIM
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None

            if self.config.sampling.cond_class and not self.config.model.is_upsampling:
                classifier = GuidedDiffusion_Classifier(
                    image_size=self.config.classifier.image_size,
                    in_channels=self.config.classifier.in_channels,
                    model_channels=self.config.classifier.model_channels,
                    out_channels=self.config.classifier.out_channels,
                    num_res_blocks=self.config.classifier.num_res_blocks,
                    attention_resolutions=self.config.classifier.attention_resolutions,
                    channel_mult=self.config.classifier.channel_mult,
                    use_fp16=self.config.classifier.use_fp16,
                    num_head_channels=self.config.classifier.num_head_channels,
                    use_scale_shift_norm=self.config.classifier.use_scale_shift_norm,
                    resblock_updown=self.config.classifier.resblock_updown,
                    pool=self.config.classifier.pool
                )
                ckpt_dir = os.path.expanduser(self.config.classifier.ckpt_dir)
                states = torch.load(
                    ckpt_dir,
                    map_location=self.config.device,
                )
                classifier = classifier.to(self.device)
                classifier.load_state_dict(states, strict=True)
                if self.config.classifier.use_fp16:
                    classifier.convert_to_fp16()
                classifier = torch.nn.DataParallel(classifier)
            else:
                classifier = None
        else:
            classifier = None
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model, classifier=classifier)
            if True: #not os.path.exists(os.path.join(self.args.exp, "fid.npy")):
                logging.info("Begin to compute FID...")
                fid = calculate_fid_given_paths((self.config.sampling.fid_stats_dir, self.args.image_folder), batch_size=self.config.sampling.fid_batch_size, device='cuda:0', dims=2048, num_workers=8)
                logging.info("FID: {}".format(fid))
                np.save(os.path.join(self.args.exp, "fid"), fid)
        elif self.args.likelihood:
            self.cal_likelihood(model)
        # elif self.args.interpolation:
        #     self.sample_interpolation(model)
        # elif self.args.sequence:
        #     self.sample_sequence(model)
        elif self.args.train_recorder:
            self.train_rec(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    @torch.no_grad()
    def cal_likelihood(self, model, n_estimates=1):

        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)

        if args.subsample is None:
            subsample = len(dataset)
        else:
            subsample = args.subsample
        idx = torch.randperm(len(dataset))[:subsample]
        train_loader = data.DataLoader(
            torch.utils.data.Subset(dataset, idx),
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )

        if os.path.exists("score_stats_{}.npz".format(self.config.data.dataset)):
            score_stats = np.load("score_stats_{}.npz".format(self.config.data.dataset))
            stats = score_stats['stats']
            summation = score_stats['summation']
        else:
            stats = []
            summation = 0
            for s in range(0, self.num_timesteps - 1):
            # for s in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 998]:
                t = s + 1
                a_t = (1-self.betas).cumprod(dim=0)[t]
                a_s = (1-self.betas).cumprod(dim=0)[s]
                SNR_t = a_t / (1 - a_t)
                SNR_s = a_s / (1 - a_s)

                score_sum = 0
                score_norm_sum = 0
                n_data = 0

                for (x, _) in train_loader:
                    x = x.to(self.device)
                    x = data_transform(self.config, x)

                    vec_t = (torch.ones(x.shape[0], device=self.device) * t).long()
                    a = (1-self.betas).cumprod(dim=0).index_select(0, vec_t).view(-1, 1, 1, 1)
        
                    for _ in range(n_estimates):
                        eps = torch.randn_like(x)
                        z = x * a.sqrt() + eps * (1.0 - a).sqrt()
                        score = model(z.float(), vec_t)
                        score_sum += score.sum(0)
                        score_norm_sum += (score.flatten(1).norm(dim=1) ** 2).sum()
                    
                    n_data += n_estimates * x.shape[0]
                score_mean = score_sum / n_data

                if 0:
                    im = (score_mean - score_mean.min()) / (score_mean.max() - score_mean.min()) * 255.
                    im = Image.fromarray(im.permute(1, 2, 0).data.cpu().numpy().astype(np.uint8))
                    im.save('score_means_{}/{}.png'.format(self.config.data.dataset, t))
                elif 0:
                    score_mean_grey = torch.sum(score_mean**2, dim=0) # H,W
                    quantile = torch.quantile(score_mean_grey.view(-1), q=0.9)
                    score_mean_grey = torch.where(score_mean_grey > quantile.view(1, 1), 
                        torch.ones_like(score_mean_grey), torch.zeros_like(score_mean_grey)) * 255.
                    im = Image.fromarray(score_mean_grey.data.cpu().numpy().astype(np.uint8))
                    im.save('score_means_{}/{}_quantile.png'.format(self.config.data.dataset, t))
                else:
                    pass

                score_norm_mean = score_norm_sum / n_data
                score_mean_norm = score_mean.view(-1).norm() ** 2
                
                stats.append(np.array([t, score_mean_norm.item(), score_mean_norm.item() * (SNR_s / SNR_t - 1).item(), score_norm_mean.item()]))
                summation += score_mean_norm.item() * (SNR_s / SNR_t - 1)
                print("t", t, "score_norm_mean", score_norm_mean.item(), "score_mean_norm", score_mean_norm.item(), "ratio", (score_norm_mean/score_mean_norm).item())
            stats = np.stack(stats)
            np.savez("score_stats_{}.npz".format(self.config.data.dataset), stats=stats, summation=summation.item())
        print(summation)
        # plot_score_mean_stats(stats)
        return

    def train_rec(self, model):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)

        if args.subsample is None:
            subsample = len(dataset)
        else:
            subsample = args.subsample
        idx = torch.randperm(len(dataset))[:subsample]
        train_loader = data.DataLoader(
            torch.utils.data.Subset(dataset, idx),
            batch_size=1000,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )

        num_steps = 20
        timesteps = list(range(0, self.num_timesteps, self.num_timesteps // num_steps)) + [self.num_timesteps - 1, ]
        print(timesteps)
        
        # estimate the ground truth
        with torch.no_grad():
            score_means = {}
            for t in timesteps:
                score_sum = 0
                n_data = 0

                for (x, _) in tqdm.tqdm(train_loader):
                    x = x.to(self.device)
                    x = data_transform(self.config, x)

                    vec_t = (torch.ones(x.shape[0], device=self.device) * t).long()
                    a = (1-self.betas).cumprod(dim=0).index_select(0, vec_t).view(-1, 1, 1, 1)
        
                    eps = torch.randn_like(x)
                    z = x * a.sqrt() + eps * (1.0 - a).sqrt()
                    score = model(z.float(), vec_t)
                    score_sum += score.sum(0)
                    n_data += x.shape[0]

                score_means[t] = score_sum / n_data

        # define the model h
        # Mapping.
        sizes = [self.num_timesteps, ] + [512, ] * (4 - 1) + [3072, ]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(sizes[-2], sizes[-1]))
        h_fn = torch.nn.Sequential(*layers).to(self.device)
        print(h_fn)

        num_epochs = 100
        optimizer = torch.optim.Adam(h_fn.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        timesteps_torch = torch.from_numpy(np.array(timesteps)).to(self.device).long()
        to_save = []
        for _ in range(num_epochs):
            # test
            errors = []
            for t in timesteps:
                gd = score_means[t].view(-1)
                input = torch.nn.functional.one_hot(torch.tensor([t]), self.num_timesteps).float().to(self.device)
                pred = h_fn(input).squeeze()
                errors.append((gd - pred).norm().item())
            to_save.append(np.array(errors))
            print(_, to_save[-1])

            for (x, __) in train_loader:
                x = x.to(self.device)
                x = data_transform(self.config, x)
                t = timesteps_torch[torch.randint(high=len(timesteps), size=(x.shape[0],))]
                # t = torch.ones_like(t) * t[0]

                with torch.no_grad():
                    a = (1-self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                    eps = torch.randn_like(x)
                
                    z = x * a.sqrt() + eps * (1.0 - a).sqrt()
                    score = model(z.float(), t)

                    # score = torch.stack([score_means[i.item()] for i in t], 0)
                
                input = torch.nn.functional.one_hot(t, self.num_timesteps).float()
                loss = ((h_fn(input) - score.flatten(1)) ** 2).sum(-1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            scheduler.step()
        to_save = np.stack(to_save)
        np.save("h_losses_{}.npz".format(self.config.data.dataset), to_save)
        return

    def sample_fid(self, model, classifier=None):
        config = self.config
        img_id = 0
        total_n_samples = config.sampling.fid_total_samples
        if total_n_samples % config.sampling.batch_size != 0:
            raise ValueError("Total samples for sampling must be divided exactly by config.sampling.batch_size, but got {} and {}".format(total_n_samples, config.sampling.batch_size))
        if len(glob.glob(f"{self.args.image_folder}/*")) == total_n_samples:
            n_rounds = 0
        else:
            n_rounds = total_n_samples // config.sampling.batch_size

        if self.config.model.is_upsampling:
            base_samples_total = load_data_for_worker(self.args.base_samples, config.sampling.batch_size, config.sampling.cond_class)

        # t_list = []
        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                # torch.cuda.synchronize()
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()

                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                if self.config.model.is_upsampling:
                    base_samples = next(base_samples_total)
                else:
                    base_samples = None

                x, classes = self.sample_image(x, model, classifier=classifier, base_samples=base_samples)
                x = inverse_data_transform(config, x)

                # end.record()
                # torch.cuda.synchronize()
                # t_list.append(start.elapsed_time(end))

                for i in range(n):
                    if classes is None:
                        path = os.path.join(self.args.image_folder, f"{img_id}.png")
                    else:
                        path = os.path.join(self.args.image_folder, f"{img_id}_{int(classes[i])}.png")
                    tvu.save_image(x[i], path)
                    img_id += 1

        # # Remove the time evaluation of the first batch, because it contains extra initializations
        # print('time / batch', np.mean(t_list[1:]) / 1000., 'std', np.std(t_list[1:]) / 1000.)

    def sample_sequence(self, model, classifier=None):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False, classifier=classifier)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))
    
    def sample_image(self, x, model, last=True, classifier=None, base_samples=None):
        if not hasattr(self, 'wrapper'):
            noise_schedule = NoiseScheduleVP(schedule=self.config.sampling.schedule)
            self.wrapper = OurModelWrapper(noise_schedule, self.args, self.config, x.device,
                steps=1000, eps=self.args.start_time, skip_type=self.args.skip_type)
        return self._sample_image(x, model, last, classifier, base_samples)

    def _sample_image(self, x, model, last=True, classifier=None, base_samples=None):
        assert last
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.config.sampling.cond_class:
            if self.args.fixed_class is None:
                classes = torch.randint(low=0, high=self.config.data.num_classes, size=(x.shape[0],)).to(x.device)
            else:
                classes = torch.randint(low=self.args.fixed_class, high=self.args.fixed_class + 1, size=(x.shape[0],)).to(x.device)
        else:
            classes = None
        
        if base_samples is None:
            if classes is None:
                model_kwargs = {}
            else:
                model_kwargs = {"y": classes}
        else:
            model_kwargs = {"y": base_samples["y"], "low_res": base_samples["low_res"]}

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps
            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        return torch.split(out, 3, dim=1)[0]
                return out
            xs, _ = generalized_steps(x, seq, model_fn, self.betas, eta=self.args.eta, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=self.config.sampling.classifier_scale, **model_kwargs)
            x = xs[-1]
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps
            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        return torch.split(out, 3, dim=1)[0]
                return out
            xs, _ = ddpm_steps(x, seq, model_fn, self.betas, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=self.config.sampling.classifier_scale, **model_kwargs)
            x = xs[-1]
        elif self.args.sample_type == "dpm_solver":
            from dpm_solver.sampler import NoiseScheduleVP, model_wrapper, DPM_Solver
            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                # If the model outputs both 'mean' and 'variance' (such as improved-DDPM and guided-diffusion),
                # We only use the 'mean' output for DPM-Solver, because DPM-Solver is based on diffusion ODEs.
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        return torch.split(out, 3, dim=1)[0]
                return out
            self.wrapper._model_fn = model_fn
            noise_schedule = NoiseScheduleVP(schedule=self.config.sampling.schedule)
            model_fn_continuous = model_wrapper(
                self.wrapper,
                noise_schedule,
                is_cond_classifier=self.config.sampling.cond_class,
                classifier_fn=classifier,
                classifier_scale=self.config.sampling.classifier_scale,
                time_input_type=self.config.sampling.time_input_type,
                total_N=self.config.sampling.total_N,
                model_kwargs=model_kwargs
            )
            dpm_solver = DPM_Solver(model_fn_continuous, noise_schedule)
            x = dpm_solver.sample(
                x,
                steps=self.args.timesteps,
                eps=self.args.start_time,
                order=self.args.dpm_solver_order,
                skip_type=self.args.skip_type,
                adaptive_step_size=self.args.adaptive_step_size,
                fast_version=self.args.dpm_solver_fast,
                atol=self.args.dpm_solver_atol,
                rtol=self.args.dpm_solver_rtol
            )
        else:
            raise NotImplementedError
        return x, classes

    def test(self):
        pass


def plot_score_mean_stats(stats):

    plt.figure(dpi=100,figsize=(5.5, 5))
    # plt.grid(linestyle = "--")
    plt.plot(stats[:, 0], stats[:, 1], ms = 5, lw=2, color='Blue')
    x_index=['4','8','16','32','64'] #,'128','256'
    _ = plt.xticks([0, 200, 400, 600, 800, 999], ['0', '200', '400', '600', '800', '1000'])
    # plt.legend(loc='best',prop = {'size':14},framealpha=0.3)
    plt.xlabel("T",fontsize=16)
    plt.ylabel("Norm of the average predicted noise",fontsize=16)
    plt.savefig('score_mean_norm.pdf')