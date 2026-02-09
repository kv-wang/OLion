import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig, 
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy
)

from models.sit import SiT_models
from loss import SILoss
from utils import load_encoders
from muon import Muon
from olion import OLion

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    Handles both regular tensors and DTensors (FSDP2).
    """
    from torch.distributed.tensor import DTensor
    
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        
        # Handle DTensor (FSDP2): convert to regular tensor before updating EMA
        if isinstance(param, DTensor):
            # For DTensor, get the full tensor (all-gather from all ranks)
            param_data = param.full_tensor()
        else:
            param_data = param.data
        
        # Update EMA parameter (always a regular tensor)
        ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def get_muon_params(model):
    """Get parameters that should use Muon/OLion optimization (non-embedding/norm parameters, must be 2D)"""
    muon_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_name = name.lower()
       
        is_embln = any(
            key in param_name
            for key in [
                "wte", "wpe", "embd", "embed", "bias",
                "ln", "norm", "lm_head",
                "output", "final_layer"
            ]
        )
        # 非 embedding/norm 参数且必须是二维参数才使用 Muon/OLion 优化
        if not is_embln and param.ndim == 2:
            muon_params.append(param)
    return muon_params


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    # Configure FSDP if enabled
    fsdp_plugin = None
    if args.use_fsdp:
        state_dict_config = FullStateDictConfig(
            offload_to_cpu=args.fsdp_offload_to_cpu,
            rank0_only=args.fsdp_state_dict_rank0_only
        )
        
        # FSDP2 configuration
        # Convert sharding_strategy to reshard_after_forward for FSDP2
        # FULL_SHARD -> True, SHARD_GRAD_OP -> False
        if hasattr(args, 'fsdp_reshard_after_forward'):
            # BooleanOptionalAction can be True, False, or None (if not specified)
            if args.fsdp_reshard_after_forward is not None:
                reshard_after_forward = args.fsdp_reshard_after_forward
            else:
                # Fallback: convert from old sharding_strategy
                reshard_after_forward = (args.fsdp_sharding_strategy == "FULL_SHARD")
        else:
            # Fallback: convert from old sharding_strategy
            reshard_after_forward = (args.fsdp_sharding_strategy == "FULL_SHARD")
        
        fsdp_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,  # Use FSDP2
            state_dict_config=state_dict_config,
            reshard_after_forward=reshard_after_forward,
            # backward_prefetch is removed in FSDP2 (defaults to BACKWARD_PRE)
            # use_orig_params is removed in FSDP2 (always uses original parameters)
            mixed_precision_policy=None,  # Will use accelerator's mixed_precision
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.no_report else args.report_to,
        project_config=accelerator_project_config,
        fsdp_plugin=fsdp_plugin,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        logger = create_logger(save_dir)
        if args.use_fsdp:
            # Determine reshard_after_forward value for logging
            if hasattr(args, 'fsdp_reshard_after_forward') and args.fsdp_reshard_after_forward is not None:
                reshard_value = args.fsdp_reshard_after_forward
            else:
                reshard_value = (args.fsdp_sharding_strategy == "FULL_SHARD")
            reshard_info = f"reshard_after_forward={reshard_value}"
            logger.info(f"Using FSDP2 with {reshard_info}")
        else:
            logger.info("Using DDP (Distributed Data Parallel)")
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints/{args.optimizer}"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    if args.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
            )
    else:
        raise NotImplementedError()
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    vae = AutoencoderKL.from_pretrained(f"/root/autodl-tmp/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)
    
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    elif args.optimizer.lower() == "muon":
        # Split parameters for MUON and AdamW
        muon_params = get_muon_params(model)
        all_trainable_params = set(p for p in model.parameters() if p.requires_grad)
        muon_params_set = set(muon_params)
        adamw_params = [p for p in all_trainable_params if p not in muon_params_set]
        
        param_groups = []
        if muon_params:
            param_groups.append({
                "params": muon_params,
                "use_muon": True,
                "lr": args.learning_rate,
                "momentum": args.momentum,
                "weight_decay": args.adam_weight_decay,
                "rms_scale": args.rms_scale,
                "nesterov": args.nesterov,
                "ns_steps": args.ns_steps,
            })
        if adamw_params:
            param_groups.append({
                "params": adamw_params,
                "use_muon": False,
                "lr": args.learning_rate,
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_epsilon,
                "weight_decay": args.adam_weight_decay,
            })
        
        if not param_groups:
            raise ValueError("No trainable parameters found!")
        
        optimizer = Muon(
            param_groups,
            defaults=dict(lr=args.learning_rate),
            is_deepspeed_enabled=False,
        )
        if accelerator.is_main_process:
            logger.info(f"Using MUON optimizer: {sum(p.numel() for p in muon_params)} MUON params, {sum(p.numel() for p in adamw_params)} AdamW params")
    elif args.optimizer.lower() == "olion":
        # Split parameters for OLion and AdamW
        olion_params = get_muon_params(model)  # Same function for OLion
        all_trainable_params = set(p for p in model.parameters() if p.requires_grad)
        olion_params_set = set(olion_params)
        adamw_params = [p for p in all_trainable_params if p not in olion_params_set]

        param_groups = []
        if olion_params:
            param_groups.append({
                "params": olion_params,
                "use_olion": True,
                "lr": args.learning_rate,
                "momentum": args.momentum,
                "momentum_2": args.olion_momentum_2,
                "weight_decay": args.adam_weight_decay,
                "rms_scale": args.rms_scale,
                "nesterov": args.nesterov,
                "ns_steps": args.ns_steps,
                "eps": args.olion_eps,
                "use_scale": args.olion_use_scale,
            })
        if adamw_params:
            param_groups.append({
                "params": adamw_params,
                "use_olion": False,
                "lr": args.learning_rate,
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_epsilon,
                "weight_decay": args.adam_weight_decay,
            })

        if not param_groups:
            raise ValueError("No trainable parameters found!")

        optimizer = OLion(
            param_groups,
            defaults=dict(lr=args.learning_rate),
            is_deepspeed_enabled=False,
        )
        if accelerator.is_main_process:
            logger.info(f"Using OLion optimizer: {sum(p.numel() for p in olion_params)} OLion params, {sum(p.numel() for p in adamw_params)} AdamW params")
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}. Choose from 'adamw', 'muon', 'olion'")    
    
    # Setup data:
    train_dataset = CustomDataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process and not args.no_report:
        tracker_config = vars(copy.deepcopy(args))
        # Auto-generate wandb run name: {exp_name}-{optimizer}
        wandb_run_name = f"{args.exp_name}-{args.optimizer}"
        accelerator.init_trackers(
            project_name="REPA", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": wandb_run_name}
            },
        )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    gt_raw_images, gt_xs, _ = next(iter(train_dataloader))
    assert gt_raw_images.shape[-1] == args.resolution
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
        )
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)
        
    for epoch in range(args.epochs):
        model.train()
        for raw_image, x, y in train_dataloader:
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            z = None
            if args.legacy:
                # In our early experiments, we accidentally apply label dropping twice: 
                # once in train.py and once in sit.py. 
                # We keep this option for exact reproducibility with previous runs.
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, y)
            else:
                labels = y
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                zs = []
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        zs.append(z)

            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                loss, proj_loss = loss_fn(model, x, model_kwargs, zs=zs)
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                loss = loss_mean + proj_loss_mean * args.proj_coeff
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step == 2 or (global_step % 5000 == 0 and global_step > 0):
                # Handle FSDP state dict saving
                # IMPORTANT: For FSDP, get_state_dict() must be called on ALL processes,
                # not just main_process, because it requires collective communication (all-gather)
                if args.use_fsdp:
                    # For FSDP2, use accelerator.get_state_dict() which handles FSDP properly
                    # This method automatically handles FSDP state dict gathering
                    # All processes must participate in the all-gather operation
                    model_state_dict = accelerator.get_state_dict(model)
                else:
                    # For DDP, only main process needs to get state dict
                    if accelerator.is_main_process:
                        model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    else:
                        model_state_dict = None
                
                # Only main process saves the checkpoint file
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model_state_dict,
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                # Wait for all processes to complete checkpoint saving
                accelerator.wait_for_everyone()

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                from samplers import euler_sampler
                with torch.no_grad():
                    samples = euler_sampler(
                        model, 
                        xT, 
                        ys,
                        num_steps=50, 
                        cfg_scale=4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)
                    samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                if not args.no_report:
                    accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                     "gt_samples": wandb.Image(array2grid(gt_samples))})
                logging.info("Generating EMA samples done.")

            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(), 
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item()
                #"grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            progress_bar.set_postfix(**logs)
            if not args.no_report:
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--no-report", action="store_true", 
                        help="Disable all logging (wandb, etc.). When set, no trackers will be initialized.")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimizer
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon", "olion"],
                        help="Optimizer type: adamw, muon, or olion")
    # Common parameters for MUON and OLion
    parser.add_argument("--momentum", type=float, default=0.95, help="Momentum for MUON/OLion optimizer")
    parser.add_argument("--rms-scale", action=argparse.BooleanOptionalAction, default=True,
                        help="Use RMS scaling for MUON/OLion")
    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True,
                        help="Use Nesterov momentum for MUON/OLion")
    parser.add_argument("--ns-steps", type=int, default=5, help="Number of Newton-Schulz iterations for MUON/OLion")
    # OLion-specific parameters
    parser.add_argument("--olion-momentum-2", type=float, default=0.98, help="Second momentum for OLion optimizer")
    parser.add_argument("--olion-eps", type=float, default=1e-8, help="Epsilon for OLion optimizer")
    parser.add_argument("--olion-use-scale", action=argparse.BooleanOptionalAction, default=True,
                        help="Use scaling for OLion optimizer")

    # FSDP
    parser.add_argument("--use-fsdp", action="store_true", help="Use Fully Sharded Data Parallel (FSDP2)")
    parser.add_argument("--fsdp-reshard-after-forward", action=argparse.BooleanOptionalAction, default=True,
                        help="FSDP2: Reshard parameters after forward pass (True=FULL_SHARD, False=SHARD_GRAD_OP)")
    parser.add_argument("--fsdp-sharding-strategy", type=str, default="FULL_SHARD", 
                        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"],
                        help="[Deprecated for FSDP2] FSDP1 sharding strategy (used as fallback if --fsdp-reshard-after-forward not set)")
    parser.add_argument("--fsdp-backward-prefetch", type=str, default="BACKWARD_PRE", 
                        choices=["BACKWARD_PRE", "BACKWARD_POST", "NONE"],
                        help="[Deprecated for FSDP2] FSDP1 backward prefetch mode (FSDP2 always uses BACKWARD_PRE)")
    parser.add_argument("--fsdp-offload-to-cpu", action="store_true", 
                        help="Offload FSDP state dict to CPU")
    parser.add_argument("--fsdp-state-dict-rank0-only", action="store_true",
                        help="Only rank 0 saves state dict in FSDP")

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0.1, help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)