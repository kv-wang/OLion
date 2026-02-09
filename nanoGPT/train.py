

import os
import argparse

# Remove wandb proxy environment variables if they are set
os.environ.pop('WANDB_HTTP_PROXY', None)
os.environ.pop('WANDB_HTTPS_PROXY', None)

# Also explicitly ignore system-wide proxies (if necessary)
os.environ['no_proxy'] = '*'

import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

from Muon import Muon
from AdaMuon import AdaMuon
from OLion import OLion
from Lion import Lion

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 200
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())

prefix = '/home/'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 8 # used to simulate larger batch sizes
batch_size = 60 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model

dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

eps = 1e-8
# optimizer
sophia_config = True
opt = 'adamw'  # default optimizer
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 10000 # total number of training iterations
weight_decay = 1e-1

beta1 = 0.9
# beta2 will be set from command line arguments
#beta3 = 0.999
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 20000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
print("dtype = ", dtype)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train GPT model with different optimizers')
parser.add_argument('--opt', type=str, default='adamw', choices=['adamw', 'muon', 'adamuon', 'olion', 'lion'],
                    help='Optimizer to use: adamw, muon, adamuon, olion, lion')
parser.add_argument('--wandb_project', type=str, default='owt', 
                    help='Wandb project name')

parser.add_argument('--learning_rate', type=float, default=6e-4,
                    help='Learning rate')
parser.add_argument('--batch_size', type=int, default=12,
                    help='Batch size')
parser.add_argument('--max_iters', type=int, default=100000,
                    help='Maximum number of iterations')
parser.add_argument('--weight_decay', type=float, default=1e-1,
                    help='Weight decay')
parser.add_argument('--eval_interval', type=int, default=2000,
                    help='Evaluation interval')
parser.add_argument('--out_dir', type=str, default='out',
                    help='Output directory for checkpoints')
parser.add_argument('--compile', action='store_true', default=True,
                    help='Compile the model with torch.compile')
parser.add_argument('--no_compile', dest='compile', action='store_false',
                    help='Disable model compilation')
parser.add_argument('--n_layer', type=int, default=12,
                    help='Number of transformer layers')
parser.add_argument('--n_head', type=int, default=12,
                    help='Number of attention heads')
parser.add_argument('--n_embd', type=int, default=768,
                    help='Number of embedding dimensions')
parser.add_argument('--beta2', type=float, default=0.0,
                    help='Beta2 parameter for optimizers (default: 0.99)')
parser.add_argument('--use_scale', type=bool, default=True,
                    help='Use learning rate acaling')
parser.add_argument('--momentum', type=float, default=0.95,
                    help='Momentum parameter for optimizers (default: 0.95)')
parser.add_argument('--momentum_2', type=float, default=0.98,
                    help='Second momentum parameter for olion/lion (default: 0.98)')
parser.add_argument('--norm_stats_interval', type=int, default=0,
                    help='Interval for computing and logging matrix norms (spectral and L_infinity). 0 means disabled (default: 0)')
args, unknown = parser.parse_known_args()

# Override the optimizer choice and other parameters
opt = args.opt.lower()
wandb_project = args.wandb_project
wandb_run_name = args.opt + "_beta2_" + str(args.beta2)
if args.opt in ['olion', 'lion']:
    wandb_run_name = wandb_run_name + "_momentum_" + str(args.momentum) + "_momentum_2_" + str(args.momentum_2)

learning_rate = args.learning_rate
batch_size = args.batch_size
max_iters = args.max_iters
weight_decay = args.weight_decay
eval_interval = args.eval_interval
out_dir = args.out_dir
compile = args.compile
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd

beta2 = args.beta2
# Update betas tuple with the new beta2 value

betas = (beta1, beta2)
use_scale = args.use_scale
momentum = args.momentum
momentum_2 = args.momentum_2
norm_stats_interval = args.norm_stats_interval

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
#exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
print("the data type is ", dtype)
# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'best_ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)



# Parameter grouping for different optimizers
muon_params = []
adamw_params = []

for name, param in model.named_parameters():
    if any(key in name for key in [
                        "wte", "wpe", "embd", "embed",
                        "ln", "norm", "lm_head",
                        "output", "final_layer","bias"
                    ]):
        adamw_params.append(param)
    else:
        muon_params.append(param)

print(f"Muon parameters: {len(muon_params)}")
print(f"AdamW parameters: {len(adamw_params)}")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer configuration
print(f"Using optimizer: {opt}")
print(f"Wandb project: {wandb_project}")
print(f"Wandb run name: {wandb_run_name}")
print(f"Learning rate: {learning_rate}")
print(f"Batch size: {batch_size}")
print(f"Max iterations: {max_iters}")
print(f"Weight decay: {weight_decay}")
print(f"Evaluation interval: {eval_interval}")
print(f"Output directory: {out_dir}")
print(f"Model compilation: {compile}")
print(f"Number of layers: {n_layer}")
print(f"Number of heads: {n_head}")
print(f"Embedding dimensions: {n_embd}")
print(f"Beta2: {beta2}")
if opt in ['adamuon', 'muon', 'olion', 'lion']:
    print(f"Momentum: {momentum}")
if opt in ['olion', 'lion']:
    print(f"Use scale: {use_scale}")
if opt in ['olion', 'lion']:
    print(f"Momentum_2: {momentum_2}")
print(f"Norm stats interval: {norm_stats_interval} (0 means disabled)")
print("-" * 50)

beta2_adamw = 0.95

if opt == 'adamw':
    # Use AdamW for all parameters
    # Group parameters by weight decay
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, beta2_adamw), eps=eps)
    optimizers = [optimizer]
elif opt == 'muon':
    # Use Muon for muon_params and AdamW for adamw_params
    optimizers = [Muon(muon_params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay,
                      nesterov=True, ns_steps=5, rank=ddp_rank, world_size=ddp_world_size),
                  torch.optim.AdamW(adamw_params, lr=learning_rate, betas=(0.9, beta2_adamw), weight_decay=weight_decay, eps=eps)]
elif opt == 'adamuon':
    # Use AdaMuon for muon_params and AdamW for adamw_params
    optimizers = [AdaMuon(muon_params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay,
                         nesterov=True, ns_steps=5, eps=eps, rank=ddp_rank, world_size=ddp_world_size),
                  torch.optim.AdamW(adamw_params, lr=learning_rate, betas=(0.9, beta2_adamw), weight_decay=weight_decay, eps=eps)]
elif opt == 'olion':
    # Use OLion for muon_params and AdamW for adamw_params
    optimizers = [OLion(muon_params, lr=learning_rate, momentum=momentum, momentum_2=momentum_2, weight_decay=weight_decay, 
                       nesterov=True, ns_steps=5, eps=eps, rank=ddp_rank, world_size=ddp_world_size, use_scale=use_scale),
                  torch.optim.AdamW(adamw_params, lr=learning_rate, betas=(0.9, beta2_adamw), weight_decay=weight_decay, eps=eps)]
elif opt == 'lion':
    # Use Lion optimizer for muon_params and AdamW for adamw_params
    optimizers = [Lion(muon_params, lr=learning_rate, momentum=momentum, momentum_2=momentum_2, weight_decay=weight_decay, 
                       nesterov=True, ns_steps=5, eps=eps, rank=ddp_rank, world_size=ddp_world_size, use_scale=use_scale),
                  torch.optim.AdamW(adamw_params, lr=learning_rate, betas=(0.9, beta2_adamw), weight_decay=weight_decay, eps=eps)]
else:
    # Fallback to other optimizers from model.configure_optimizers
    optimizers = model.configure_optimizers(opt, sophia_config, weight_decay, learning_rate, betas, n_head, device_type, epsilon=eps)

if init_from == 'resume':
    if isinstance(optimizers, list):
        for optimizer, state_dict in zip(optimizers, checkpoint['optimizer']):
            optimizer.load_state_dict(state_dict)
    elif isinstance(optimizers, dict):
        for key, optimizer in optimizers.items():
            if key in checkpoint['optimizer']:
                optimizer.load_state_dict(checkpoint['optimizer'][key])
    else:
        optimizers.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# compute matrix norms (spectral norm and L_infinity norm) grouped by shape
@torch.no_grad()
def compute_matrix_norms(model):
    """
    计算模型中不同形状矩阵的spectral norm和L_infinity norm
    相同形状的矩阵取平均值
    返回字典: {shape_str: {'spectral_norm': avg_spectral, 'linf_norm': avg_linf}}
    """
    model.eval()
    shape_stats = {}  # {shape_str: {'spectral_norms': [], 'linf_norms': []}}
    
    # 获取原始模型（如果是DDP或编译后的模型）
    raw_model = model.module if hasattr(model, 'module') else model
    
    for name, param in raw_model.named_parameters():
        # 只处理2D或更高维度的矩阵参数
        if param.dim() >= 2:
            # 转换为字符串作为key
            shape_str = str(tuple(param.shape))
            
            # 初始化该形状的统计
            if shape_str not in shape_stats:
                shape_stats[shape_str] = {'spectral_norms': [], 'linf_norms': []}
            
            # 计算L_infinity norm（最大元素绝对值）
            linf_norm = param.abs().max().item()
            shape_stats[shape_str]['linf_norms'].append(linf_norm)
            
            # 计算spectral norm（最大奇异值）
            # 对于大矩阵，使用power iteration方法会更高效
            try:
                # 将参数reshape为2D矩阵（如果是高维的）
                if param.dim() > 2:
                    # 将除了最后两维之外的所有维度flatten
                    param_2d = param.view(-1, param.shape[-1])
                else:
                    param_2d = param
                
                # 使用SVD计算最大奇异值
                # 对于大矩阵，使用full_matrices=False可以节省内存
                # 只计算最大的几个奇异值
                m, n = param_2d.shape
                if m * n > 10000:
                    # 对于大矩阵，使用power iteration近似计算最大奇异值
                    # 或者使用torch.linalg.norm计算2-范数（对于矩阵就是spectral norm）
                    # 但torch.linalg.norm默认计算Frobenius norm，所以使用SVD但只计算top-1
                    try:
                        # 尝试使用更高效的方法：只计算top-1奇异值
                        # 如果PyTorch版本支持，使用torch.linalg.svdvals只计算奇异值
                        s = torch.linalg.svdvals(param_2d)
                        spectral_norm = s[0].item()
                    except:
                        # 回退到完整SVD
                        u, s, v = torch.linalg.svd(param_2d, full_matrices=False)
                        spectral_norm = s[0].item()
                else:
                    # 对于小矩阵，直接使用SVD
                    u, s, v = torch.linalg.svd(param_2d, full_matrices=False)
                    spectral_norm = s[0].item()
                
                shape_stats[shape_str]['spectral_norms'].append(spectral_norm)
            except Exception as e:
                # 如果计算失败，跳过这个参数
                print(f"Warning: Failed to compute spectral norm for {name}: {e}")
                continue
    
    # 计算平均值
    result = {}
    for shape_str, stats in shape_stats.items():
        if stats['spectral_norms'] and stats['linf_norms']:
            result[shape_str] = {
                'spectral_norm': sum(stats['spectral_norms']) / len(stats['spectral_norms']),
                'linf_norm': sum(stats['linf_norms']) / len(stats['linf_norms']),
                'count': len(stats['spectral_norms'])  # 记录该形状的矩阵数量
            }
    
    model.train()
    return result

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    api_key = os.environ.get('WANDB_API_KEY')
    wandb.login(key=api_key)
    import uuid
    run_id = wandb_run_name + "_" + str(uuid.uuid4())[:8]  # 确保run_id唯一
    
    wandb.init(project=wandb_project, name=wandb_run_name, id=run_id, config=config, reinit=True)
# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Log to wandb
        wandb_log_dict = {
            "iter": iter_num,
            "train/loss": losses['train'],
            "val/loss": losses['val'],
            "lr": lr,
            "mfu": running_mfu*100, # convert to percentage
        }

        if wandb_log:
            wandb.log(wandb_log_dict)
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': [optimizer.state_dict() for optimizer in optimizers],
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'best_ckpt.pt'))
        if  always_save_checkpoint and (iter_num % 2000 == 0):
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': [optimizer.state_dict() for optimizer in optimizers],
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, f"iter_{iter_num}_ckpt.pt"))
    
    # compute and log matrix norms at specified intervals
    if norm_stats_interval > 0 and iter_num % norm_stats_interval == 0 and master_process:
        print(f"Computing matrix norms at iteration {iter_num}...")
        norm_stats = compute_matrix_norms(model)
        
        # Log to wandb
        if wandb_log and norm_stats:
            norm_log_dict = {"iter": iter_num}
            for shape_str, stats in norm_stats.items():
                # 使用安全的key名称（wandb不支持某些特殊字符）
                # 将形状字符串转换为wandb友好的格式
                shape_key = shape_str.replace(' ', '').replace('(', '').replace(')', '').replace(',', '_')
                norm_log_dict[f"norms/spectral_norm_{shape_key}"] = stats['spectral_norm']
                norm_log_dict[f"norms/linf_norm_{shape_key}"] = stats['linf_norm']
                norm_log_dict[f"norms/count_{shape_key}"] = stats['count']
            
            wandb.log(norm_log_dict)
            print(f"Logged matrix norms for {len(norm_stats)} different shapes")
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        for optimizer in optimizers:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    for optimizer in optimizers:
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination condition
    if iter_num > max_iters:
        break

if ddp:
    try:
        destroy_process_group()
    except Exception as e:
        print(f"Warning: Error during process group destruction: {e}") 