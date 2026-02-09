# ruff: noqa
# type: ignore
# fmt: off

# credits to https://gist.github.com/main-horse/7314170780e36f7443d1926418d75823

import math
from typing import Protocol
import torch
from torch.distributed.tensor import DTensor
from torch.distributed import  gather, scatter
from collections import deque

__version__ = "0.3.0"

__all__ = ["OLion"]


@torch.compile(fullgraph=True)
def nsloop_torch(X: torch.Tensor, steps: int, *, a=3.4445, b=-4.7750, c=2.0315):
    """
    When compiled down, inductor produces the following steps:
    1. A = matmul X with reinterpret_tensor(X)
    2. (triton) read A -> write b*A and c*A
    3. B = addmm(b*A, c*A, A)
    4. (triton) read X -> write a*X (this is stupid)
    5. X = addmm(a*X, B, X)
    """
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X

def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    X = nsloop_torch(X, steps, a=a, b=b, c=c)
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def apply_momentum(grad, momentum_buffer, momentum, momentum_2, nesterov):
    
    momentum_buffer.lerp_(grad, 1 - momentum_2)
    update = grad.lerp_(momentum_buffer, momentum) if nesterov else momentum_buffer
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    return update

def apply_scaling(grad, rms_scale=False ):
    if rms_scale:
        # https://github.com/MoonshotAI/Moonlight/blob/5afcb6911077e7f182d05865fe90d9f39abcbcbd/examples/toy_train.py#L146
        grad *= 0.2 * math.sqrt(max(grad.shape[1], grad.shape[0]))
        return grad
    else:
        # https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L40
        grad *= max(1, grad.size(-2) / grad.size(-1))**0.5
        return grad

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)




class Work(Protocol):
    
    def __init__(self, param, state, group, index: int):
        ...
    
    def start(self):
        ...
    
    def finish(self):
        ...
    

class Fsdp1dWork:
    """
    OLion handle for fsdp2 1d mesh.
    """
    
    def __init__(self, param, state, group, index: int):
        self.param = param
        self.state = state
        self.group = group
        
        self.index = index
    
        self._intermediate_state = None
    
    def start(self):
        # Apply momentum to gradient (OLion style)
        self.param.grad = apply_momentum(self.param.grad, self.state["momentum_buffer"], self.group["momentum"], self.group["momentum_2"], self.group["nesterov"])
        
        grad = self.param.grad
        assert isinstance(grad, DTensor), "only supports DTensor parameters"
        assert grad.device_mesh.ndim == 1, "only supports 1D mesh"
        
        rank = grad.device_mesh.get_rank()
        world_size = grad.device_mesh.size()
        pg = grad.device_mesh.get_group()
        
        dest_rank = self.index % world_size
        
        # Gather gradient
        if rank == dest_rank:
            gather_lists = [torch.zeros_like(input=grad.to_local()) for _ in range(world_size)]
            gather_handle = gather(grad.to_local(), gather_lists, dst=dest_rank, group=pg, async_op=True)
        else:
            gather_lists = None
            gather_handle = gather(grad.to_local(), None, dst=dest_rank, group=pg, async_op=True)
        
       
            
        self._intermediate_state = [dest_rank, gather_handle, gather_lists]

    def finish(self):
        
        assert self._intermediate_state is not None, "gather work must be called first"
        
        grad = self.param.grad
        rank = grad.device_mesh.get_rank()
        world_size = grad.device_mesh.size()
        pg = grad.device_mesh.get_group()
        
        dest_rank, gather_handle, gather_lists = self._intermediate_state
        
        # Wait for both gather operations to complete
        gather_handle.wait()
        
        if rank == dest_rank:
            # Concatenate gradient from all processes
            g_full_block = torch.cat(gather_lists, dim=0)
            
            # Apply OLion update logic to the full block
            g_full_block = self._apply_olion_update_to_block(g_full_block)
            g_full_block = g_full_block.type_as(grad)
            chunks = list(g_full_block.chunk(chunks=world_size, dim=0))
            scatter(grad.to_local(), scatter_list=chunks, src=dest_rank, group=pg, async_op=False)
        else:
            scatter(grad.to_local(), None, src=dest_rank, group=pg, async_op=False)
        
        update = apply_scaling(grad, self.group["rms_scale"])

        self.param.mul_(1 - self.group["lr"] * self.group["weight_decay"])
        self.param.add_(update.reshape(self.param.shape), alpha=-self.group["lr"])
        #

    def _apply_olion_update_to_block(self, g_full_block):
        """
        Apply OLion update logic to the full gradient block.
        This is the core OLion logic adapted for the distributed case.
        """
        # Initialize step counter if needed
        if "step" not in self.state:
            self.state["step"] = 0
        
        self.state["step"] += 1
        
        # Apply OLion logic - g_full_block already has momentum applied from start()
        if g_full_block.ndim == 4:
            g_full_block = g_full_block.view(len(g_full_block), -1)
           
        
     
        g_full_block = zeropower_via_newtonschulz5(g_full_block, steps=self.group["ns_steps"])
      
       
        
        eps = self.group["eps"]
       
        
        g_full_block = torch.sign(g_full_block)
        
        # RMS-aligned Rescaling
        if self.group["use_scale"]:
            scale = min(self.param.shape)**0.5 / (g_full_block.norm() + eps)
            g_full_block.mul_(scale)
        
        return g_full_block


class TpFsdp2dWork:
    """
    OLion work for TP + FSDP mesh
    """
    
    def __init__(self, param, state, group, index: int):
        raise NotImplementedError("not implemented")
    
class EpFsdp2dWork:
    """
    OLion work for EP mesh
    """
    
    def __init__(self, param, state, group, index: int):
        raise NotImplementedError("not implemented")
    
class TpEpFsdp3dWork:
    """
    OLion work for TP + EP mesh
    """
    
    def __init__(self, param, state, group, index: int):
        raise NotImplementedError("not implemented")

class SingelDeviceWork:
    """
    OLion handle for single device.
    """
    
    def __init__(self, param, state, group, index: int):
        self.param = param
        self.state = state
        self.group = group
        
    def start(self):
        # Initialize momentum buffer and step if needed
        if "momentum_buffer" not in self.state:
            self.state["momentum_buffer"] = torch.zeros_like(self.param.grad)
            self.state["step"] = 0
        
        # Use olion_update function for single device
        update = olion_update(self.param.grad, self.state["momentum_buffer"], 
                            self.group["momentum"], self.group["momentum_2"], 
                            self.group["nesterov"], self.group["ns_steps"], 
                            self.group["rms_scale"], self.group["eps"], 
                            self.group["use_scale"], self.state["step"])
        
        self.param.mul_(1 - self.group["lr"] * self.group["weight_decay"])
        if self.group["use_scale"]:
            self.param.add_(update.reshape(self.param.shape), alpha=-self.group["lr"] * 0.2 * max(self.param.size(-2), self.param.size(-1))**0.5)
        else:
            self.param.add_(update.reshape(self.param.shape), alpha=-self.group["lr"])
        
    def finish(self):
        pass
    
    
class OLion(torch.optim.Optimizer):
    """
    DTensor variant of OLion, original code from olion.py
    also support single device variant.
    
    Notable changes:
        - add rms_scale argument to the optimizer following the moonlight paper https://arxiv.org/abs/2502.16982
    
    example usage:
    
    ```python
    
    from olion_fsdp import OLion


    optimizer = OLion([
        dict(
            params=model.square_params(),
            lr=1e-3,
            use_olion=True
        ),
        dict(
            params=model.non_square_params(),
            lr=1e-3,
            use_olion=False
        )
    ])   
    ```
    
    
    param_groups args:
        lr: learning rate
        momentum: momentum
        weight_decay: weight decay
        use_olion: whether to use olion
        rms_scale: whether to scale the gradient by the RMS of the gradient . If true use the rms scale from the moonlight paper.
                https://github.com/MoonshotAI/Moonlight/blob/5afcb6911077e7f182d1d7faa3c2cd45acba4666/examples/toy_train.py#L146
                This variant adjust the update so that the RMS match the one of adam, allowing to only have one learning rate for all parameters.

    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_olion" in group
            if group["use_olion"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["momentum_2"] = group.get("momentum_2", 0.98)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["rms_scale"] = group.get("rms_scale", True)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["eps"] = group.get("eps", 1e-8)
                group["use_scale"] = group.get("use_scale", True)
                assert set(group.keys()) == set(["params", "lr", "momentum", "momentum_2", "weight_decay", "use_olion", "rms_scale", "nesterov", "ns_steps", "eps", "use_scale"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_olion"])
        super().__init__(param_groups, dict())

    def _get_work_class(self, p: torch.Tensor) -> tuple[type[Work], int]:
        """
        dispatch the work class based on the mesh dimension.
        """
        if isinstance(p, DTensor):
            if p.device_mesh.ndim == 1:
                return Fsdp1dWork, 8
            elif p.device_mesh.ndim == 2:
                return TpFsdp2dWork, 8
            else:
                raise ValueError(f"Unsupported mesh dimension: {p.device_mesh.ndim}")
        else:
            return SingelDeviceWork, 1
        
    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        
        dq: deque[Work] = deque()

        for group in self.param_groups:
            
            if group["use_olion"]:
                for i ,p in enumerate(group["params"]):
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p.grad)
                    
                    class_work, prefetch_factor = self._get_work_class(p)
                        
                    work = class_work(p, state, group, i)
                    work.start()
                    dq.append(work)
                    
                    
                    if len(dq) > prefetch_factor:
                        dq.popleft().finish()
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        for work in dq:
            work.finish()
            
        return loss
    

    
