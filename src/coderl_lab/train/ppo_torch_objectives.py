"""Torch-autograd PPO/GAE tensor objectives (TRAIN-010A).

No rollout collector or value-head architecture here yet. This module makes
the NumPy TRAIN-010A CPU reference differentiable and testable on actual CUDA.
A successful synthetic parameter update is NOT yet full model RLHF/PPO.
"""
from __future__ import annotations

from typing import Any


def _require_torch_tensor(name: str, x: Any):
    import torch
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a Torch Tensor")
    if x.ndim!=2 or not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError(f"{name} must be finite [B,T] float Tensor")
    return x


def _binary_mask(name: str, x, ref):
    import torch
    if not isinstance(x, torch.Tensor) or x.shape!=ref.shape:
        raise ValueError(f"{name} must match sampled completion token shape")
    if not torch.all((x==0)|(x==1)):
        raise ValueError(f"{name} must be binary")
    return x.bool()


def _masked_mean(loss, mask):
    import torch
    n=int(mask.sum().item())
    if n<1:
        raise ValueError("no valid completion tokens; not an RLHF optimizer update")
    return torch.masked_select(loss,mask).mean()


def ppo_actor_loss_torch(*,new_logprobs,old_logprobs,advantages,valid_mask,
                         clip_epsilon:float=0.2):
    import torch
    new=_require_torch_tensor("new_logprobs",new_logprobs)
    old=_require_torch_tensor("old_logprobs",old_logprobs)
    adv=_require_torch_tensor("advantages",advantages)
    if new.shape!=old.shape or old.shape!=adv.shape:
        raise ValueError("actor ratio/advantages tensors have incompatible shape")
    mask=_binary_mask("valid_mask",valid_mask,new)
    if not 0<clip_epsilon<1:
        raise ValueError("PPO ratio clip must be (0,1)")
    delta=new-old
    if torch.any(delta[mask].abs()>20):
        raise ValueError("PPO ratio logprob difference exceeds stable budget")
    ratio=torch.exp(delta)
    clipped=ratio.clamp(1-clip_epsilon,1+clip_epsilon)
    conservative=torch.minimum(ratio*adv,clipped*adv)
    loss=-_masked_mean(conservative,mask)
    stats={
        "clip_fraction":float(_masked_mean((ratio!=clipped).float(),mask).detach()),
        "mean_ratio":float(_masked_mean(ratio,mask).detach()),
        "sampled_approx_kl":float(_masked_mean((ratio-1)-delta,mask).detach()),
        "nonzero_advantage_fraction":float(_masked_mean((adv!=0).float(),mask).detach()),
    }
    return loss,stats


def ppo_critic_loss_torch(*,new_values,old_values,returns,valid_mask,
                          clip_epsilon:float=0.2):
    import torch
    new=_require_torch_tensor("new_values",new_values)
    old=_require_torch_tensor("old_values",old_values)
    target=_require_torch_tensor("returns",returns)
    if new.shape!=old.shape or old.shape!=target.shape:
        raise ValueError("critic value/return tensors have incompatible shape")
    mask=_binary_mask("valid_mask",valid_mask,new)
    if not 0<clip_epsilon<1:
        raise ValueError("PPO value clip must be (0,1)")
    clipped=old+(new-old).clamp(-clip_epsilon,clip_epsilon)
    conservative=torch.maximum((new-target).square(),(clipped-target).square())
    loss=0.5*_masked_mean(conservative,mask)
    stats={"value_clip_fraction":float(_masked_mean(
        ((new-old).abs()>clip_epsilon).float(),mask).detach())}
    return loss,stats


def frozen_reference_kl_torch(*,new_logprobs,reference_logprobs,valid_mask):
    import torch
    new=_require_torch_tensor("new_logprobs",new_logprobs)
    ref=_require_torch_tensor("reference_logprobs",reference_logprobs)
    if new.shape!=ref.shape:
        raise ValueError("frozen reference logprobs must match policy")
    mask=_binary_mask("valid_mask",valid_mask,new)
    logratio=ref-new
    if torch.any(logratio[mask].abs()>20):
        raise ValueError("KL logprob difference exceeds stable budget")
    per_token=logratio.exp()-logratio-1
    return _masked_mean(per_token,mask)


def torch_gae(*,rewards,values,valid_mask,terminal_mask,
              gamma:float=0.99,gae_lambda:float=0.95):
    import torch
    r=_require_torch_tensor("rewards",rewards)
    val=_require_torch_tensor("values",values)
    if val.shape!=(r.shape[0],r.shape[1]+1):
        raise ValueError("values require next-state bootstrap [B,T+1]")
    valid=_binary_mask("valid_mask",valid_mask,r)
    term=_binary_mask("terminal_mask",terminal_mask,r)
    if not 0<=gamma<=1 or not 0<=gae_lambda<=1:
        raise ValueError("invalid GAE gamma/lambda")
    if (term&~valid).any():
        raise ValueError("terminal marker on padding")
    if (~valid).all(dim=1).any():
        raise ValueError("empty sampled completion is not PPO data")
    for row in range(valid.shape[0]):
        idx=torch.nonzero(valid[row],as_tuple=False).flatten()
        if not torch.equal(idx,torch.arange(len(idx),device=idx.device)):
            raise ValueError("PPO sampled completion tokens must form contiguous prefix")
        eos=torch.nonzero(term[row],as_tuple=False).flatten()
        if eos.numel()>1 or (eos.numel()==1 and eos[0]!=idx[-1]):
            raise ValueError("natural terminal EOS must mark final valid token")
    with torch.no_grad():
        # Returns and advantages are frozen rollout targets: never backprop
        # through old value estimates to update policy or the value baseline.
        target=val.detach()
        out=torch.zeros_like(r)
        carry=torch.zeros(r.shape[0],dtype=r.dtype,device=r.device)
        for pos in range(r.shape[1]-1,-1,-1):
            active=valid[:,pos].to(r.dtype)
            continuation=(~term[:,pos]).to(r.dtype)
            delta=r[:,pos]+gamma*target[:,pos+1]*continuation-target[:,pos]
            carry=(delta+gamma*gae_lambda*continuation*carry)*active
            out[:,pos]=carry
        ret=(out+target[:,:-1])*valid.to(r.dtype)
    return out.detach(),ret.detach()


def differentiable_ppo_loss(*,new_logprobs,old_logprobs,advantages,
    new_values,old_values,returns,reference_logprobs,valid_mask,
    entropy_mean,policy_clip:float=0.2,value_clip:float=0.2,
    value_coef:float=0.5,kl_coef:float=0.02,entropy_coef:float=0.01):
    import torch
    policy,stats=ppo_actor_loss_torch(new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,advantages=advantages,valid_mask=valid_mask,
        clip_epsilon=policy_clip)
    critic,critic_stats=ppo_critic_loss_torch(new_values=new_values,
        old_values=old_values,returns=returns,valid_mask=valid_mask,
        clip_epsilon=value_clip)
    kl=frozen_reference_kl_torch(new_logprobs=new_logprobs,
        reference_logprobs=reference_logprobs,valid_mask=valid_mask)
    if not isinstance(entropy_mean,torch.Tensor) or entropy_mean.numel()!=1:
        raise ValueError("entropy must be a scalar tensor")
    if not all(0<=f for f in (value_coef,kl_coef,entropy_coef)):
        raise ValueError("PPO loss coefficient must be non-negative")
    total=policy+value_coef*critic+kl_coef*kl-entropy_coef*entropy_mean
    if not bool(torch.isfinite(total).item()):
        raise FloatingPointError("nonfinite differentiable PPO total loss")
    return total,{
        "policy_loss":float(policy.detach()),
        "critic_loss":float(critic.detach()),
        "reference_kl":float(kl.detach()),
        "entropy":float(entropy_mean.detach()),
        **stats,**critic_stats,
    }
