"""TRAIN-010A: audited CPU-numerical PPO/GAE reference objectives.

This is NOT a PPO training loop. It supplies deterministic mathematical
oracles and contract checks before a future Torch actor/value rollout runner
is allowed to use the trained TRAIN-008A scalar reward model.

Conventions:
  reward, old/new/ref_logprobs, values: [batch, sequence]
  value_bootstrap: [batch, sequence+1], including V(s_T)
  valid_mask: 1 for sampled completion tokens, 0 for padding
  terminal_mask: 1 at natural EOS and ONLY at the last valid token
  reward is a scalar assigned at the last valid token of a finished rollout
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _matrix(name: str, x: Any) -> np.ndarray:
    arr=np.asarray(x,dtype=np.float64)
    if arr.ndim!=2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite rank-two tensor")
    return arr


def _mask(name: str, x: Any, shape: tuple[int,int]) -> np.ndarray:
    arr=_matrix(name,x)
    if arr.shape!=shape or not np.all(np.logical_or(arr==0,arr==1)):
        raise ValueError(f"{name} must be a matching binary token mask")
    return arr.astype(np.bool_)


def _mean_over_tokens(tensor:np.ndarray, mask:np.ndarray) -> float:
    count=int(mask.sum())
    if count==0:
        raise ValueError("zero effective completion tokens; cannot claim PPO update")
    return float(tensor[mask].sum()/count)


def validate_termination(
    *,valid_mask:Any,terminal_mask:Any,
    require_any_completed:bool=True,
)->dict[str,Any]:
    valid=_matrix("valid_mask",valid_mask)
    if valid.shape[0]<1 or valid.shape[1]<1:
        raise ValueError("zero rollout or completion sequence length")
    token_mask=_mask("valid_mask",valid,valid.shape)
    terminated=_mask("terminal_mask",terminal_mask,valid.shape)
    if np.any(terminated & ~token_mask):
        raise ValueError("EOS terminal cannot occur on a padded token")
    lengths=token_mask.sum(axis=1)
    if np.any(lengths<1):
        raise ValueError("empty sampled completion")
    for i in range(len(lengths)):
        idx=np.flatnonzero(token_mask[i])
        if not np.array_equal(idx,np.arange(idx[-1]+1)):
            raise ValueError("valid_mask must be contiguous prefix tokens")
        stops=np.flatnonzero(terminated[i])
        if len(stops)>1 or (len(stops)==1 and stops[0]!=idx[-1]):
            raise ValueError("natural EOS must be unique and the last valid token")
    completed=terminated.any(axis=1)
    if require_any_completed and not completed.any():
        raise ValueError("all rollouts truncated: no terminal quality evidence")
    return {
        "rollouts":int(len(lengths)),
        "terminated_rollouts":int(completed.sum()),
        "truncated_rollouts":int((~completed).sum()),
        "terminated_fraction":float(completed.mean()),
        "all_truncated":bool(not completed.any()),
        "sampled_completion_tokens":int(token_mask.sum()),
        "mean_completion_length":float(lengths.mean()),
    }


def assign_terminal_reward(
    *,
    learned_reward:Any,
    valid_mask:Any,
    terminal_mask:Any,
    truncated_penalty:float=-1.0,
    require_any_completed:bool=True,
)->tuple[np.ndarray,dict[str,Any]]:
    """Never blindly pay a high RM score to an unfinished completion.

    For genuine EOS-terminated text the score goes to its last valid token.
    Truncated rollouts instead receive an explicit, fixed penalty. A formal
    protocol must not proceed if every sampled completion is truncated.
    """
    valid=_matrix("valid_mask",valid_mask)
    mask=_mask("valid_mask",valid,valid.shape)
    term=_mask("terminal_mask",terminal_mask,valid.shape)
    audit=validate_termination(valid_mask=valid,terminal_mask=term,
                               require_any_completed=require_any_completed)
    scores=np.asarray(learned_reward,dtype=np.float64)
    if scores.ndim!=1 or len(scores)!=mask.shape[0] or not np.isfinite(scores).all():
        raise ValueError("one finite learned score per sampled rollout required")
    if not np.isfinite(truncated_penalty):
        raise ValueError("truncated penalty must be finite")
    out=np.zeros_like(valid)
    for i in range(mask.shape[0]):
        last=int(np.flatnonzero(mask[i])[-1])
        out[i,last]=float(scores[i]) if term[i,last] else truncated_penalty
    audit["truncation_reward_policy"]="fixed_penalty_not_learned_rm"
    audit["truncated_penalty"]=float(truncated_penalty)
    return out,audit


def generalized_advantage_estimate(
    *,
    rewards:Any,
    values:Any,
    valid_mask:Any,
    terminal_mask:Any,
    gamma:float=0.99,
    gae_lambda:float=0.95,
)->tuple[np.ndarray,np.ndarray]:
    """Masked on-policy GAE with explicit natural terminal and bootstrap.

    values has one extra final state so truncated sequences can bootstrap.
    """
    r=_matrix("rewards",rewards)
    v=_matrix("values",values)
    if v.shape!=(r.shape[0],r.shape[1]+1):
        raise ValueError("values must include V(s_0)...V(s_T) bootstrap")
    mask=_mask("valid_mask",valid_mask,r.shape)
    term=_mask("terminal_mask",terminal_mask,r.shape)
    validate_termination(valid_mask=mask.astype(int),terminal_mask=term.astype(int),
                         require_any_completed=False)
    if not (0<=gamma<=1 and 0<=gae_lambda<=1):
        raise ValueError("gamma/lambda must lie in [0,1]")
    advantages=np.zeros_like(r)
    for row in range(r.shape[0]):
        gae=0.0
        for pos in range(r.shape[1]-1,-1,-1):
            if not mask[row,pos]:
                gae=0.0
                continue
            nonterminal=not bool(term[row,pos])
            delta=r[row,pos]+gamma*v[row,pos+1]*nonterminal-v[row,pos]
            gae=delta+gamma*gae_lambda*nonterminal*gae
            advantages[row,pos]=gae
    returns=advantages+v[:,:-1]
    returns[~mask]=0.0
    return advantages,returns


def ppo_clipped_policy_loss(
    *,new_logprobs:Any,old_logprobs:Any,advantages:Any,
    valid_mask:Any,clip_epsilon:float=0.2,
)->dict[str,float]:
    new=_matrix("new_logprobs",new_logprobs)
    old=_matrix("old_logprobs",old_logprobs)
    adv=_matrix("advantages",advantages)
    if new.shape!=old.shape or old.shape!=adv.shape:
        raise ValueError("PPO logprob/advantage shapes differ")
    mask=_mask("valid_mask",valid_mask,new.shape)
    if not (0<clip_epsilon<1):
        raise ValueError("PPO clip_epsilon must be in (0,1)")
    log_ratio=new-old
    if np.any(np.abs(log_ratio[mask])>20):
        raise ValueError("PPO importance ratio too extreme; refuse overflow")
    ratio=np.exp(log_ratio)
    clipped=np.clip(ratio,1-clip_epsilon,1+clip_epsilon)
    conservative=np.minimum(ratio*adv,clipped*adv)
    loss=-_mean_over_tokens(conservative,mask)
    return {
        "policy_loss":loss,
        "clip_fraction":_mean_over_tokens((ratio!=clipped).astype(float),mask),
        "mean_ratio":_mean_over_tokens(ratio,mask),
        "sampled_approx_kl":_mean_over_tokens(
            (ratio-1)-log_ratio,mask),
        "nonzero_advantage_fraction":_mean_over_tokens((adv!=0).astype(float),mask),
    }


def ppo_clipped_value_loss(
    *,new_values:Any,old_values:Any,returns:Any,
    valid_mask:Any,clip_epsilon:float=0.2,
)->dict[str,float]:
    new=_matrix("new_values",new_values)
    old=_matrix("old_values",old_values)
    target=_matrix("returns",returns)
    if new.shape!=old.shape or old.shape!=target.shape:
        raise ValueError("PPO value tensors must match")
    mask=_mask("valid_mask",valid_mask,new.shape)
    if not (0<clip_epsilon<1):
        raise ValueError("PPO value clip_epsilon must be (0,1)")
    clipped=old+np.clip(new-old,-clip_epsilon,clip_epsilon)
    loss=np.maximum((new-target)**2,(clipped-target)**2)
    return {"value_loss":0.5*_mean_over_tokens(loss,mask),
            "value_clip_fraction":_mean_over_tokens(
                (np.abs(new-old)>clip_epsilon).astype(float),mask)}


def frozen_reference_kl(
    *,new_logprobs:Any,ref_logprobs:Any,valid_mask:Any,
)->float:
    new=_matrix("new_logprobs",new_logprobs)
    ref=_matrix("ref_logprobs",ref_logprobs)
    if new.shape!=ref.shape:
        raise ValueError("reference-policy token logprob shapes differ")
    mask=_mask("valid_mask",valid_mask,new.shape)
    logratio=ref-new
    if np.any(np.abs(logratio[mask])>20):
        raise ValueError("frozen reference KL ratio overflow")
    per_token=np.exp(logratio)-logratio-1
    return max(0.0,_mean_over_tokens(per_token,mask))


def ppo_combined_objective(
    *,policy:dict[str,float],value:dict[str,float],
    frozen_ref_kl:float,entropy_mean:float,
    value_coefficient:float=0.5,
    kl_coefficient:float=0.02,
    entropy_coefficient:float=0.01,
)->dict[str,float]:
    vals=(policy["policy_loss"],value["value_loss"],frozen_ref_kl,
          entropy_mean,value_coefficient,kl_coefficient,entropy_coefficient)
    if not all(np.isfinite(v) for v in vals):
        raise ValueError("nonfinite PPO objective inputs")
    if min(value_coefficient,kl_coefficient,entropy_coefficient)<0 or frozen_ref_kl<0:
        raise ValueError("PPO penalties must be non-negative")
    total=(policy["policy_loss"]+value_coefficient*value["value_loss"]
           +kl_coefficient*frozen_ref_kl-entropy_coefficient*entropy_mean)
    return {"total_loss":float(total),
            "policy_loss":float(policy["policy_loss"]),
            "value_loss":float(value["value_loss"]),
            "frozen_reference_kl":float(frozen_ref_kl),
            "entropy_mean":float(entropy_mean)}
