"""TRAIN-010A differentiable PPO CPU/CUDA unit contracts.

Torch is intentionally optional in default GitHub CPU-only CI. Real Tensor
and gradient verification is executed against the pinned Tang CUDA environment.
"""
from __future__ import annotations

import numpy as np
import pytest

torch=pytest.importorskip("torch")

from coderl_lab.train.ppo_objectives import (
    generalized_advantage_estimate,ppo_clipped_policy_loss,
    ppo_clipped_value_loss,frozen_reference_kl,
)
from coderl_lab.train.ppo_torch_objectives import (
    torch_gae,ppo_actor_loss_torch,ppo_critic_loss_torch,
    frozen_reference_kl_torch,differentiable_ppo_loss,
)


@pytest.mark.parametrize("device",["cpu"]+(["cuda"] if torch.cuda.is_available() else []))
def test_torch_gae_matches_frozen_numpy_terminal_and_bootstrap(device):
    r=np.array([[0.1,0.4,0],[0.2,-0.8,0.3]],dtype=np.float64)
    v=np.array([[0.1,0.3,0.4,99],[0.4,-0.1,0.05,0.7]],dtype=np.float64)
    valid=np.array([[1,1,0],[1,1,1]],dtype=np.float64)
    terminal=np.array([[0,1,0],[0,0,0]],dtype=np.float64)
    np_adv,np_ret=generalized_advantage_estimate(
        rewards=r,values=v,valid_mask=valid,terminal_mask=terminal,
        gamma=.95,gae_lambda=.91)
    values=torch.tensor(v,device=device,dtype=torch.float64,requires_grad=True)
    tx,ty=torch_gae(
        rewards=torch.tensor(r,device=device,dtype=torch.float64),
        values=values,
        valid_mask=torch.tensor(valid,device=device),
        terminal_mask=torch.tensor(terminal,device=device),
        gamma=.95,gae_lambda=.91)
    np.testing.assert_allclose(tx.detach().cpu().numpy(),np_adv,rtol=0,atol=1e-12)
    np.testing.assert_allclose(ty.detach().cpu().numpy(),np_ret,rtol=0,atol=1e-12)
    assert tx.requires_grad is False and ty.requires_grad is False


@pytest.mark.parametrize("device",["cpu"]+(["cuda"] if torch.cuda.is_available() else []))
def test_torch_policy_value_and_reference_kl_equal_numpy_oracle(device):
    t=lambda a:torch.tensor(a,dtype=torch.float64,device=device)
    old=np.array([[0,-.2],[0,0.]])
    new=np.array([[.1,-.25],[.1,.07]])
    adv=np.array([[1,-1],[.3,.7]])
    vals=np.array([[.25,.1],[.12,.2]])
    orig=np.array([[.2,.15],[.25,.14]])
    ret=np.array([[.4,.2],[.1,.15]])
    ref=np.array([[-.05,-.2],[0,0.]])
    mask=np.ones_like(old)
    ref_actor=ppo_clipped_policy_loss(
        new_logprobs=new,old_logprobs=old,advantages=adv,valid_mask=mask)
    ref_value=ppo_clipped_value_loss(
        new_values=vals,old_values=orig,returns=ret,valid_mask=mask)
    ref_kl=frozen_reference_kl(new_logprobs=new,ref_logprobs=ref,valid_mask=mask)
    value,_=ppo_actor_loss_torch(new_logprobs=t(new),old_logprobs=t(old),
                                advantages=t(adv),valid_mask=t(mask))
    critic,_=ppo_critic_loss_torch(new_values=t(vals),old_values=t(orig),
                                  returns=t(ret),valid_mask=t(mask))
    kl=frozen_reference_kl_torch(new_logprobs=t(new),reference_logprobs=t(ref),
                                  valid_mask=t(mask))
    assert float(value)==pytest.approx(ref_actor["policy_loss"],abs=1e-12)
    assert float(critic)==pytest.approx(ref_value["value_loss"],abs=1e-12)
    assert float(kl)==pytest.approx(ref_kl,abs=1e-12)


@pytest.mark.parametrize("device",["cpu"]+(["cuda"] if torch.cuda.is_available() else []))
def test_real_torch_ppo_actor_critic_optimizer_updates_nonzero_weights(device):
    actor=torch.nn.Parameter(torch.tensor([[-.07,.12,.24]],device=device))
    value=torch.nn.Parameter(torch.tensor([[.3,.18,.12]],device=device))
    before=[actor.detach().clone(),value.detach().clone()]
    optimizer=torch.optim.AdamW([actor,value],lr=.05)
    prior=torch.zeros_like(actor)
    reference=torch.zeros_like(actor)
    returns=torch.tensor([[.8,.3,-.1]],device=device)
    advantage=torch.tensor([[.7,.15,-.4]],device=device)
    mask=torch.ones_like(actor)
    loss,metrics=differentiable_ppo_loss(
        new_logprobs=actor,old_logprobs=prior,advantages=advantage,
        new_values=value,old_values=value.detach().clone(),returns=returns,
        reference_logprobs=reference,valid_mask=mask,
        entropy_mean=torch.tensor(0.,device=device),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert actor.grad is not None and torch.isfinite(actor.grad).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert actor.grad.norm().item()>0 and value.grad.norm().item()>0
    optimizer.step()
    assert not torch.equal(actor,before[0])
    assert not torch.equal(value,before[1])
    assert metrics["reference_kl"]>=0
    assert metrics["nonzero_advantage_fraction"]==pytest.approx(1)


@pytest.mark.parametrize("device",["cpu"]+(["cuda"] if torch.cuda.is_available() else []))
def test_mask_gates_and_nonterminal_bootstrap_fail_closed(device):
    r=torch.tensor([[0.,1.,0.]],device=device)
    v=torch.tensor([[0.,0.,0.,0.]],device=device)
    with pytest.raises(ValueError,match="contiguous"):
        torch_gae(rewards=r,values=v,
                  valid_mask=torch.tensor([[1,0,1]],device=device),
                  terminal_mask=torch.tensor([[0,0,1]],device=device))
    with pytest.raises(ValueError,match="final valid"):
        torch_gae(rewards=r,values=v,
                  valid_mask=torch.tensor([[1,1,1]],device=device),
                  terminal_mask=torch.tensor([[1,0,0]],device=device))
    with pytest.raises(ValueError,match="no valid"):
        ppo_actor_loss_torch(new_logprobs=r,old_logprobs=r,
                             advantages=v[:,:-1],
                             valid_mask=torch.zeros_like(r))
    with pytest.raises(ValueError,match="stable budget"):
        ppo_actor_loss_torch(new_logprobs=torch.full_like(r,80),
                             old_logprobs=r,advantages=r,
                             valid_mask=torch.ones_like(r))
