from pathlib import Path
from types import SimpleNamespace
import copy
import json
import unittest
import tempfile
from unittest.mock import patch
import yaml
from coderl_lab.train import exp030_budget_grpo as budget
from coderl_lab.train import exp027_matched_reward as matched
from coderl_lab.train import exp028_durable_reward as durable
from coderl_lab.train import exp028_cpu_guard as guard


def config():
    return yaml.safe_load((Path(__file__).parents[1]/'configs/exp030_budget.yaml').read_text())


class BudgetTests(unittest.TestCase):
 def test_fixed_recipe_and_registered_seeds(self):
    for seed in (51,52,53):budget.validate_config(config(),seed)
    for seed in (42,44,True):
        with self.assertRaises(ValueError):budget.validate_config(config(),seed)
    for section,key,value in [('budget','max_steps',129),('budget','checkpoint_steps',[32,128]),
        ('grpo','learning_rate',2e-5),('grpo','num_generations',8),('lora','r',16)]:
        cfg=config();cfg[section][key]=value
        with self.assertRaises(ValueError):budget.validate_config(cfg,51)


 def test_reward_budget_allows_full_128_groups(self):
    tmp_path=Path(tempfile.mkdtemp())
    calls=[]
    context=patch.object(matched,'reward_factory',lambda *a:lambda *a,**k:calls.append(1) or [1.]*4)
    context.start();self.addCleanup(context.stop)
    reward=budget.budget_reward(tmp_path/'rewards',tmp_path/'cache')
    for _ in range(128):assert reward(['p']*4,['c']*4)==[1.]*4
    with self.assertRaises(RuntimeError):reward(['p']*4,['c']*4)
    assert len(calls)==128


 def test_v3_context_and_partial_group_evidence(self):
    tmp_path=Path(tempfile.mkdtemp())
    old=matched.guarded
    def fail(*a,**k):
        assert (tmp_path/'pending_groups/group_0000.json').is_file()
        raise RuntimeError('indeterminate execution')
    context=patch.object(durable,'_BASE_FACTORY',lambda *a:fail)
    context.start();self.addCleanup(context.stop)
    with budget.v3.activated():
        assert matched.guarded is guard
        assert matched.protocol()['cpu_seconds']==15
        reward=budget.budget_reward(tmp_path/'rewards',tmp_path/'cache')
        with self.assertRaises(RuntimeError):reward(['p']*4,['c']*4,task_id=['t']*4,test_cases=[[]]*4)
        assert not (tmp_path/'reward_groups_ready').exists()
    assert matched.guarded is old


 def test_output_and_cache_refuse_overwrite_or_overlap(self):
    tmp_path=Path(tempfile.mkdtemp())
    assets=tmp_path/'assets';assets.mkdir()
    args=SimpleNamespace(output_dir=tmp_path/'run',data_dir=assets/'narrow',
        sft_adapter=tmp_path/'sft',config=tmp_path/'config.yaml',gate_report=tmp_path/'gate.json',cache_root=tmp_path/'cache')
    budget.validate_paths(args)
    args.output_dir.mkdir()
    with self.assertRaises(FileExistsError):budget.validate_paths(args)
    args.output_dir=assets/'evil'
    with self.assertRaises(ValueError):budget.validate_paths(args)
    args.output_dir=tmp_path/'fresh';args.cache_root=assets/'cache'
    with self.assertRaises(ValueError):budget.validate_paths(args)


if __name__=="__main__":unittest.main()
