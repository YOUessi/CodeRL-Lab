"""CPU-only contract tests; no downloaded weights or CUDA are needed."""
import importlib.util
import math
import unittest

from coderl_lab.train import ppo_math


class EpisodeContractTests(unittest.TestCase):
    def test_raw_policy_sampling_rejects_warped_distribution(self):
        self.assertTrue(hasattr(ppo_math, "validate_sampling_contract"))
        ppo_math.validate_sampling_contract(temperature=1.0, top_p=1.0)
        for temperature, top_p in [(0.8, 0.95), (0.8, 1), (1, 0.95), (math.nan, 1)]:
            with self.subTest(temperature=temperature, top_p=top_p):
                with self.assertRaisesRegex(ValueError, "raw-softmax"):
                    ppo_math.validate_sampling_contract(temperature=temperature, top_p=top_p)

    def test_eos_is_an_action_and_post_eos_padding_is_excluded(self):
        self.assertTrue(hasattr(ppo_math, "episode_response"))
        ids, reason = ppo_math.episode_response([4, 2, 2, 2], eos_token_id=2, max_new_tokens=4)
        self.assertEqual((ids, reason), ([4, 2], "eos"))
        ids, reason = ppo_math.episode_response([2, 2], eos_token_id=2, max_new_tokens=2)
        self.assertEqual((ids, reason), ([2], "eos"))

    def test_length_limit_is_terminal_and_never_inserts_eos(self):
        self.assertTrue(hasattr(ppo_math, "episode_response"))
        self.assertEqual(ppo_math.episode_response([4, 5], eos_token_id=2, max_new_tokens=2), ([4, 5], "length"))
        self.assertEqual(ppo_math.episode_response([4, 2], eos_token_id=2, max_new_tokens=2), ([4, 2], "eos"))
        for tokens in ([], [4], [4, 5, 6]):
            with self.assertRaises(ValueError):
                ppo_math.episode_response(tokens, eos_token_id=2, max_new_tokens=2)

    def test_gae_nonzero_values_and_discount_have_zero_terminal_bootstrap(self):
        # delta=[0.24,1.4]; A0=0.24+0.9*0.5*1.4=0.87.
        adv, ret = ppo_math.generalized_advantages([0.2, 2.0], [0.5, 0.6], gamma=0.9, gae_lambda=0.5)
        for actual, expected in zip(adv + ret, [0.87, 1.4, 1.37, 2.0]):
            self.assertAlmostEqual(actual, expected)
        adv, ret = ppo_math.generalized_advantages([2.0], [9.0], gamma=0.9, gae_lambda=0.5)
        self.assertEqual((adv, ret), ([-7.0], [2.0]))

    def test_reward_on_last_valid_action_includes_eos(self):
        rewards = ppo_math.terminal_kl_rewards(old_logprobs=[-1, -2], reference_logprobs=[-1.5, -1.5], scalar_reward=3, kl_coefficient=0.2, reward_scale=1, reward_cap=2)
        self.assertAlmostEqual(rewards[0], -0.1)
        self.assertAlmostEqual(rewards[1], 2.1)

    def test_raw_and_temperature_probabilities_really_differ(self):
        # A two-action policy: raw p(action 0)=2/3, T=.5 gives 4/5.
        raw = 2 / 3
        warped = 4 / 5
        self.assertNotAlmostEqual(raw, warped)
        self.assertAlmostEqual(math.exp(math.log(raw) - math.log(raw)), 1)
        self.assertNotAlmostEqual(math.exp(math.log(raw) - math.log(warped)), 1)


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_YAML = importlib.util.find_spec("yaml") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(HAS_TORCH and HAS_YAML, "requires CPU torch and PyYAML")
class TorchContractTests(unittest.TestCase):
    def test_state_action_shift_masks_prompt_and_preserves_eos_gradient(self):
        import torch
        from types import SimpleNamespace
        from coderl_lab.train.ppo_online import _token_logprobs_and_values

        class TinyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.table = torch.nn.Parameter(torch.tensor([[0., 0., 0.], [0., math.log(2), 0.], [0., 0., math.log(3)], [9., 0., 0.]]))

            def forward(self, input_ids, **kwargs):
                hidden = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
                return SimpleNamespace(logits=self.table.unsqueeze(0), hidden_states=[hidden])

        model = TinyPolicy()
        critic = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            critic.weight.fill_(1)
        logs, values = _token_logprobs_and_values(model, critic, [0, 0, 1, 2], 2, device="cpu", hidden=True)
        torch.testing.assert_close(logs, torch.tensor([math.log(0.5), math.log(0.6)]))
        torch.testing.assert_close(values, torch.tensor([1., 2.]))
        torch.testing.assert_close((logs - logs.detach()).exp(), torch.ones(2))
        (-logs.sum()).backward()
        self.assertEqual(model.table.grad[0].abs().sum().item(), 0)
        self.assertEqual(model.table.grad[3].abs().sum().item(), 0)
        self.assertGreater(model.table.grad[2].abs().sum().item(), 0)

    @unittest.skipUnless(HAS_TRANSFORMERS, "requires transformers")
    def test_generation_config_has_no_inherited_warpers(self):
        from coderl_lab.train import ppo_online
        self.assertTrue(hasattr(ppo_online, "_generation_config"))
        from types import SimpleNamespace
        cfg = ppo_online._generation_config({"sampling_temperature": 1, "sampling_top_p": 1, "max_new_tokens": 3}, SimpleNamespace(eos_token_id=2, pad_token_id=2))
        self.assertEqual((cfg.temperature, cfg.top_p, cfg.top_k), (1, 1, 0))
        self.assertEqual((cfg.repetition_penalty, cfg.num_beams, cfg.min_new_tokens), (1, 1, None))
        self.assertIsNone(cfg.forced_eos_token_id)
        self.assertTrue(cfg.do_sample)

    @unittest.skipUnless(HAS_TRANSFORMERS, "requires transformers")
    def test_real_tiny_transformer_rollout_scores_actual_sampling_distribution(self):
        import torch
        from types import SimpleNamespace
        from transformers import GPT2Config, GPT2LMHeadModel
        from coderl_lab.train.ppo_online import _make_rollout, _token_logprobs_and_values

        torch.manual_seed(71)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=7, n_positions=16, n_embd=8, n_layer=1, n_head=1))
        model.set_adapter = lambda name: None  # No adapters in this local CPU fixture.
        # Poison inherited defaults; the production generation call must ignore them.
        model.generation_config.top_k = 1
        model.generation_config.temperature = 0.1
        model.generation_config.repetition_penalty = 2.0
        model.generation_config.forced_eos_token_id = 2
        model.generation_config.min_length = 9
        critic = torch.nn.Linear(8, 1)
        torch.nn.init.zeros_(critic.weight)
        torch.nn.init.zeros_(critic.bias)
        tok = SimpleNamespace(eos_token_id=None, pad_token_id=0, encode=lambda *a, **k: [0, 1], decode=lambda *a, **k: "")
        cfg = {
            "rollout": {"sampling_temperature": 1, "sampling_top_p": 1, "max_new_tokens": 3, "policy_prompt_max_tokens": 8},
            "reward": {"empty_generation_reward": -2, "scalar_scale": 1, "scalar_clip_abs": 2},
            "ppo": {"kl_coefficient": 0.1, "gamma": 1, "gae_lambda": 1},
        }
        real_generate = model.generate
        captured = {}

        def capture_generate(**kwargs):
            output = real_generate(**kwargs, return_dict_in_generate=True, output_scores=True)
            captured["output"] = output
            return output.sequences

        model.generate = capture_generate
        episode = _make_rollout(model, critic, None, tok, prompt={"prompt": "fixture", "prompt_sha256": "fixture"}, config=cfg, device="cpu")
        output = captured["output"]
        sampled = output.sequences[0, 2:]
        behavior_logs = torch.stack([score[0].log_softmax(-1)[action] for score, action in zip(output.scores, sampled)])
        torch.testing.assert_close(torch.tensor(episode["old_logprobs"]), behavior_logs, atol=1e-6, rtol=1e-6)
        # Every vocabulary action retains finite mass, despite poisoned top-k=1.
        self.assertTrue(all(torch.isfinite(score).all().item() for score in output.scores))
        self.assertEqual(episode["end_reason"], "length")
        self.assertEqual(episode["returns"], [-2.0, -2.0, -2.0])
        new, _ = _token_logprobs_and_values(model, critic, episode["full_ids"], 2, device="cpu", hidden=False)
        torch.testing.assert_close((new - behavior_logs).exp(), torch.ones(3), atol=1e-6, rtol=1e-6)

    @unittest.skipUnless(HAS_TRANSFORMERS, "requires transformers")
    def test_rollout_crops_post_eos_tokens_before_scoring_and_reward(self):
        import torch
        from types import SimpleNamespace
        from coderl_lab.train.ppo_online import _make_rollout

        class EosPolicy(torch.nn.Module):
            def set_adapter(self, name):
                pass

            def generate(self, **kwargs):
                return torch.tensor([[0, 1, 2, 2, 2]])

            def forward(self, input_ids, **kwargs):
                # If post-EOS tokens leak into the scoring path this fails.
                self.scored_ids = input_ids[0].tolist()
                n = input_ids.shape[1]
                return SimpleNamespace(logits=torch.zeros(1, n, 3), hidden_states=[torch.zeros(1, n, 1)])

        model = EosPolicy()
        critic = torch.nn.Linear(1, 1)
        torch.nn.init.zeros_(critic.weight)
        torch.nn.init.zeros_(critic.bias)
        tok = SimpleNamespace(eos_token_id=2, pad_token_id=2, encode=lambda *a, **k: [0, 1], decode=lambda *a, **k: "")
        cfg = {"rollout": {"sampling_temperature": 1, "sampling_top_p": 1, "max_new_tokens": 3, "policy_prompt_max_tokens": 8}, "reward": {"empty_generation_reward": -2, "scalar_scale": 1, "scalar_clip_abs": 2}, "ppo": {"kl_coefficient": 0.1, "gamma": 1, "gae_lambda": 0.95}}
        episode = _make_rollout(model, critic, None, tok, prompt={"prompt": "fixture", "prompt_sha256": "fixture"}, config=cfg, device="cpu")
        self.assertEqual(model.scored_ids, [0, 1, 2])
        self.assertEqual(episode["response_tokens"], 1)
        self.assertEqual(episode["end_reason"], "eos")
        self.assertEqual(episode["returns"], [-2.0])
        self.assertEqual(len(episode["old_logprobs"]), 1)


if __name__ == "__main__":
    unittest.main()
