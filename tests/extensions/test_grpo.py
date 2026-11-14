"""The algorithm's invariants, its degenerate cases and its checkpoint contract.

What has to hold whatever the reward is: a group whose rewards are all equal
moves nothing; the KL penalty is non-negative, zero at the reference and
increasing in divergence; the surrogate at the first inner epoch is the plain
policy gradient; the loss is a token mean over the whole group and not over each
completion; and a resumed run continues the run it resumed rather than starting a
new one.

The tiny model here is a real ``MixtralForCausalLM`` -- the policy's own
architecture -- at a size that runs on a CPU, so the generation, the action mask
and the checkpoint round-trip are exercised on the thing they will face rather
than on a stub.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.capability.extensions import grpo  # noqa: E402


def _tiny_policy():
    from transformers import MixtralConfig, MixtralForCausalLM

    torch.manual_seed(0)
    config = MixtralConfig(
        vocab_size=31, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, num_local_experts=2,
        num_experts_per_tok=2, max_position_embeddings=64, bos_token_id=1,
        eos_token_id=2, pad_token_id=0, output_router_logits=False,
    )
    return MixtralForCausalLM(config).to(torch.float32).eval()


def _tokenizer_source(tmp_path, model=None):
    """A stand-in base directory: the declaration files a checkpoint copies verbatim."""

    source = tmp_path / "base"
    source.mkdir(parents=True, exist_ok=True)
    (source / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    (source / "tokenizer_config.json").write_text('{"add_bos_token": false}',
                                                  encoding="utf-8")
    # The base's own architectural declaration, in the 4.x spelling the
    # evaluation lane reads, including the rotary base a 5.x rewrite would move
    # out of ``rope_theta`` and into a block 4.x does not read.
    declaration = dict((model or _tiny_policy()).config.to_dict())
    declaration.update(rope_theta=1000000.0, torch_dtype="bfloat16",
                       declared_by="the base")
    declaration.pop("rope_parameters", None)
    (source / "config.json").write_text(json.dumps(declaration), encoding="utf-8")
    return source


class TestGroupAdvantages:
    def test_a_degenerate_group_contributes_exactly_zero(self):
        """Every reward equal: no direction exists, and none is invented."""

        advantages = grpo.group_advantages([0.4, 0.4, 0.4, 0.4])
        assert np.array_equal(advantages, np.zeros(4))

    def test_a_degenerate_group_of_zeros_is_also_exactly_zero(self):
        assert np.array_equal(grpo.group_advantages([0.0] * 6), np.zeros(6))

    def test_the_advantage_is_the_reward_minus_the_group_mean(self):
        advantages = grpo.group_advantages([0.0, 1.0, 2.0, 5.0])
        assert advantages == pytest.approx([-2.0, -1.0, 0.0, 3.0])
        assert float(advantages.sum()) == pytest.approx(0.0)

    def test_no_standard_deviation_is_divided_out(self):
        """Scaling the rewards scales the advantages; a standardised one would not."""

        plain = grpo.group_advantages([0.0, 1.0, 2.0])
        scaled = grpo.group_advantages([0.0, 10.0, 20.0])
        assert scaled == pytest.approx(10.0 * plain)

    def test_a_group_of_one_is_refused(self):
        with pytest.raises(ValueError, match="at least two rewards"):
            grpo.group_advantages([0.5])

    def test_a_non_finite_reward_is_refused(self):
        with pytest.raises(ValueError, match="finite reward"):
            grpo.group_advantages([0.1, float("nan")])

    def test_the_group_summary_flags_degeneracy(self):
        assert grpo.group_summary([0.3, 0.3, 0.3])["degenerate_group"] is True
        assert grpo.group_summary([0.3, 0.4, 0.3])["degenerate_group"] is False


class TestPolicyGradientLoss:
    def _inputs(self, *, advantages=(1.0, -1.0), tokens=3):
        logp_old = torch.full((len(advantages), tokens), -1.0)
        mask = torch.ones((len(advantages), tokens), dtype=torch.bool)
        return logp_old, torch.tensor(advantages), mask

    def test_at_the_first_inner_epoch_the_ratio_is_one_and_the_clip_is_inactive(self):
        logp_old, advantages, mask = self._inputs()
        logp = logp_old.clone().requires_grad_(True)
        loss, diagnostics = grpo.policy_gradient_loss(
            logp, logp_old, advantages, mask, clip_epsilon=0.2
        )
        assert diagnostics["ratio_mean"] == pytest.approx(1.0)
        assert diagnostics["clip_fraction"] == 0.0
        # -mean(A) over tokens, which for a centred group is zero
        assert float(loss.detach()) == pytest.approx(0.0)

    def test_the_first_epoch_gradient_is_the_plain_policy_gradient(self):
        """d/dlogp of -A*ratio at ratio one is -A, divided by the token count."""

        logp_old, advantages, mask = self._inputs(advantages=(2.0, -1.0), tokens=4)
        logp = logp_old.clone().requires_grad_(True)
        loss, _ = grpo.policy_gradient_loss(logp, logp_old, advantages, mask,
                                            clip_epsilon=0.2)
        loss.backward()
        expected = -advantages[:, None].expand_as(logp) / mask.sum()
        assert logp.grad == pytest.approx(expected, abs=1e-7)

    def test_a_positive_advantage_pushes_the_log_probability_up(self):
        logp_old, advantages, mask = self._inputs(advantages=(1.0, -1.0))
        logp = logp_old.clone().requires_grad_(True)
        loss, _ = grpo.policy_gradient_loss(logp, logp_old, advantages, mask,
                                            clip_epsilon=0.2)
        loss.backward()
        assert (logp.grad[0] < 0).all()  # descending the loss raises logp of the better one
        assert (logp.grad[1] > 0).all()

    def test_the_clip_bounds_the_surrogate_once_the_policy_has_moved(self):
        logp_old, advantages, mask = self._inputs(advantages=(1.0, -1.0))
        moved = logp_old + 1.0  # ratio e ~ 2.72, far outside 1 +- 0.2
        loss, diagnostics = grpo.policy_gradient_loss(
            moved, logp_old, advantages, mask, clip_epsilon=0.2
        )
        assert diagnostics["clip_fraction"] == 1.0
        # the improved completion is capped at 1.2 while the worsened one is not
        assert float(loss) == pytest.approx(-(1.2 * 1.0 + np.e * -1.0) / 2.0, abs=1e-5)

    def test_the_loss_is_a_token_mean_over_the_group_not_over_each_completion(self):
        """A long completion contributes in proportion to its tokens, as declared.

        A per-completion normaliser would give the two rows equal weight; the
        declared one divides by the group's total tokens, so the four-token row
        carries twice the weight of the two-token row.
        """

        logp_old = torch.full((2, 4), -1.0)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        advantages = torch.tensor([1.0, 1.0])
        logp = logp_old + torch.tensor([[0.5], [0.0]])
        loss, diagnostics = grpo.policy_gradient_loss(
            logp, logp_old, advantages, mask, clip_epsilon=1.0 - 1e-9
        )
        assert diagnostics["action_tokens"] == 6
        # clip range is effectively (0, 2), so the surrogate is ratio * A
        expected = -(4 * min(np.exp(0.5), 2.0 - 1e-9) + 2 * 1.0) / 6
        assert float(loss) == pytest.approx(expected, abs=1e-5)

    def test_masked_tokens_do_not_enter_the_loss(self):
        logp_old = torch.full((2, 4), -1.0)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
        advantages = torch.tensor([1.0, -1.0])
        logp = logp_old.clone()
        logp.data[:, 2:] = -50.0  # nonsense in the padded tail
        loss, diagnostics = grpo.policy_gradient_loss(
            logp, logp_old, advantages, mask, clip_epsilon=0.2
        )
        assert diagnostics["action_tokens"] == 4
        assert float(loss) == pytest.approx(0.0)

    def test_a_group_with_no_action_token_is_refused(self):
        logp_old = torch.full((2, 3), -1.0)
        mask = torch.zeros((2, 3), dtype=torch.bool)
        with pytest.raises(ValueError, match="at least one action token"):
            grpo.policy_gradient_loss(logp_old, logp_old, torch.tensor([1.0, -1.0]),
                                      mask, clip_epsilon=0.2)

    def test_mismatched_shapes_are_refused(self):
        logp_old = torch.full((2, 3), -1.0)
        mask = torch.ones((2, 3), dtype=torch.bool)
        with pytest.raises(ValueError, match="one advantage per completion"):
            grpo.policy_gradient_loss(logp_old, logp_old, torch.tensor([1.0]), mask,
                                      clip_epsilon=0.2)
        with pytest.raises(ValueError, match="share one shape"):
            grpo.policy_gradient_loss(logp_old, torch.full((2, 4), -1.0),
                                      torch.tensor([1.0, -1.0]), mask, clip_epsilon=0.2)

    def test_a_clip_range_outside_the_unit_interval_is_refused(self):
        logp_old, advantages, mask = self._inputs()
        for bad in (0.0, 1.0, -0.1, 2.0):
            with pytest.raises(ValueError, match="strictly inside"):
                grpo.policy_gradient_loss(logp_old, logp_old, advantages, mask,
                                          clip_epsilon=bad)


class TestReferenceKL:
    def test_it_is_zero_exactly_at_the_reference(self):
        logp = torch.full((3, 5), -2.0)
        mask = torch.ones((3, 5), dtype=torch.bool)
        assert float(grpo.reference_kl(logp, logp.clone(), mask)) == pytest.approx(0.0)

    def test_it_is_non_negative_and_grows_with_divergence_in_either_direction(self):
        mask = torch.ones((1, 4), dtype=torch.bool)
        reference = torch.full((1, 4), -2.0)
        near = float(grpo.reference_kl(reference + 0.1, reference, mask))
        far = float(grpo.reference_kl(reference + 1.0, reference, mask))
        below = float(grpo.reference_kl(reference - 1.0, reference, mask))
        assert 0.0 < near < far
        assert below > 0.0

    def test_the_penalty_raises_the_loss_when_the_policy_leaves_the_reference(self):
        """The sign that matters: diverging costs, it does not pay."""

        mask = torch.ones((2, 3), dtype=torch.bool)
        logp_old = torch.full((2, 3), -1.0)
        reference = torch.full((2, 3), -1.0)
        advantages = torch.tensor([0.0, 0.0])  # no policy-gradient term at all
        at_reference, _ = grpo.total_loss(logp_old, logp_old, reference, advantages, mask,
                                          clip_epsilon=0.2, kl_coefficient=0.5)
        away, _ = grpo.total_loss(logp_old - 0.5, logp_old, reference, advantages, mask,
                                  clip_epsilon=0.2, kl_coefficient=0.5)
        assert float(away) > float(at_reference)
        assert float(at_reference) == pytest.approx(0.0, abs=1e-7)

    def test_the_gradient_of_the_penalty_pulls_back_toward_the_reference(self):
        mask = torch.ones((1, 3), dtype=torch.bool)
        reference = torch.full((1, 3), -1.0)
        logp = (reference - 0.5).clone().requires_grad_(True)
        grpo.reference_kl(logp, reference, mask).backward()
        # below the reference, so the pull is upward: descending reduces the gap
        assert (logp.grad < 0).all()

    def test_a_zero_coefficient_removes_the_penalty_entirely(self):
        mask = torch.ones((2, 3), dtype=torch.bool)
        logp_old = torch.full((2, 3), -1.0)
        reference = torch.full((2, 3), -3.0)
        advantages = torch.tensor([0.0, 0.0])
        loss, diagnostics = grpo.total_loss(logp_old, logp_old, reference, advantages, mask,
                                            clip_epsilon=0.2, kl_coefficient=0.0)
        assert diagnostics["reference_kl"] > 0.0
        assert float(loss) == pytest.approx(0.0, abs=1e-7)


class TestSettings:
    def test_the_declared_fields_are_validated(self):
        def make(**overrides):
            fields = dict(group_size=8, steps=4, inner_epochs=2, clip_epsilon=0.2,
                          kl_coefficient=0.02, learning_rate=1e-6, grad_clip=1.0,
                          max_new_tokens=16, rollout_temperature=1.0, seed=1,
                          checkpoint_every=2, milestone_every=4)
            fields.update(overrides)
            return grpo.GRPOSettings(**fields)

        make()  # the valid case
        with pytest.raises(ValueError, match="at least two completions"):
            make(group_size=1)
        with pytest.raises(ValueError, match="positive"):
            make(inner_epochs=0)
        with pytest.raises(ValueError, match="strictly inside"):
            make(clip_epsilon=0.0)
        with pytest.raises(ValueError, match="non-negative"):
            make(kl_coefficient=-0.1)
        with pytest.raises(ValueError, match="positive"):
            make(learning_rate=0.0)
        with pytest.raises(ValueError, match="completion budget"):
            make(max_new_tokens=0)
        with pytest.raises(ValueError, match="rollout temperature"):
            make(rollout_temperature=0.0)


class TestActionMask:
    def test_it_covers_the_stop_token_and_nothing_after_it(self):
        completions = torch.tensor([[6, 7, 2, 0, 0], [6, 7, 8, 9, 10]])
        mask = grpo._action_mask(completions, eos_id=2)
        assert mask.tolist() == [[True, True, True, False, False],
                                 [True, True, True, True, True]]

    def test_only_the_first_stop_closes_a_row(self):
        completions = torch.tensor([[6, 2, 2, 7, 2]])
        assert grpo._action_mask(completions, eos_id=2).tolist() == [
            [True, True, False, False, False]
        ]

    def test_an_immediate_stop_is_one_action_and_not_zero(self):
        """Stopping at once is a choice, and the gradient has to be able to reach it."""

        mask = grpo._action_mask(torch.tensor([[2, 6, 7]]), eos_id=2)
        assert mask.tolist() == [[True, False, False]]
        assert int(mask.sum()) == 1


class TestRolloutAndScoring:
    def test_the_rollout_is_reproducible_from_its_declared_seed(self):
        model = _tiny_policy()
        first = grpo.rollout(model, prompt_ids=[1, 4], group_size=4, max_new_tokens=8,
                             temperature=1.0, eos_id=2, pad_id=0, seed=1234)
        again = grpo.rollout(model, prompt_ids=[1, 4], group_size=4, max_new_tokens=8,
                             temperature=1.0, eos_id=2, pad_id=0, seed=1234)
        other = grpo.rollout(model, prompt_ids=[1, 4], group_size=4, max_new_tokens=8,
                             temperature=1.0, eos_id=2, pad_id=0, seed=1235)
        assert torch.equal(first["input_ids"], again["input_ids"])
        assert not torch.equal(first["input_ids"], other["input_ids"])

    def test_token_log_probs_match_a_direct_forward_pass(self):
        model = _tiny_policy()
        ids = torch.tensor([[1, 4, 6, 7, 8], [1, 4, 9, 10, 11]])
        mask = torch.ones_like(ids)
        with torch.no_grad():
            got = grpo.token_log_probs(model, ids, mask, prompt_length=2)
            logits = model(input_ids=ids, attention_mask=mask).logits
        expected = torch.log_softmax(logits.float(), dim=-1)
        for row in range(ids.shape[0]):
            for position in range(3):
                target = int(ids[row, 2 + position])
                assert float(got[row, position]) == pytest.approx(
                    float(expected[row, 1 + position, target]), abs=1e-5
                )

    def test_a_rollout_group_of_one_is_refused(self):
        with pytest.raises(ValueError, match="at least two completions"):
            grpo.rollout(_tiny_policy(), prompt_ids=[1, 4], group_size=1,
                         max_new_tokens=4, temperature=1.0, eos_id=2, pad_id=0, seed=1)

    def test_entropy_is_positive_and_bounded_by_the_log_of_the_vocabulary(self):
        model = _tiny_policy()
        ids = torch.tensor([[1, 4, 6, 7, 8]])
        mask = torch.ones_like(ids)
        action = torch.ones((1, 3), dtype=torch.bool)
        value = grpo.sequence_entropy(model, ids, mask, action, prompt_length=2)
        assert 0.0 < value <= float(np.log(31))


class TestCheckpointAndResume:
    def _run(self, model, optimiser, steps, *, start=0):
        """A deterministic surrogate of the trainer's inner loop.

        The reward is a fixed vector and the rollout is replaced by fixed ids, so
        the only thing under test is whether the optimiser and the weights resume
        where they stopped. The real loop's rollout is reproducible from its step
        seed, which :class:`TestRolloutAndScoring` checks separately; the real
        loop's reward for two of the three arms is a GPU fold, which this project
        has measured not to be bit-reproducible, so bitwise resume equality is a
        property of the trainer and not of those arms.
        """

        ids = torch.tensor([[1, 4, 6, 7, 8], [1, 4, 9, 10, 11]])
        mask = torch.ones_like(ids)
        action = torch.ones((2, 3), dtype=torch.bool)
        advantages = torch.tensor([0.5, -0.5])
        for step in range(start, steps):
            with torch.no_grad():
                logp_old = grpo.token_log_probs(model, ids, mask, prompt_length=2)
            reference = logp_old.clone()
            for _ in range(2):
                logp = grpo.token_log_probs(model, ids, mask, prompt_length=2)
                loss, _ = grpo.total_loss(logp, logp_old, reference, advantages, action,
                                          clip_epsilon=0.2, kl_coefficient=0.02)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
        return model

    def test_a_resumed_run_reaches_the_weights_an_uninterrupted_run_reaches(self, tmp_path):
        settings = grpo.GRPOSettings(
            group_size=2, steps=4, inner_epochs=2, clip_epsilon=0.2, kl_coefficient=0.02,
            learning_rate=1e-3, grad_clip=1.0, max_new_tokens=8, rollout_temperature=1.0,
            seed=7, checkpoint_every=2, milestone_every=4,
        )
        straight = _tiny_policy().train()
        straight_optimiser = torch.optim.AdamW(straight.parameters(), lr=settings.learning_rate)
        self._run(straight, straight_optimiser, 4)

        interrupted = _tiny_policy().train()
        interrupted_optimiser = torch.optim.AdamW(interrupted.parameters(),
                                                  lr=settings.learning_rate)
        self._run(interrupted, interrupted_optimiser, 2)
        directory = tmp_path / grpo.LATEST_DIR
        grpo.save_checkpoint(directory, model=interrupted,
                             tokenizer_source=_tokenizer_source(tmp_path),
                             optimiser=interrupted_optimiser, step=1, settings=settings,
                             extra={"arm": "environment"})

        from transformers import MixtralForCausalLM

        resumed = MixtralForCausalLM.from_pretrained(directory, dtype=torch.float32).train()
        resumed_optimiser = torch.optim.AdamW(resumed.parameters(), lr=settings.learning_rate)
        state = grpo.load_checkpoint(directory, model=resumed, optimiser=resumed_optimiser)
        assert state["step"] == 1
        assert state["extra"]["arm"] == "environment"
        self._run(resumed, resumed_optimiser, 4, start=2)

        for (name, left), (_, right) in zip(straight.named_parameters(),
                                            resumed.named_parameters()):
            assert torch.allclose(left, right, atol=1e-6), name

    def test_the_checkpoint_is_a_loadable_checkpoint_directory(self, tmp_path):
        settings = grpo.GRPOSettings(
            group_size=2, steps=1, inner_epochs=1, clip_epsilon=0.2, kl_coefficient=0.0,
            learning_rate=1e-4, grad_clip=1.0, max_new_tokens=4, rollout_temperature=1.0,
            seed=1, checkpoint_every=1, milestone_every=1,
        )
        model = _tiny_policy()
        optimiser = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
        directory = grpo.save_checkpoint(tmp_path / "latest", model=model,
                                         tokenizer_source=_tokenizer_source(tmp_path),
                                         optimiser=optimiser, step=0, settings=settings)
        assert (directory / "config.json").is_file()
        written = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        # the base's declaration, carried verbatim but for the stored precision
        assert written["declared_by"] == "the base"
        assert written["rope_theta"] == 1000000.0
        assert written[grpo.STORED_PRECISION_FIELD] == "float32"
        assert "rope_parameters" not in written
        assert (directory / grpo.TRAINER_STATE).is_file()
        assert (directory / "tokenizer.json").is_file()
        assert (directory / "tokenizer_config.json").is_file()
        assert not (directory.parent / "latest.pending").exists()

    def test_overwriting_a_checkpoint_leaves_no_partial_directory(self, tmp_path):
        settings = grpo.GRPOSettings(
            group_size=2, steps=2, inner_epochs=1, clip_epsilon=0.2, kl_coefficient=0.0,
            learning_rate=1e-4, grad_clip=1.0, max_new_tokens=4, rollout_temperature=1.0,
            seed=1, checkpoint_every=1, milestone_every=2,
        )
        model = _tiny_policy()
        optimiser = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
        for step in range(3):
            grpo.save_checkpoint(tmp_path / "latest", model=model,
                                 tokenizer_source=_tokenizer_source(tmp_path),
                                 optimiser=optimiser, step=step, settings=settings)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["base", "latest"]
        state = grpo.load_checkpoint(tmp_path / "latest", model=model, optimiser=optimiser)
        assert state["step"] == 2

    def test_a_source_with_no_config_is_refused(self, tmp_path):
        settings = grpo.GRPOSettings(
            group_size=2, steps=1, inner_epochs=1, clip_epsilon=0.2, kl_coefficient=0.0,
            learning_rate=1e-4, grad_clip=1.0, max_new_tokens=4, rollout_temperature=1.0,
            seed=1, checkpoint_every=1, milestone_every=1,
        )
        model = _tiny_policy()
        optimiser = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
        source = _tokenizer_source(tmp_path)
        (source / "config.json").unlink()
        with pytest.raises(ValueError, match="no config.json"):
            grpo.save_checkpoint(tmp_path / "latest", model=model, tokenizer_source=source,
                                 optimiser=optimiser, step=0, settings=settings)

    def test_a_tokenizer_source_with_no_tokenizer_is_refused(self, tmp_path):
        settings = grpo.GRPOSettings(
            group_size=2, steps=1, inner_epochs=1, clip_epsilon=0.2, kl_coefficient=0.0,
            learning_rate=1e-4, grad_clip=1.0, max_new_tokens=4, rollout_temperature=1.0,
            seed=1, checkpoint_every=1, milestone_every=1,
        )
        model = _tiny_policy()
        optimiser = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="no tokenizer file"):
            grpo.save_checkpoint(tmp_path / "latest", model=model, tokenizer_source=empty,
                                 optimiser=optimiser, step=0, settings=settings)

    def test_a_resume_under_changed_hyperparameters_is_refused(self):
        settings = grpo.GRPOSettings(
            group_size=8, steps=4, inner_epochs=2, clip_epsilon=0.2, kl_coefficient=0.02,
            learning_rate=1e-6, grad_clip=1.0, max_new_tokens=16, rollout_temperature=1.0,
            seed=1, checkpoint_every=2, milestone_every=4,
        )
        grpo.require_same_settings(settings.as_record(), settings)
        # extending the budget is allowed: it changes nothing a completed step did
        import dataclasses

        grpo.require_same_settings(settings.as_record(),
                                   dataclasses.replace(settings, steps=64))
        for changed in ("learning_rate", "group_size", "kl_coefficient", "seed"):
            altered = dataclasses.replace(settings, **{changed: 0.5 if changed in
                                                       ("learning_rate", "kl_coefficient")
                                                       else 99})
            with pytest.raises(ValueError, match="changed hyperparameters"):
                grpo.require_same_settings(settings.as_record(), altered)


class TestTrajectory:
    def test_a_resumed_step_replaces_its_earlier_record(self, tmp_path):
        """A preemption between a checkpoint and the next leaves steps replayed."""

        path = tmp_path / grpo.TRAJECTORY
        for step, reward in ((0, 0.1), (1, 0.2), (2, 0.3)):
            grpo.append_trajectory(path, {"step": step, "reward_mean": reward})
        grpo.append_trajectory(path, {"step": 2, "reward_mean": 0.9})
        trajectory = grpo.read_trajectory(path)
        assert [row["step"] for row in trajectory] == [0, 1, 2]
        assert trajectory[-1]["reward_mean"] == 0.9

    def test_an_absent_trajectory_is_empty_rather_than_an_error(self, tmp_path):
        assert grpo.read_trajectory(tmp_path / "nothing.jsonl") == []

    def test_each_record_is_one_flushed_line(self, tmp_path):
        path = tmp_path / grpo.TRAJECTORY
        grpo.append_trajectory(path, {"step": 0, "reward_mean": 0.5})
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1 and json.loads(lines[0])["step"] == 0

    def test_movement_refuses_a_window_the_trajectory_cannot_fill(self, tmp_path):
        trajectory = [{"step": index, "reward_mean": 0.1 * index} for index in range(10)]
        assert grpo.optimisation_moved(trajectory, window=20)["measurable"] is False
        moved = grpo.optimisation_moved(trajectory, window=3)
        assert moved["measurable"] is True
        assert moved["change"] == pytest.approx(0.7, abs=1e-9)


class TestEntropy:
    """Per-sequence entropy, which the declared entropy family resamples."""

    def _batch(self):
        ids = torch.tensor([[1, 4, 6, 7, 8, 0], [1, 4, 9, 10, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]])
        action = mask[:, 2:].bool()
        return ids, mask, action

    def test_each_sequence_carries_its_own_value(self):
        model = _tiny_policy()
        ids, mask, action = self._batch()
        values = grpo.per_sequence_entropy(model, ids, mask, action, prompt_length=2)
        assert len(values) == 2
        assert all(0.0 < value <= float(np.log(31)) for value in values)

    def test_the_group_mean_is_the_token_weighted_mean_of_the_per_sequence_values(self):
        """A per-sequence family and a group diagnostic must not disagree."""

        model = _tiny_policy()
        ids, mask, action = self._batch()
        values = grpo.per_sequence_entropy(model, ids, mask, action, prompt_length=2)
        counts = action.sum(dim=1).tolist()
        weighted = sum(v * c for v, c in zip(values, counts)) / sum(counts)
        group = grpo.sequence_entropy(model, ids, mask, action, prompt_length=2)
        assert group == pytest.approx(weighted, abs=1e-5)

    def test_padded_positions_do_not_enter_a_sequence_value(self):
        model = _tiny_policy()
        ids, mask, action = self._batch()
        padded = grpo.per_sequence_entropy(model, ids, mask, action, prompt_length=2)
        short_ids = torch.tensor([[1, 4, 9, 10]])
        short_mask = torch.ones_like(short_ids)
        alone = grpo.per_sequence_entropy(model, short_ids, short_mask,
                                          short_mask[:, 2:].bool(), prompt_length=2)
        assert padded[1] == pytest.approx(alone[0], abs=1e-5)

    def test_token_entropies_has_one_value_per_generated_position(self):
        model = _tiny_policy()
        ids, mask, _ = self._batch()
        per_token = grpo.token_entropies(model, ids, mask, prompt_length=2)
        assert tuple(per_token.shape) == (2, ids.shape[1] - 2)
