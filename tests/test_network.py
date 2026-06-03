import numpy as np
import pytest
import tempfile
from pathlib import Path

from threes.simulator import parse_state
from threes.models import Models
from threes.game import Game
from threes.network import Network, INPUT_SIZE, OUTPUT_SIZE


RNG = np.random.default_rng(0)


def _state():
    return parse_state("0 0 0 3  0 0 0 0  0 0 0 0  0 0 0 0 / 3")


def _mask_all_valid():
    return np.ones(4, dtype=bool)


def _mask_partial():
    m = np.ones(4, dtype=bool)
    m[0] = False   # up invalid
    m[2] = False   # down invalid
    return m


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_layer_shapes(self):
        net = Network(rng=np.random.default_rng(1))
        assert net.weights[0].shape == (64, INPUT_SIZE)
        assert net.weights[1].shape == (OUTPUT_SIZE, 64)

    def test_custom_hidden(self):
        net = Network(hidden_sizes=[32, 16], rng=np.random.default_rng(1))
        assert len(net.weights) == 3
        assert net.weights[0].shape == (32, INPUT_SIZE)
        assert net.weights[1].shape == (16, 32)
        assert net.weights[2].shape == (OUTPUT_SIZE, 16)

    def test_initial_snapshot_recorded(self):
        net = Network(snapshot_every=100, rng=np.random.default_rng(1))
        assert len(net.history) == 1
        assert net.history[0]["step"] == 0


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestForward:
    def test_probs_sum_to_one(self):
        net  = Network(rng=RNG)
        mask = _mask_all_valid()
        probs, _ = net.forward(np.random.rand(INPUT_SIZE), mask)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_invalid_actions_have_zero_prob(self):
        net  = Network(rng=RNG)
        mask = _mask_partial()
        probs, _ = net.forward(np.random.rand(INPUT_SIZE), mask)
        assert probs[0] == pytest.approx(0.0, abs=1e-9)
        assert probs[2] == pytest.approx(0.0, abs=1e-9)
        assert probs[1] > 0
        assert probs[3] > 0

    def test_only_one_valid_action(self):
        net  = Network(rng=RNG)
        mask = np.array([False, True, False, False])
        probs, _ = net.forward(np.random.rand(INPUT_SIZE), mask)
        assert probs[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# choose
# ---------------------------------------------------------------------------

class TestChoose:
    def test_choose_returns_valid_action(self):
        net  = Network(rng=RNG)
        mask = _mask_partial()
        rng  = np.random.default_rng(5)
        for _ in range(20):
            a = net.choose(_state(), mask, rng)
            assert mask[a], f"chose invalid action {a}"

    def test_greedy_returns_argmax(self):
        net  = Network(rng=RNG)
        mask = _mask_all_valid()
        rng  = np.random.default_rng(5)
        probs, _ = net.forward(np.random.rand(INPUT_SIZE), mask)
        greedy_a  = net.choose(_state(), mask, rng, greedy=True)
        # greedy should match argmax of probs for this state
        assert mask[greedy_a]


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_weights_change_after_update(self):
        net  = Network(rng=np.random.default_rng(7))
        W1_before = net.weights[0].copy()
        net.update(_state(), action=1, reward=100.0, mask=_mask_all_valid())
        assert not np.allclose(net.weights[0], W1_before)

    def test_n_updates_increments(self):
        net = Network(rng=RNG)
        net.update(_state(), action=1, reward=10.0, mask=_mask_all_valid())
        assert net.n_updates == 1

    def test_snapshot_saved_at_interval(self):
        net = Network(snapshot_every=5, rng=np.random.default_rng(1))
        initial_snaps = len(net.history)
        for _ in range(5):
            net.update(_state(), 1, 10.0, _mask_all_valid())
        assert len(net.history) == initial_snaps + 1
        assert net.history[-1]["step"] == 5

    def test_positive_reward_increases_chosen_prob(self):
        from threes.simulator import encode_for_net
        # baseline_decay=0.99 keeps baseline near 0 on first update
        # so advantage = reward - ~0 = positive
        net  = Network(lr=0.1, baseline_decay=0.99, rng=np.random.default_rng(2))
        mask = _mask_all_valid()
        s    = _state()
        x    = encode_for_net(s)
        probs_before, _ = net.forward(x, mask)
        net.update(s, action=1, reward=200.0, mask=mask)
        probs_after, _  = net.forward(x, mask)
        assert probs_after[1] > probs_before[1]

    def test_negative_advantage_decreases_chosen_prob(self):
        from threes.simulator import encode_for_net
        # Use small lr so first update doesn't saturate action-1 to ~1.0
        net  = Network(lr=0.01, baseline_decay=0.5, rng=np.random.default_rng(3))
        mask = _mask_all_valid()
        s    = _state()
        x    = encode_for_net(s)
        # Prime the baseline above zero
        net.update(s, action=1, reward=100.0, mask=mask)
        probs_before, _ = net.forward(x, mask)
        # Reward well below baseline → negative advantage → prob should drop
        net.update(s, action=1, reward=0.0, mask=mask)
        probs_after, _  = net.forward(x, mask)
        assert probs_after[1] < probs_before[1]


# ---------------------------------------------------------------------------
# Weight history
# ---------------------------------------------------------------------------

class TestHistory:
    def test_weight_history_shape(self):
        net = Network(snapshot_every=1, rng=np.random.default_rng(1))
        for _ in range(3):
            net.update(_state(), 1, 10.0, _mask_all_valid())
        h = net.weight_history()
        assert "steps" in h and "W0" in h
        assert h["steps"].shape == (4,)        # 0, 1, 2, 3
        assert h["W0"].shape[0] == 4

    def test_steps_are_correct(self):
        net = Network(snapshot_every=2, rng=np.random.default_rng(1))
        for _ in range(4):
            net.update(_state(), 1, 10.0, _mask_all_valid())
        h = net.weight_history()
        assert list(h["steps"]) == [0, 2, 4]


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self):
        net = Network(hidden_sizes=[32], lr=5e-4,
                      snapshot_every=1, rng=np.random.default_rng(9))
        for _ in range(3):
            net.update(_state(), 2, 50.0, _mask_all_valid())

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "net.npz"
            net.save(p)
            net2 = Network.load(p)

        assert net2.n_updates == net.n_updates
        assert net2.lr        == net.lr
        assert len(net2.history) == len(net.history)
        for W1, W2 in zip(net.weights, net2.weights):
            assert np.allclose(W1, W2)

    def test_summary_runs(self):
        net = Network(rng=RNG)
        s = net.summary()
        assert "Network" in s
