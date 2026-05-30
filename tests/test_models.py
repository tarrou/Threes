import numpy as np
import pytest
import tempfile
from pathlib import Path

from threes.simulator import GameState, NextTile, Observation, parse_state, _apply_slide
from threes.models import (
    NextTileModel, PlacementModel, StartBoardModel, Models,
    step, GAME_OVER, TILE_INDEX, N_TILES,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_obs(before_str: str, action: int, after_str: str, is_real: bool) -> Observation:
    return Observation(
        state_before=parse_state(before_str),
        action=action,
        state_after=parse_state(after_str),
        is_real=is_real,
    )


# ---------------------------------------------------------------------------
# NextTileModel
# ---------------------------------------------------------------------------

class TestNextTileModel:
    def test_update_and_pmf_sums_to_one(self):
        m = NextTileModel()
        obs = make_obs(
            "0 0 0 0  0 0 0 0  0 0 0 0  0 0 3 0 / 1",
            3,
            "0 0 0 0  0 0 0 0  0 0 0 0  0 3 0 0 / 2",
            is_real=True,
        )
        m.update(obs)
        board = parse_state("0 0 0 0  0 0 0 0  0 0 0 0  0 0 3 0 / 1").board
        p = m.pmf(board)
        assert abs(p.sum() - 1.0) < 1e-9
        assert p[TILE_INDEX[2]] > 0.9   # only observed value is 2

    def test_prior_before_any_data(self):
        m = NextTileModel()
        board = np.zeros((4, 4), dtype=int)
        p = m.pmf(board)
        assert abs(p.sum() - 1.0) < 1e-9
        # Prior: uniform over 1, 2, 3
        assert p[TILE_INDEX[1]] == pytest.approx(1/3)
        assert p[TILE_INDEX[2]] == pytest.approx(1/3)
        assert p[TILE_INDEX[3]] == pytest.approx(1/3)

    def test_bonus_hint_splits_credit(self):
        m = NextTileModel()
        obs = make_obs(
            "0 0 0 0  0 0 0 0  0 0 0 0  0 0 3 0 / 1",
            3,
            "0 0 0 0  0 0 0 0  0 0 0 0  0 3 0 0 / 6 12 24",
            is_real=True,
        )
        m.update(obs)
        board = parse_state("0 0 0 0  0 0 0 0  0 0 0 0  0 0 3 0 / 1").board
        p = m.pmf(board)
        # Each of 6, 12, 24 should have equal weight
        assert p[TILE_INDEX[6]] == pytest.approx(p[TILE_INDEX[12]])
        assert p[TILE_INDEX[12]] == pytest.approx(p[TILE_INDEX[24]])

    def test_real_weight_dominates(self):
        m = NextTileModel()
        # 1 real obs for tile=1, 100 simulated for tile=3
        board = np.zeros((4, 4), dtype=int)
        real_obs = make_obs(
            "0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 / 2",
            3,
            "0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 / 1",
            is_real=True,
        )
        m.update(real_obs)
        for _ in range(100):
            sim_obs = make_obs(
                "0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 / 2",
                3,
                "0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 / 3",
                is_real=False,
            )
            m.update(sim_obs)
        p = m.pmf(board, real_weight=10.0)
        # 1 real (weight 10) vs 100 sim (weight 1): tile=1 gets 10/(10+100) ≈ 9%
        # tile=3 gets 100/(10+100) ≈ 91%
        assert p[TILE_INDEX[1]] == pytest.approx(10 / 110)
        assert p[TILE_INDEX[3]] == pytest.approx(100 / 110)

    def test_sample_returns_valid_tile(self):
        m = NextTileModel()
        board = np.zeros((4, 4), dtype=int)
        rng = np.random.default_rng(0)
        tile = m.sample(board, rng)
        assert tile in {1, 2, 3}

    def test_save_load_roundtrip(self):
        m = NextTileModel()
        obs = make_obs(
            "0 0 0 0  0 0 0 0  0 0 0 0  0 0 3 0 / 1",
            3,
            "0 0 0 0  0 0 0 0  0 0 0 0  0 3 0 0 / 2",
            is_real=True,
        )
        m.update(obs)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "nt.npz"
            m.save(p)
            m2 = NextTileModel.load(p)
        board = parse_state("0 0 0 0  0 0 0 0  0 0 0 0  0 0 3 0 / 1").board
        assert np.allclose(m.pmf(board), m2.pmf(board))


# ---------------------------------------------------------------------------
# PlacementModel
# ---------------------------------------------------------------------------

class TestPlacementModel:
    def _make_placement_obs(self, is_real=True):
        # Slide left on a board where col 3 is vacated in rows 0 and 2.
        # We manually construct state_after with the tile placed at (0,3).
        before = parse_state("0 0 0 3  0 0 0 0  0 0 0 6  0 0 0 0 / 3")
        # After slide left: row0 -> [3,0,0,0], row2 -> [6,0,0,0]
        # eligible = [(0,3), (2,3)]. Place tile at (0,3).
        after_board = np.array([
            3, 0, 0, 3,
            0, 0, 0, 0,
            6, 0, 0, 0,
            0, 0, 0, 0,
        ], dtype=int).reshape(4, 4)
        after = GameState(after_board, NextTile([1]))
        return Observation(before, 3, after, is_real=is_real)

    def test_update_and_pmf(self):
        m = PlacementModel()
        obs = self._make_placement_obs(is_real=True)
        for _ in range(10):
            m.update(obs)
        # eligible sorted = [(0,3), (2,3)]; always placed at (0,3) = index 0
        p = m.pmf(3, [(0, 3), (2, 3)], real_weight=10.0)
        assert p[0] > p[1]   # index 0 should dominate

    def test_uniform_fallback(self):
        m = PlacementModel()
        p = m.pmf(0, [(0, 0), (1, 0), (2, 0)], real_weight=10.0)
        assert np.allclose(p, [1/3, 1/3, 1/3])

    def test_sample_returns_eligible_position(self):
        m = PlacementModel()
        eligible = [(0, 3), (2, 3)]
        rng = np.random.default_rng(0)
        pos = m.sample(3, eligible, rng)
        assert pos in eligible

    def test_save_load_roundtrip(self):
        m = PlacementModel()
        obs = self._make_placement_obs(is_real=True)
        m.update(obs)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "pl.npz"
            m.save(p)
            m2 = PlacementModel.load(p)
        elig = [(0, 3), (2, 3)]
        assert np.allclose(m.pmf(3, elig), m2.pmf(3, elig))


# ---------------------------------------------------------------------------
# StartBoardModel
# ---------------------------------------------------------------------------

class TestStartBoardModel:
    def test_sample_returns_none_when_empty(self):
        m = StartBoardModel()
        assert m.sample(RNG) is None

    def test_sample_after_update(self):
        m = StartBoardModel()
        s = parse_state("1 2 1 0  0 3 0 0  0 0 1 0  2 0 0 0 / 1")
        m.update(s, is_real=True)
        result = m.sample(np.random.default_rng(0))
        assert result is not None
        assert np.array_equal(result.board, s.board)

    def test_real_examples_sampled_more(self):
        m = StartBoardModel()
        real_state = parse_state("3 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 / 1")
        sim_state  = parse_state("0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 6 / 2")
        m.update(real_state, is_real=True)
        for _ in range(10):
            m.update(sim_state, is_real=False)
        rng = np.random.default_rng(1)
        samples = [m.sample(rng, real_weight=10.0) for _ in range(1000)]
        real_count = sum(1 for s in samples if s.board[0, 0] == 3)
        # 1 real (weight 10) vs 10 sim (weight 1): expect ~10/20 = 50% real
        assert 400 < real_count < 600

    def test_save_load_roundtrip(self):
        m = StartBoardModel()
        s1 = parse_state("1 2 1 0  0 3 0 0  0 0 1 0  2 0 0 0 / 1")
        s2 = parse_state("1 2 1 0  0 3 0 0  0 0 1 0  2 0 0 0 / 6 12 24")
        m.update(s1, is_real=True)
        m.update(s2, is_real=False)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sb.npz"
            m.save(p)
            m2 = StartBoardModel.load(p)
        assert m2.n_observations() == m.n_observations()
        assert m2._next_tiles[1] == [6, 12, 24]


# ---------------------------------------------------------------------------
# Models bundle
# ---------------------------------------------------------------------------

class TestModels:
    def test_save_load_roundtrip(self):
        m = Models()
        obs = make_obs(
            "0 0 0 3  0 0 0 0  0 0 0 6  0 0 0 0 / 3",
            3,
            "3 0 0 3  0 0 0 0  6 0 0 0  0 0 0 0 / 1",
            is_real=True,
        )
        m.update(obs)
        with tempfile.TemporaryDirectory() as td:
            m.save(td)
            m2 = Models.load(td)
        assert m2.next_tile.n_observations()["real"] == 1

    def test_summary_runs(self):
        m = Models()
        s = m.summary()
        assert "NextTile" in s


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

class TestStep:
    def test_valid_move_returns_gamestate(self):
        state = parse_state("0 0 0 3  0 0 0 0  0 0 0 0  0 0 0 0 / 3")
        models = Models()
        rng = np.random.default_rng(0)
        result = step(state, 3, models, rng)   # slide left
        assert isinstance(result, GameState)
        # The 3 should have slid to col 0
        assert result.board[0, 0] == 3

    def test_invalid_move_returns_none(self):
        # Board already pushed left — nothing can move left
        state = parse_state("3 6 12 24  6 12 24 48  3 6 12 24  6 12 24 48 / 3")
        models = Models()
        rng = np.random.default_rng(0)
        result = step(state, 3, models, rng)
        assert result is None

    def test_game_over_on_12288(self):
        # Two 6144 tiles adjacent — sliding into each other produces 12288
        state = parse_state("6144 6144 0 0  0 0 0 0  0 0 0 0  0 0 0 0 / 3")
        models = Models()
        rng = np.random.default_rng(0)
        result = step(state, 3, models, rng)   # slide left: 6144+6144=12288
        assert result == GAME_OVER

    def test_new_tile_placed_on_board(self):
        state = parse_state("0 0 0 3  0 0 0 0  0 0 0 0  0 0 0 0 / 3")
        models = Models()
        rng = np.random.default_rng(0)
        result = step(state, 3, models, rng)
        assert isinstance(result, GameState)
        # After slide left, a tile must have been placed on col 3
        assert result.board[0, 3] in {1, 2, 3}  # default prior
