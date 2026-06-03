import numpy as np
import pytest
from threes.simulator import parse_state, score_board, GameState, NextTile
from threes.models import Models
from threes.game import Game, ROLLOUT_N


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# score_board
# ---------------------------------------------------------------------------

class TestScoreBoard:
    def _board(self, values):
        return np.array(values, dtype=int).reshape(4, 4)

    def test_empty_board(self):
        assert score_board(self._board([0]*16)) == 0.0

    def test_ones_and_twos_score_zero(self):
        b = self._board([1, 2, 1, 2] + [0]*12)
        assert score_board(b) == 0.0

    def test_single_3(self):
        b = self._board([3] + [0]*15)
        assert score_board(b) == pytest.approx(3.0)

    def test_single_6(self):
        b = self._board([6] + [0]*15)
        assert score_board(b) == pytest.approx(9.0)

    def test_single_12(self):
        b = self._board([12] + [0]*15)
        assert score_board(b) == pytest.approx(27.0)

    def test_additive(self):
        b = self._board([3, 6, 0, 0] + [0]*12)
        assert score_board(b) == pytest.approx(3.0 + 9.0)


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

def _make_game(state: GameState | None = None) -> Game:
    models = Models()
    rng    = np.random.default_rng(0)
    game   = Game(models, rng)
    if state is not None:
        game.state = state
        game.done  = False
    return game


class TestGame:
    def test_reset_returns_gamestate(self):
        game = _make_game()
        state = game.reset()
        assert isinstance(state, GameState)
        assert (state.board > 0).sum() == 9
        assert not game.done

    def test_valid_actions_has_some_true(self):
        game = _make_game()
        mask = game.valid_actions()
        assert mask.shape == (4,)
        assert mask.any()

    def test_valid_actions_all_false_when_stuck(self):
        # Fully packed board with no mergeable neighbours
        stuck = parse_state(
            "3  6 12 24  6 12 24 48  3  6 12 24  6 12 24 48 / 3")
        game = _make_game(stuck)
        assert not game.valid_actions().any()

    def test_act_valid_move(self):
        state = parse_state("0 0 0 3  0 0 0 0  0 0 0 0  0 0 0 0 / 3")
        game  = _make_game(state)
        new_state, done = game.act(3)  # slide left
        assert isinstance(new_state, GameState)
        assert game.state is new_state

    def test_act_invalid_noop(self):
        # All moves blocked except left/right — pick up on a full board
        stuck = parse_state(
            "3  6 12 24  6 12 24 48  3  6 12 24  6 12 24 48 / 3")
        game = _make_game(stuck)
        original_board = game.state.board.copy()
        new_state, done = game.act(0)  # up — invalid
        assert np.array_equal(game.state.board, original_board)
        assert not done

    def test_act_game_over(self):
        # Two 6144 tiles: merge → 12288 → game over
        go_state = parse_state(
            "6144 6144 0 0  0 0 0 0  0 0 0 0  0 0 0 0 / 3")
        game = _make_game(go_state)
        _, done = game.act(3)  # slide left
        assert done
        assert game.done

    def test_rollout_returns_float(self):
        state = parse_state("0 0 0 3  0 0 0 0  0 0 0 0  0 0 0 0 / 3")
        game  = _make_game(state)
        score = game.rollout(n=4)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_rollout_restores_state(self):
        state = parse_state("0 0 0 3  0 0 0 0  0 0 0 0  0 0 0 0 / 3")
        game  = _make_game(state)
        board_before = game.state.board.copy()
        game.rollout(n=8)
        assert np.array_equal(game.state.board, board_before)

    def test_rollout_n_default(self):
        assert ROLLOUT_N == 8
