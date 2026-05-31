"""
Smoke tests for the entry module.
"""
import io, sys
import numpy as np
import pytest

from threes.simulator import parse_state
from threes.models import Models
from threes import entry


@pytest.fixture(autouse=True)
def isolate_data_dir(tmp_path):
    """Point _current_data_dir at tmp_path for every test so no test writes to data/."""
    entry._current_data_dir = tmp_path
    yield
    entry._current_data_dir = entry.DEFAULT_DATA_DIR


def _run_with_input(inputs: list[str], fn, *args):
    joined = "\n".join(inputs) + "\n"
    old_stdin = sys.stdin
    sys.stdin = io.StringIO(joined)
    try:
        result = fn(*args)
    finally:
        sys.stdin = old_stdin
    return result


# ---------------------------------------------------------------------------
# _do_one_observation
# ---------------------------------------------------------------------------

class TestDoOneObservation:
    def _before(self):
        # Simple board: one tile in col 3 of row 0, next tile = 3
        return parse_state("0 0 0 3  0 0 0 0  0 0 0 0  0 0 0 0 / 3")

    def test_happy_path_records(self, capsys):
        models = Models()
        before = self._before()
        inputs = [
            "l",        # action: left
            "0",        # tile placed in row 0 (col 3 is fixed for left)
            "1",        # next tile shown is 1
            "y",        # confirm record
        ]
        recorded, after, _ = _run_with_input(inputs, entry._do_one_observation, models, before)
        assert recorded is True
        assert after is not None
        assert models.placement.n_observations()["real"] == 1
        assert models.next_tile.n_observations()["real"] == 1

    def test_skip_does_not_record(self, capsys):
        models = Models()
        before = self._before()
        inputs = ["l", "0", "1", "n"]   # decline to record
        recorded, after, _ = _run_with_input(inputs, entry._do_one_observation, models, before)
        assert recorded is False
        assert after is not None   # after_state still returned for optional chaining

    def test_bad_action_returns_none(self, capsys):
        models = Models()
        before = self._before()
        inputs = ["x"]   # invalid action
        recorded, after, _ = _run_with_input(inputs, entry._do_one_observation, models, before)
        assert recorded is False
        assert after is None

    def test_bonus_tile_asks_which_candidate(self, capsys):
        models = Models()
        before = parse_state("0 0 0 3  0 0 0 0  0 0 0 0  0 0 0 0 / 6 12 24")
        inputs = [
            "l",    # action
            "0",    # row for placement
            "1",    # choose candidate index 1 (=12)
            "3",    # next tile
            "y",    # confirm
        ]
        recorded, after, _ = _run_with_input(inputs, entry._do_one_observation, models, before)
        assert recorded is True
        # Tile placed should be 12 (candidate index 1)
        assert after.board[0, 3] == 12

    def test_correct_fixed_col_for_right_action(self, capsys):
        models = Models()
        before = parse_state("3 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 / 3")
        inputs = [
            "r",    # right → new tile in col 0, ask which row
            "0",    # row 0
            "1",    # next tile
            "y",
        ]
        recorded, after, _ = _run_with_input(inputs, entry._do_one_observation, models, before)
        assert recorded is True
        assert after.board[0, 0] == 3   # tile placed at (row=0, col=0)

    def test_correct_fixed_row_for_up_action(self, capsys):
        models = Models()
        before = parse_state("0 0 0 0  0 0 0 0  0 0 0 0  3 0 0 0 / 3")
        inputs = [
            "u",    # up → new tile in row 3, ask which col
            "0",    # col 0
            "1",    # next tile
            "y",
        ]
        recorded, after, _ = _run_with_input(inputs, entry._do_one_observation, models, before)
        assert recorded is True
        assert after.board[3, 0] == 3   # tile placed at (row=3, col=0)

    def test_impossible_position_reprompts_then_aborts(self, capsys):
        models = Models()
        # Rows 1 and 3 are fully packed (nothing moves), col3 stays occupied → not eligible
        # Rows 0 and 2 have a tile only in col3 → it slides, col3 frees → eligible
        # Eligible rows after left slide: 0 and 2 only
        before = parse_state("0 0 0 3  3 6 12 24  0 0 0 6  6 12 24 48 / 3")
        inputs = [
            "l",
            "1",    # row 1 — impossible (fully packed, not eligible)
            "q",    # abort
        ]
        recorded, after, _ = _run_with_input(inputs, entry._do_one_observation, models, before)
        assert recorded is False

    def test_single_eligible_auto_places(self, capsys):
        models = Models()
        # Row 0: [0,0,0,3] → slides, col3 free — only eligible row
        # Rows 1-3: fully packed, nothing moves, col3 stays occupied
        before = parse_state("0 0 0 3  3 6 12 24  6 12 24 48  3 6 24 48 / 3")
        inputs = [
            "l",    # only row 0 eligible → auto-placed, no row prompt
            "1",    # next tile
            "y",
        ]
        recorded, after, _ = _run_with_input(inputs, entry._do_one_observation, models, before)
        assert recorded is True
        assert after.board[0, 3] == 3


# ---------------------------------------------------------------------------
# _record_chain
# ---------------------------------------------------------------------------

class TestRecordChain:
    def test_single_observation(self, capsys, tmp_path):
        entry._current_data_dir = tmp_path
        models = Models()
        inputs = [
            # before state
            "0 0 0 3  0 0 0 0  0 0 0 0  0 0 0 0 / 3",
            "l",   # action
            "0",   # row
            "1",   # next tile
            "y",   # record
            "n",   # don't continue
        ]
        n = _run_with_input(inputs, entry._record_chain, models)
        assert n == 1

    def test_chained_two_observations(self, capsys, tmp_path):
        entry._current_data_dir = tmp_path
        models = Models()
        inputs = [
            # before state
            "0 0 0 3  0 0 0 0  0 0 0 0  0 0 0 0 / 3",
            "l",   # action 1
            "0",   # row
            "2",   # next tile after move 1
            "y",   # record
            "y",   # continue
            # (board is now shown automatically, no need to re-enter)
            "l",   # action 2
            "1",   # row
            "1",   # next tile after move 2
            "y",   # record
            "n",   # don't continue
        ]
        n = _run_with_input(inputs, entry._record_chain, models)
        assert n == 2
        assert models.next_tile.n_observations()["real"] == 2
