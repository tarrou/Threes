"""
Smoke tests for the entry module — tests the helper functions without
requiring interactive input.
"""
import numpy as np
from threes.simulator import parse_state
from threes.models import Models
from threes.entry import _record_observation, _record_start
import io, sys


def _run_with_input(inputs: list[str], fn, *args):
    """Run fn(*args) with stdin replaced by the given lines."""
    joined = "\n".join(inputs) + "\n"
    old_stdin = sys.stdin
    sys.stdin = io.StringIO(joined)
    try:
        result = fn(*args)
    finally:
        sys.stdin = old_stdin
    return result


class TestEntryHelpers:
    def test_record_observation_happy_path(self, capsys):
        models = Models()
        inputs = [
            # BEFORE state
            "0 0 0 3  0 0 0 0  0 0 0 6  0 0 0 0 / 3",
            # action
            "l",
            # AFTER state — 3 slid to col 0, 6 slid to col 0 row 2, tile 3 placed at (0,3)
            "3 0 0 3  0 0 0 0  6 0 0 0  0 0 0 0 / 1",
            # confirm
            "y",
        ]
        recorded = _run_with_input(inputs, _record_observation, models)
        assert recorded is True
        assert models.next_tile.n_observations()["real"] == 1

    def test_record_observation_user_cancels(self, capsys):
        models = Models()
        inputs = [
            "0 0 0 3  0 0 0 0  0 0 0 6  0 0 0 0 / 3",
            "l",
            "3 0 0 3  0 0 0 0  6 0 0 0  0 0 0 0 / 1",
            "n",   # decline to record
        ]
        recorded = _run_with_input(inputs, _record_observation, models)
        assert recorded is False
        assert models.next_tile.n_observations()["real"] == 0

    def test_record_observation_bad_parse(self, capsys):
        models = Models()
        inputs = [
            "not a valid state",   # bad before state
        ]
        recorded = _run_with_input(inputs, _record_observation, models)
        assert recorded is False

    def test_record_start_happy_path(self, capsys):
        models = Models()
        inputs = [
            "1 2 1 0  0 3 0 0  0 0 1 0  2 0 0 0 / 1",
            "y",
        ]
        recorded = _run_with_input(inputs, _record_start, models)
        assert recorded is True
        assert models.start_board.n_observations()["real"] == 1

    def test_record_start_user_cancels(self, capsys):
        models = Models()
        inputs = [
            "1 2 1 0  0 3 0 0  0 0 1 0  2 0 0 0 / 1",
            "n",
        ]
        recorded = _run_with_input(inputs, _record_start, models)
        assert recorded is False
