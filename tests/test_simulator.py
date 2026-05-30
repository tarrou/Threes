import numpy as np
import pytest
from threes.simulator import (
    can_merge, merge_value,
    _slide_row_left, _apply_slide,
    GameState, NextTile, Observation,
    parse_state, encode_for_net,
    TILE_VALUES,
)


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

class TestMerge:
    def test_1_plus_2(self):
        assert can_merge(1, 2) and merge_value(1, 2) == 3
        assert can_merge(2, 1) and merge_value(2, 1) == 3

    def test_matching_ge3(self):
        assert can_merge(3, 3) and merge_value(3, 3) == 6
        assert can_merge(48, 48) and merge_value(48, 48) == 96

    def test_no_merge(self):
        assert not can_merge(1, 1)
        assert not can_merge(2, 2)
        assert not can_merge(1, 3)
        assert not can_merge(3, 6)
        assert not can_merge(0, 3)


# ---------------------------------------------------------------------------
# Row slide
# ---------------------------------------------------------------------------

class TestSlideRow:
    def test_slide_one_step_into_empty(self):
        # 3 slides one step; 6 slides one step (not all the way past the gap)
        row, moved = _slide_row_left([0, 3, 0, 6])
        assert row == [3, 0, 6, 0]
        assert moved

    def test_slide_through_vacated_space(self):
        # 3 slides to pos0; 6 slides into the space 3 vacated
        row, moved = _slide_row_left([0, 3, 6, 0])
        assert row == [3, 6, 0, 0]
        assert moved

    def test_merge_equal_at_wall(self):
        # Left 3 is at the wall — right 3 merges into it
        row, moved = _slide_row_left([3, 3, 0, 0])
        assert row == [6, 0, 0, 0]
        assert moved

    def test_merge_1_2_at_wall(self):
        row, moved = _slide_row_left([1, 2, 0, 0])
        assert row == [3, 0, 0, 0]
        assert moved

    def test_no_merge_when_target_can_move(self):
        # Left 3 has empty space ahead — right 3 cannot merge, both just slide
        row, moved = _slide_row_left([0, 3, 3, 0])
        assert row == [3, 3, 0, 0]
        assert moved

    def test_no_merge_across_gap(self):
        # Two 3s separated by a gap — each moves one step, no merge
        row, moved = _slide_row_left([0, 0, 3, 3])
        assert row == [0, 3, 3, 0]
        assert moved

    def test_merge_when_blocked_by_incompatible_tile(self):
        # 1 can't pass the 3, so 2 behind can merge into 1
        row, moved = _slide_row_left([3, 1, 2, 0])
        assert row == [3, 3, 0, 0]
        assert moved

    def test_chain_1_plus_2_then_slide(self):
        # 1+2=3 at wall, then 3 and 6 slide one step each
        row, moved = _slide_row_left([1, 2, 3, 6])
        assert row == [3, 3, 6, 0]
        assert moved

    def test_no_move(self):
        row, moved = _slide_row_left([3, 6, 12, 24])
        assert row == [3, 6, 12, 24]
        assert not moved

    def test_already_leftmost(self):
        row, moved = _slide_row_left([6, 0, 0, 0])
        assert row == [6, 0, 0, 0]
        assert not moved


# ---------------------------------------------------------------------------
# Board slide + eligible positions
# ---------------------------------------------------------------------------

class TestApplySlide:
    def _board(self, values):
        return np.array(values, dtype=int).reshape(4, 4)

    def test_slide_left_eligible(self):
        # Row 0: [3, 0, 0, 6] -> [3, 6, 0, 0]; col-3 now empty -> eligible (0,3)
        # Row 1: [0, 0, 0, 0] -> no change; no move so not counted
        # Row 2: [3, 6, 12, 0] -> no change (already blocked); col-3 was already 0
        # Row 3: [6, 6, 0, 0] -> [12, 0, 0, 0]; col-3 already 0 -> eligible (3,3)
        b = self._board([
            3, 0, 0, 6,
            0, 0, 0, 0,
            3, 6, 12, 0,
            6, 6, 0, 0,
        ])
        new_b, eligible = _apply_slide(b, 3)  # left
        assert (0, 3) in eligible   # freed by slide
        assert (2, 3) in eligible   # col-3 was already empty; board moved so it's eligible
        assert (3, 3) in eligible   # was already empty, still eligible

    def test_slide_right_eligible(self):
        # Trailing edge for right is col 0
        b = self._board([
            3, 0, 0, 0,
            0, 0, 0, 0,
            3, 6, 0, 0,
            0, 0, 0, 0,
        ])
        new_b, eligible = _apply_slide(b, 1)  # right
        # Row 0: [3,0,0,0] slides right -> [0,0,0,3]; col-0 had 3, now 0 -> eligible (0,0)
        assert (0, 0) in eligible
        # Row 2: [3,6,0,0] -> [0,0,3,6]; col-0 had 3, now 0 -> eligible (2,0)
        assert (2, 0) in eligible

    def test_no_move_returns_empty_eligible(self):
        # All rows fully packed with no mergeable neighbours -> nothing moves left
        b = self._board([
            3,  6, 12, 24,
            6, 12, 24, 48,
            3,  6, 12, 24,
            6, 12, 24, 48,
        ])
        new_b, eligible = _apply_slide(b, 3)  # left, nothing can move
        assert eligible == []
        assert np.array_equal(new_b, b)

    def test_slide_up_eligible(self):
        # Trailing edge for up is row 3
        b = self._board([
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            3, 6, 0, 0,
        ])
        new_b, eligible = _apply_slide(b, 0)  # up
        # Col 0: [0,0,0,3] -> [3,0,0,0]; row-3 had 3, now 0 -> eligible (3,0)
        assert (3, 0) in eligible
        # Col 1: [0,0,0,6] -> [6,0,0,0]; row-3 had 6, now 0 -> eligible (3,1)
        assert (3, 1) in eligible

    def test_slide_down_eligible(self):
        # Trailing edge for down is row 0
        b = self._board([
            3, 6, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        ])
        new_b, eligible = _apply_slide(b, 2)  # down
        assert (0, 0) in eligible
        assert (0, 1) in eligible


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParse:
    def test_normal_tile(self):
        s = parse_state("0 0 0 1  2 3 6 6  3 12 24 12  1 1 0 48 / 3")
        assert s.board[0, 3] == 1
        assert s.board[3, 3] == 48
        assert s.next_tile.candidates == [3]
        assert not s.next_tile.is_bonus()

    def test_bonus_tile(self):
        s = parse_state("0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 / 6 12 24")
        assert s.next_tile.candidates == [6, 12, 24]
        assert s.next_tile.is_bonus()

    def test_bad_count(self):
        with pytest.raises(ValueError):
            parse_state("0 0 0 / 3")

    def test_bad_tile_value(self):
        with pytest.raises(ValueError):
            parse_state("0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 5 / 3")

    def test_missing_slash(self):
        with pytest.raises(ValueError):
            parse_state("0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  3")


# ---------------------------------------------------------------------------
# Net encoding
# ---------------------------------------------------------------------------

class TestEncoding:
    def test_shape(self):
        s = parse_state("0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 / 3")
        enc = encode_for_net(s)
        assert enc.shape == (19,)
        assert enc.dtype == np.float32

    def test_zero_tiles_encode_zero(self):
        s = parse_state("0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 / 3")
        enc = encode_for_net(s)
        assert np.all(enc[:16] == 0.0)

    def test_normal_tile_fills_first_slot_only(self):
        s = parse_state("0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 / 3")
        enc = encode_for_net(s)
        assert enc[16] > 0.0   # tile = 3
        assert enc[17] == 0.0  # absent
        assert enc[18] == 0.0  # absent

    def test_bonus_tile_fills_three_slots(self):
        s = parse_state("0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 / 6 12 24")
        enc = encode_for_net(s)
        assert enc[16] > 0.0
        assert enc[17] > 0.0
        assert enc[18] > 0.0
        # Values should be increasing (6 < 12 < 24)
        assert enc[16] < enc[17] < enc[18]

    def test_max_tile_encodes_to_one(self):
        s = parse_state("6144 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 / 3")
        enc = encode_for_net(s)
        assert abs(enc[0] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Observation inference
# ---------------------------------------------------------------------------

class TestObservation:
    def test_placed_tile_and_position(self):
        before = parse_state("0 0 0 3  0 0 0 6  0 0 0 12  0 0 0 24 / 3")
        # One-step left slide: each tile in col 3 moves one step to col 2.
        # New tile (3) placed at (0,3).
        after_board = np.array([
            0, 0, 3, 3,   # 3 slid to col 2, new 3 placed at (0,3)
            0, 0, 6, 0,
            0, 0, 12, 0,
            0, 0, 24, 0,
        ], dtype=int).reshape(4, 4)
        after = GameState(after_board, NextTile([1]))  # next tile doesn't matter here
        obs = Observation(before, 3, after, is_real=True)
        assert obs.placed_tile() == 3
        assert obs.placed_position() == (0, 3)
