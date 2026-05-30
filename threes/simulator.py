"""
Threes game simulator.

Board is a 4x4 numpy int array, row-major (row 0 = top).
Tile values: 0 (empty), 1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144.

Actions: 0=up, 1=right, 2=down, 3=left
Trailing edges (where the new tile spawns):
    up    -> row 3 (bottom)
    right -> col 0 (left)
    down  -> row 0 (top)
    left  -> col 3 (right)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE_VALUES = [0, 1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144]
TILE_SET = set(TILE_VALUES)
_LOG_MAX = np.log2(6144 + 1)   # normalisation denominator for net encoding

ACTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def can_merge(a: int, b: int) -> bool:
    """True if tile a can merge with tile b."""
    if a == 0 or b == 0:
        return False
    if (a == 1 and b == 2) or (a == 2 and b == 1):
        return True
    return a == b and a >= 3


def merge_value(a: int, b: int) -> int:
    """Return the value produced by merging a and b (assumes can_merge is True)."""
    if (a == 1 and b == 2) or (a == 2 and b == 1):
        return 3
    return a + b


# ---------------------------------------------------------------------------
# Single-row slide (leftward)
# ---------------------------------------------------------------------------

def _slide_row_left(row: list[int]) -> tuple[list[int], bool]:
    """
    Slide a single row to the left in-place style.
    Returns (new_row, moved) where moved=True if anything changed.
    Each tile slides as far left as possible; a merge can only be received
    once per cell per move.
    """
    row = list(row)
    merged = [False] * 4   # cell i has already absorbed a merge this move
    moved = False

    for i in range(1, 4):
        if row[i] == 0:
            continue
        left = i - 1
        if row[left] == 0:
            # Empty space: slide one step left
            row[left] = row[i]
            row[i] = 0
            moved = True
        elif can_merge(row[left], row[i]) and not merged[left]:
            # Left neighbour is settled (provably blocked by the sequential
            # left-to-right pass) so a merge is allowed
            row[left] = merge_value(row[left], row[i])
            row[i] = 0
            merged[left] = True
            moved = True
        # else: tiles are incompatible and there's no empty space — stay put

    return row, moved


# ---------------------------------------------------------------------------
# Full-board slide
# ---------------------------------------------------------------------------

def _apply_slide(board: np.ndarray, action: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Apply a slide action to the board.

    Returns:
        new_board   : 4x4 int array after the slide
        eligible    : list of (row, col) positions on the trailing edge that
                      are empty after the slide (valid spawn points)

    If no tile moved the board is returned unchanged and eligible is empty.
    """
    b = board.copy()

    # Normalise every direction to a leftward slide on rows,
    # then un-normalise afterward.
    #
    #   up    -> transpose          -> slide left -> transpose back
    #   right -> flip each row      -> slide left -> flip back
    #   down  -> transpose + flip   -> slide left -> flip + transpose back
    #   left  -> as-is

    if action == 0:    # up
        b = b.T.copy()
    elif action == 1:  # right
        b = np.fliplr(b)
    elif action == 2:  # down
        b = np.fliplr(b.T.copy())
    # action == 3 (left): nothing

    # Slide each row left; after the slide, any empty cell on the trailing
    # edge (col 3 in the transformed frame) is a valid spawn point.
    any_moved = False
    for r in range(4):
        new_row, moved = _slide_row_left(list(b[r]))
        b[r] = new_row
        if moved:
            any_moved = True

    if not any_moved:
        return board.copy(), []

    trailing_empty = [r for r in range(4) if b[r, 3] == 0]

    # Convert trailing_empty row indices back to (row, col) in original frame
    eligible_original = []
    for r in trailing_empty:
        if action == 0:    # up: we transposed, so (r, 3) in transposed = (3, r) in original
            eligible_original.append((3, r))
        elif action == 1:  # right: fliplr, col 3 in flipped = col 0 in original
            eligible_original.append((r, 0))
        elif action == 2:  # down: fliplr(transpose), (r,3) -> flip -> (r,0) -> transpose -> (0,r)
            eligible_original.append((0, r))
        else:              # left: col 3 stays col 3
            eligible_original.append((r, 3))

    # Un-normalise the board
    if action == 0:
        b = b.T.copy()
    elif action == 1:
        b = np.fliplr(b)
    elif action == 2:
        b = (np.fliplr(b)).T.copy()

    return b, eligible_original


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NextTile:
    """
    The tile shown to the player before a move.
    candidates holds one value (normal tile) or three consecutive values
    (bonus tile hint — the actual tile placed will be one of the three).
    """
    candidates: list[int]

    def is_bonus(self) -> bool:
        return len(self.candidates) == 3

    def __str__(self) -> str:
        return " ".join(str(c) for c in self.candidates)


@dataclass
class GameState:
    """Complete observable game state."""
    board: np.ndarray     # shape (4,4), dtype int
    next_tile: NextTile

    def copy(self) -> GameState:
        return GameState(self.board.copy(), NextTile(list(self.next_tile.candidates)))

    def __str__(self) -> str:
        rows = []
        for r in range(4):
            rows.append("  ".join(f"{v:5}" for v in self.board[r]))
        rows.append(f"next: {self.next_tile}")
        return "\n".join(rows)


@dataclass
class Observation:
    """
    A single recorded transition, used for training probability models.
    is_real=True  -> entered from an actual game (ground truth)
    is_real=False -> generated by the simulator
    """
    state_before: GameState
    action: int
    state_after: GameState
    is_real: bool = False

    def placed_tile(self) -> int | None:
        """
        Infer which tile was placed by diffing the boards after accounting
        for the slide.  Returns None if inference fails.
        """
        slid_board, _ = _apply_slide(self.state_before.board, self.action)
        diff = self.state_after.board - slid_board
        nonzero = [(r, c) for r in range(4) for c in range(4) if diff[r, c] != 0]
        if len(nonzero) == 1:
            r, c = nonzero[0]
            return int(self.state_after.board[r, c])
        return None

    def placed_position(self) -> tuple[int, int] | None:
        """Infer where the new tile was placed."""
        slid_board, _ = _apply_slide(self.state_before.board, self.action)
        diff = self.state_after.board - slid_board
        nonzero = [(r, c) for r in range(4) for c in range(4) if diff[r, c] != 0]
        if len(nonzero) == 1:
            return nonzero[0]
        return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_state(text: str) -> GameState:
    """
    Parse a state string of the form:
        <16 board values> / <next tile candidates>

    Board values are row-major, 0 = empty.
    Next tile is one value (normal) or three (bonus hint).

    Example:
        0 0 0 1 2 3 6 6 3 12 24 12 1 1 0 48 / 3
        0 0 0 1 2 3 6 6 3 12 24 12 1 1 0 48 / 6 12 24
    """
    if "/" not in text:
        raise ValueError("State string must contain '/' separating board from next tile.")

    board_part, tile_part = text.split("/", 1)
    board_vals = list(map(int, board_part.split()))
    tile_vals = list(map(int, tile_part.split()))

    if len(board_vals) != 16:
        raise ValueError(f"Expected 16 board values, got {len(board_vals)}.")
    if len(tile_vals) not in (1, 3):
        raise ValueError(f"Next tile must be 1 or 3 values, got {len(tile_vals)}.")
    for v in board_vals:
        if v not in TILE_SET:
            raise ValueError(f"Invalid tile value: {v}. Valid values: {TILE_VALUES}")
    for v in tile_vals:
        if v not in TILE_SET:
            raise ValueError(f"Invalid next-tile value: {v}.")

    board = np.array(board_vals, dtype=int).reshape(4, 4)
    return GameState(board=board, next_tile=NextTile(candidates=tile_vals))


# ---------------------------------------------------------------------------
# Neural-net encoding
# ---------------------------------------------------------------------------

def encode_for_net(state: GameState) -> np.ndarray:
    """
    Encode a GameState as a flat float32 vector for the neural network.

    Board (16 values): log2(tile + 1) / log2(6145), normalised to [0, 1].
    Next tile candidates (3 values, zero-padded if normal tile):
        same log encoding, 0.0 for absent slots.

    Output shape: (19,)
    """
    board_enc = np.log2(state.board.astype(float) + 1).flatten() / _LOG_MAX

    tile_enc = np.zeros(3, dtype=float)
    for i, v in enumerate(state.next_tile.candidates):
        tile_enc[i] = np.log2(v + 1) / _LOG_MAX

    return np.concatenate([board_enc, tile_enc]).astype(np.float32)
