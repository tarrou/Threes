"""
Probability models learned from real and simulated Threes gameplay.

Three models:
  NextTileModel    — P(next tile value | max tile, median tile on board)
  PlacementModel   — P(chosen | merge_happened, gravity, n_tiles_after)
  StartBoardModel  — distribution over starting board configurations

All models store real and simulated counts separately so they can be
blended with a real_weight multiplier at inference time.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from .simulator import (
    GameState, NextTile, Observation,
    TILE_VALUES, TILE_SET,
    _apply_slide,
)

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

TILE_INDEX: dict[int, int] = {v: i for i, v in enumerate(TILE_VALUES)}
N_TILES = len(TILE_VALUES)   # 15

REAL = 1
SIM  = 0

# Gravity categories
GRAVITY_LOW  = 0
GRAVITY_MED  = 1
GRAVITY_HIGH = 2
GRAVITY_NAMES = {GRAVITY_LOW: "low", GRAVITY_MED: "med", GRAVITY_HIGH: "high"}


def _max_tile_idx(board: np.ndarray) -> int:
    return TILE_INDEX[int(board.max())]


def _median_tile_idx(board: np.ndarray) -> int:
    """Index into TILE_VALUES for the lower-median of non-zero tiles."""
    nonzero = sorted(int(v) for v in board.flatten() if v > 0)
    if not nonzero:
        return 0
    median_val = nonzero[(len(nonzero) - 1) // 2]
    return TILE_INDEX.get(median_val, 0)


def _neighbors(r: int, c: int) -> list[tuple[int, int]]:
    """Return all board-adjacent (row, col) pairs."""
    return [(r + dr, c + dc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= r + dr < 4 and 0 <= c + dc < 4]


def _gravity_categories(board_after_slide: np.ndarray,
                         eligible: list[tuple[int, int]]
                         ) -> dict[tuple[int, int], int]:
    """
    Compute LOW/MED/HIGH gravity category for each eligible position.

    Step 1: normalised gravity = (occupied neighbors) / (total possible neighbors)
    Step 2: rank-normalise across unique values → lowest=LOW, highest=HIGH,
            rest=MED.  Single unique value → all MED.
    """
    if not eligible:
        return {}

    def norm_grav(pos: tuple[int, int]) -> float:
        nbrs = _neighbors(*pos)
        occupied = sum(1 for n in nbrs if board_after_slide[n] != 0)
        return occupied / len(nbrs)

    ng = {p: norm_grav(p) for p in eligible}
    unique = sorted(set(ng.values()))

    if len(unique) == 1:
        return {p: GRAVITY_MED for p in eligible}
    elif len(unique) == 2:
        lo, hi = unique
        return {p: GRAVITY_LOW if ng[p] == lo else GRAVITY_HIGH for p in eligible}
    else:
        lo, hi = unique[0], unique[-1]
        return {p: GRAVITY_LOW  if ng[p] == lo
                   else GRAVITY_HIGH if ng[p] == hi
                   else GRAVITY_MED
                for p in eligible}


def _n_tiles_after(board_after_slide: np.ndarray,
                   action: int,
                   pos: tuple[int, int]) -> int:
    """
    Count non-zero tiles in the row (left/right) or column (up/down) of pos,
    excluding the trailing-edge cell itself.
    """
    r, c = pos
    if action in (1, 3):   # left/right: row r, all cols except c
        return sum(1 for col in range(4) if col != c
                   and board_after_slide[r, col] != 0)
    else:                  # up/down: col c, all rows except r
        return sum(1 for row in range(4) if row != r
                   and board_after_slide[row, c] != 0)


def _had_merge(pos: tuple[int, int], action: int,
               merge_rows_transformed: frozenset[int]) -> bool:
    """
    Map an eligible position back to the transformed frame to check whether
    its row/col had a merge during the slide.

    Transformed-frame row → original frame:
      left  (3): row r → original row r
      right (1): row r → original row r  (fliplr preserves rows)
      up    (0): row r → original col r  (transpose)
      down  (2): row r → original col (3-r)  (fliplr∘transpose)
    """
    r, c = pos
    if action == 3:   # left:  eligible pos (r, 3)
        return r in merge_rows_transformed
    elif action == 1: # right: eligible pos (r, 0)
        return r in merge_rows_transformed
    elif action == 0: # up:    eligible pos (3, c)
        return c in merge_rows_transformed
    else:             # down:  eligible pos (0, c)
        return (3 - c) in merge_rows_transformed


# ---------------------------------------------------------------------------
# NextTileModel
# ---------------------------------------------------------------------------

class NextTileModel:
    """
    Learns what tile value the game will show next.

    Conditioning: (max_tile_idx, median_tile_idx) — both bucketed into
    TILE_VALUES indices.  Captures the bonus-tile unlock mechanic (max) and
    overall board density (median).

    When a bonus hint is observed (3 candidates), each candidate receives
    equal fractional credit (1/3).

    Storage: counts[source, max_tile_idx, median_tile_idx, tile_value_idx]
    Shape: (2, 15, 15, 15)
    """

    def __init__(self) -> None:
        self._counts = np.zeros((2, N_TILES, N_TILES, N_TILES), dtype=np.float64)

    # ------------------------------------------------------------------
    # Learning

    def update(self, obs: Observation) -> None:
        nt      = obs.state_after.next_tile
        max_idx = _max_tile_idx(obs.state_after.board)
        med_idx = _median_tile_idx(obs.state_after.board)
        src     = REAL if obs.is_real else SIM
        weight  = 1.0 / len(nt.candidates)
        for v in nt.candidates:
            self._counts[src, max_idx, med_idx, TILE_INDEX[v]] += weight

    # ------------------------------------------------------------------
    # Inference

    def pmf(self, board: np.ndarray, real_weight: float = 10.0) -> np.ndarray:
        """
        PMF over TILE_VALUES.  Fallback chain:
          1. exact (max, median) bucket
          2. max-tile bucket only (sum over median)
          3. global (sum over all buckets)
          4. uniform prior over {1, 2, 3}
        """
        max_idx = _max_tile_idx(board)
        med_idx = _median_tile_idx(board)

        counts = (self._counts[REAL, max_idx, med_idx] * real_weight
                  + self._counts[SIM,  max_idx, med_idx])
        if counts.sum() == 0:
            counts = (self._counts[REAL, max_idx].sum(axis=0) * real_weight
                      + self._counts[SIM,  max_idx].sum(axis=0))
        if counts.sum() == 0:
            counts = (self._counts[REAL].sum(axis=(0, 1)) * real_weight
                      + self._counts[SIM].sum(axis=(0, 1)))
        if counts.sum() == 0:
            p = np.zeros(N_TILES)
            for v in (1, 2, 3):
                p[TILE_INDEX[v]] = 1.0 / 3.0
            return p

        return counts / counts.sum()

    def sample(self, board: np.ndarray, rng: np.random.Generator,
               real_weight: float = 10.0) -> int:
        p = self.pmf(board, real_weight)
        return TILE_VALUES[int(rng.choice(N_TILES, p=p))]

    # ------------------------------------------------------------------
    # Stats

    def n_observations(self) -> dict[str, int]:
        return {"real": int(self._counts[REAL].sum()),
                "simulated": int(self._counts[SIM].sum())}

    # ------------------------------------------------------------------
    # Persistence

    def save(self, path: str | Path) -> None:
        np.savez(path, counts=self._counts)

    @classmethod
    def load(cls, path: str | Path) -> NextTileModel:
        m = cls()
        data = np.load(path)
        m._counts = data["counts"]
        return m


# ---------------------------------------------------------------------------
# PlacementModel
# ---------------------------------------------------------------------------

class PlacementModel:
    """
    Learns which eligible position the game places the new tile in.

    Per-position features:
      merge_happened : bool  — did this row/col produce a merge this move?
      gravity        : LOW/MED/HIGH — normalised + rank-normalised tile density
                       in the neighbourhood of this position
      n_tiles_after  : 0–3  — non-zero tiles in this row/col after the slide
                       (excluding the trailing-edge cell itself)

    Storage:
      _chosen[src, merge, gravity, n_tiles]  — times this feature cell was chosen
      _total [src, merge, gravity, n_tiles]  — times this feature cell was eligible
    Both shape (2, 2, 3, 4).

    Sampling: P(position chosen) ∝ _chosen / _total for each eligible position's
    feature cell; uniform fallback when a cell has no data.
    """

    def __init__(self) -> None:
        self._chosen = np.zeros((2, 2, 3, 4), dtype=np.float64)
        self._total  = np.zeros((2, 2, 3, 4), dtype=np.float64)
        # Cross-observation counters: when ≥1 eligible position had a merge,
        # how often was the chosen position one that had a merge?
        # Shape (2,): [SIM, REAL]
        self._merge_avail_chose_merge    = np.zeros(2, dtype=np.float64)
        self._merge_avail_chose_no_merge = np.zeros(2, dtype=np.float64)
        # Mixed-case counters: when BOTH merge and non-merge positions were
        # eligible simultaneously, which type was chosen?
        self._mixed_chose_merge    = np.zeros(2, dtype=np.float64)
        self._mixed_chose_no_merge = np.zeros(2, dtype=np.float64)

    # ------------------------------------------------------------------
    # Learning

    def update(self, obs: Observation) -> None:
        new_board, eligible, merge_rows = _apply_slide(
            obs.state_before.board, obs.action)
        if not eligible:
            return
        placed = obs.placed_position()
        if placed is None:
            return

        merge_flags = {pos: _had_merge(pos, obs.action, merge_rows)
                       for pos in eligible}
        gravity_cats = _gravity_categories(new_board, eligible)
        src = REAL if obs.is_real else SIM

        for pos in eligible:
            merge   = int(merge_flags[pos])
            gravity = gravity_cats[pos]
            n_tiles = min(_n_tiles_after(new_board, obs.action, pos), 3)
            self._total[src, merge, gravity, n_tiles] += 1.0
            if pos == placed:
                self._chosen[src, merge, gravity, n_tiles] += 1.0

        # Cross-observation merge-preference counters
        any_merge    = any(merge_flags.values())
        any_no_merge = not all(merge_flags.values())

        if any_merge:
            if merge_flags[placed]:
                self._merge_avail_chose_merge[src]    += 1.0
            else:
                self._merge_avail_chose_no_merge[src] += 1.0

        # Mixed case: at least one merge AND at least one non-merge eligible
        if any_merge and any_no_merge:
            if merge_flags[placed]:
                self._mixed_chose_merge[src]    += 1.0
            else:
                self._mixed_chose_no_merge[src] += 1.0

    # ------------------------------------------------------------------
    # Inference

    def score(self, pos: tuple[int, int], action: int,
              board_after_slide: np.ndarray,
              eligible: list[tuple[int, int]],
              merge_rows_transformed: frozenset[int],
              gravity_cats: dict[tuple[int, int], int],
              real_weight: float = 10.0) -> float:
        """P(chosen | features) for a single position.  Returns 1.0 if no data."""
        merge   = int(_had_merge(pos, action, merge_rows_transformed))
        gravity = gravity_cats[pos]
        n_tiles = min(_n_tiles_after(board_after_slide, action, pos), 3)
        chosen = (self._chosen[REAL, merge, gravity, n_tiles] * real_weight
                  + self._chosen[SIM,  merge, gravity, n_tiles])
        total  = (self._total[REAL,  merge, gravity, n_tiles] * real_weight
                  + self._total[SIM,   merge, gravity, n_tiles])
        return float(chosen / total) if total > 0 else 1.0

    def sample(self, action: int,
               board_after_slide: np.ndarray,
               eligible: list[tuple[int, int]],
               merge_rows_transformed: frozenset[int],
               rng: np.random.Generator,
               real_weight: float = 10.0) -> tuple[int, int]:
        gravity_cats = _gravity_categories(board_after_slide, eligible)
        scores = np.array([
            self.score(p, action, board_after_slide, eligible,
                       merge_rows_transformed, gravity_cats, real_weight)
            for p in eligible
        ])
        total = scores.sum()
        p = scores / total if total > 0 else np.ones(len(eligible)) / len(eligible)
        return eligible[int(rng.choice(len(eligible), p=p))]

    # ------------------------------------------------------------------
    # Stats

    def n_observations(self) -> dict[str, int]:
        return {"real": int(self._chosen[REAL].sum()),
                "simulated": int(self._chosen[SIM].sum())}

    def merge_preference(self, real_weight: float = 10.0
                         ) -> dict[str, object]:
        """
        Two merge-preference statistics:
        - overall: when ≥1 eligible position had a merge, how often was the
          chosen position one with a merge?
        - mixed: same, but only observations where BOTH merge and non-merge
          positions were eligible (isolates the true preference signal).
        """
        def _stat(chose_m, chose_n):
            cm = chose_m[REAL] * real_weight + chose_m[SIM]
            cn = chose_n[REAL] * real_weight + chose_n[SIM]
            total = cm + cn
            return {
                "chose_merge":    int(cm),
                "chose_no_merge": int(cn),
                "rate": float(cm / total) if total > 0 else None,
            }

        return {
            "overall": _stat(self._merge_avail_chose_merge,
                             self._merge_avail_chose_no_merge),
            "mixed":   _stat(self._mixed_chose_merge,
                             self._mixed_chose_no_merge),
        }

    def table(self, real_weight: float = 10.0
              ) -> np.ndarray:
        """
        Return P(chosen | features) as a (2, 3, 4) array:
        (merge, gravity, n_tiles).  NaN where no data.
        """
        chosen = (self._chosen[REAL] * real_weight + self._chosen[SIM])
        total  = (self._total[REAL]  * real_weight + self._total[SIM])
        with np.errstate(invalid="ignore"):
            p = np.where(total > 0, chosen / total, np.nan)
        return p   # shape (2, 2, 3, 4) sliced to [src summed] → (2, 3, 4)

    # ------------------------------------------------------------------
    # Persistence

    def save(self, path: str | Path) -> None:
        np.savez(path,
                 chosen=self._chosen,
                 total=self._total,
                 merge_avail_chose_merge=self._merge_avail_chose_merge,
                 merge_avail_chose_no_merge=self._merge_avail_chose_no_merge,
                 mixed_chose_merge=self._mixed_chose_merge,
                 mixed_chose_no_merge=self._mixed_chose_no_merge)

    @classmethod
    def load(cls, path: str | Path) -> PlacementModel:
        m = cls()
        data = np.load(path)
        m._chosen = data["chosen"]
        m._total  = data["total"]
        if "merge_avail_chose_merge" in data:
            m._merge_avail_chose_merge    = data["merge_avail_chose_merge"]
            m._merge_avail_chose_no_merge = data["merge_avail_chose_no_merge"]
        if "mixed_chose_merge" in data:
            m._mixed_chose_merge    = data["mixed_chose_merge"]
            m._mixed_chose_no_merge = data["mixed_chose_no_merge"]
        return m


# ---------------------------------------------------------------------------
# StartBoardModel  (unchanged)
# ---------------------------------------------------------------------------

class StartBoardModel:
    """
    Distribution over starting board configurations.
    Stores observed examples and samples from them directly.
    """

    def __init__(self) -> None:
        self._boards:     list[np.ndarray] = []
        self._next_tiles: list[list[int]]  = []
        self._is_real:    list[bool]       = []

    def update(self, state: GameState, is_real: bool) -> None:
        self._boards.append(state.board.copy())
        self._next_tiles.append(list(state.next_tile.candidates))
        self._is_real.append(is_real)

    def sample(self, rng: np.random.Generator,
               real_weight: float = 10.0) -> GameState | None:
        if not self._boards:
            return None
        weights = np.array([real_weight if r else 1.0 for r in self._is_real],
                           dtype=np.float64)
        weights /= weights.sum()
        idx = int(rng.choice(len(self._boards), p=weights))
        return GameState(board=self._boards[idx].copy(),
                         next_tile=NextTile(list(self._next_tiles[idx])))

    def n_observations(self) -> dict[str, int]:
        real = sum(self._is_real)
        return {"real": real, "simulated": len(self._boards) - real}

    def save(self, path: str | Path) -> None:
        boards = np.stack(self._boards) if self._boards else np.empty((0, 4, 4), dtype=int)
        nt_padded  = np.zeros((len(self._next_tiles), 3), dtype=np.int32)
        nt_lengths = np.zeros(len(self._next_tiles),      dtype=np.int32)
        for i, cands in enumerate(self._next_tiles):
            nt_lengths[i] = len(cands)
            nt_padded[i, :len(cands)] = cands
        np.savez(path, boards=boards, nt_padded=nt_padded,
                 nt_lengths=nt_lengths,
                 is_real=np.array(self._is_real, dtype=bool))

    @classmethod
    def load(cls, path: str | Path) -> StartBoardModel:
        m = cls()
        data = np.load(path)
        for i in range(len(data["boards"])):
            m._boards.append(data["boards"][i].copy())
            n = int(data["nt_lengths"][i])
            m._next_tiles.append(list(map(int, data["nt_padded"][i, :n])))
            m._is_real.append(bool(data["is_real"][i]))
        return m


# ---------------------------------------------------------------------------
# Models bundle
# ---------------------------------------------------------------------------

@dataclass
class Models:
    next_tile:   NextTileModel   = field(default_factory=NextTileModel)
    placement:   PlacementModel  = field(default_factory=PlacementModel)
    start_board: StartBoardModel = field(default_factory=StartBoardModel)

    def update(self, obs: Observation) -> None:
        self.next_tile.update(obs)
        self.placement.update(obs)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.next_tile.save(d / "next_tile.npz")
        self.placement.save(d / "placement.npz")
        self.start_board.save(d / "start_board.npz")

    @classmethod
    def load(cls, directory: str | Path) -> Models:
        d = Path(directory)
        return cls(
            next_tile=NextTileModel.load(d / "next_tile.npz"),
            placement=PlacementModel.load(d / "placement.npz"),
            start_board=StartBoardModel.load(d / "start_board.npz"),
        )

    def summary(self) -> str:
        nt = self.next_tile.n_observations()
        pl = self.placement.n_observations()
        sb = self.start_board.n_observations()
        return (
            f"NextTile   — real: {nt['real']:4d}  simulated: {nt['simulated']:6d}\n"
            f"Placement  — real: {pl['real']:4d}  simulated: {pl['simulated']:6d}\n"
            f"StartBoard — real: {sb['real']:4d}  simulated: {sb['simulated']:6d}"
        )


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

GAME_OVER = "game_over"


def step(state: GameState, action: int, models: Models,
         rng: np.random.Generator,
         real_weight: float = 10.0) -> GameState | str | None:
    """
    Advance the game by one move.

    Returns GameState, "game_over", or None (invalid move).
    """
    new_board, eligible, merge_rows = _apply_slide(state.board, action)

    if 12288 in new_board:
        return GAME_OVER

    if not eligible:
        return None

    if len(state.next_tile.candidates) == 1:
        tile_to_place = state.next_tile.candidates[0]
    else:
        tile_to_place = int(rng.choice(state.next_tile.candidates))

    pos = models.placement.sample(action, new_board, eligible,
                                  merge_rows, rng, real_weight)
    new_board[pos] = tile_to_place

    new_next_value = models.next_tile.sample(new_board, rng, real_weight)
    return GameState(new_board, NextTile([new_next_value]))
