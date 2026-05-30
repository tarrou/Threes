"""
Probability models learned from real and simulated Threes gameplay.

Three models:
  NextTileModel    — P(next tile value | max tile on board)
  PlacementModel   — P(placement slot | action, n_eligible_slots)
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

REAL  = 1
SIM   = 0


def _max_tile_idx(board: np.ndarray) -> int:
    """Index into TILE_VALUES for the largest tile on the board."""
    return TILE_INDEX[int(board.max())]


def _eligible_sorted(board: np.ndarray, action: int) -> list[tuple[int, int]]:
    _, elig = _apply_slide(board, action)
    return sorted(elig)


# ---------------------------------------------------------------------------
# NextTileModel
# ---------------------------------------------------------------------------

class NextTileModel:
    """
    Learns what tile value the game will show next.

    Conditioning variable: index of the board's current max tile.
    This captures the main mechanic — bonus tiles (value > 3) only appear
    once the board has high enough tiles.

    When a bonus hint is observed (3 candidates), each candidate receives
    equal fractional credit (1/3) since we don't yet know which will land.

    Storage: counts[source, max_tile_idx, tile_value_idx]
      source 0 = simulated, source 1 = real
    """

    def __init__(self) -> None:
        self._counts = np.zeros((2, N_TILES, N_TILES), dtype=np.float64)

    # ------------------------------------------------------------------
    # Learning

    def update(self, obs: Observation) -> None:
        """Record the next tile shown in obs.state_after."""
        nt = obs.state_after.next_tile
        max_idx = _max_tile_idx(obs.state_after.board)
        src = REAL if obs.is_real else SIM
        weight = 1.0 / len(nt.candidates)
        for v in nt.candidates:
            self._counts[src, max_idx, TILE_INDEX[v]] += weight

    # ------------------------------------------------------------------
    # Inference

    def pmf(self, board: np.ndarray, real_weight: float = 10.0) -> np.ndarray:
        """
        Probability mass function over TILE_VALUES given the current board.

        real_weight: multiplier applied to real-observation counts vs simulated.
        Falls back to global (all max-tile buckets) if this bucket is empty,
        then to a uniform prior over {1, 2, 3} if no data exists at all.
        """
        max_idx = _max_tile_idx(board)
        counts = (self._counts[REAL, max_idx] * real_weight
                  + self._counts[SIM,  max_idx])
        total = counts.sum()

        if total == 0:
            # Fall back to global distribution
            counts = (self._counts[REAL].sum(axis=0) * real_weight
                      + self._counts[SIM].sum(axis=0))
            total = counts.sum()

        if total == 0:
            # No data at all — prior: uniform over 1, 2, 3
            p = np.zeros(N_TILES)
            for v in (1, 2, 3):
                p[TILE_INDEX[v]] = 1.0 / 3.0
            return p

        return counts / total

    def sample(self, board: np.ndarray, rng: np.random.Generator,
               real_weight: float = 10.0) -> int:
        """Sample a next tile value given the current board."""
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
    Learns which eligible slot the new tile is placed in after a swipe.

    Conditioning: (action, n_eligible).
    Eligible slots are sorted by (row, col) and we track which sorted index
    was chosen.  This is direction-agnostic about *which* cells are eligible,
    capturing only positional bias within the available slots.

    Storage: dict keyed by (source, action, n_eligible) -> count array of
    length n_eligible.
    """

    def __init__(self) -> None:
        # (source, action, n_eligible) -> np.ndarray of length n_eligible
        self._counts: dict[tuple[int, int, int], np.ndarray] = {}

    def _get(self, src: int, action: int, n: int) -> np.ndarray:
        key = (src, action, n)
        if key not in self._counts:
            self._counts[key] = np.zeros(n, dtype=np.float64)
        return self._counts[key]

    # ------------------------------------------------------------------
    # Learning

    def update(self, obs: Observation) -> None:
        """Record which eligible slot was chosen in this observation."""
        eligible = _eligible_sorted(obs.state_before.board, obs.action)
        if not eligible:
            return
        pos = obs.placed_position()
        if pos is None or pos not in eligible:
            return
        idx = eligible.index(pos)
        src = REAL if obs.is_real else SIM
        self._get(src, obs.action, len(eligible))[idx] += 1.0

    # ------------------------------------------------------------------
    # Inference

    def pmf(self, action: int, eligible: list[tuple[int, int]],
            real_weight: float = 10.0) -> np.ndarray:
        """
        Probability mass function over the sorted eligible positions.
        Falls back to uniform when no data exists for this (action, n) combo.
        """
        n = len(eligible)
        real_arr = self._counts.get((REAL, action, n), np.zeros(n))
        sim_arr  = self._counts.get((SIM,  action, n), np.zeros(n))
        counts = real_arr * real_weight + sim_arr
        total = counts.sum()
        if total == 0:
            return np.ones(n) / n   # uniform prior
        return counts / total

    def sample(self, action: int, eligible: list[tuple[int, int]],
               rng: np.random.Generator, real_weight: float = 10.0
               ) -> tuple[int, int]:
        """Sample a placement position from the eligible slots."""
        eligible_sorted = sorted(eligible)
        p = self.pmf(action, eligible_sorted, real_weight)
        idx = int(rng.choice(len(eligible_sorted), p=p))
        return eligible_sorted[idx]

    # ------------------------------------------------------------------
    # Stats

    def n_observations(self) -> dict[str, int]:
        real  = sum(v.sum() for (s, _, __), v in self._counts.items() if s == REAL)
        sim   = sum(v.sum() for (s, _, __), v in self._counts.items() if s == SIM)
        return {"real": int(real), "simulated": int(sim)}

    # ------------------------------------------------------------------
    # Persistence

    def save(self, path: str | Path) -> None:
        slot_keys = list(self._counts.keys())
        arrays    = [self._counts[k] for k in slot_keys]
        np.savez(path,
                 slot_keys=np.array(slot_keys, dtype=np.int32),
                 **{f"arr_{i}": a for i, a in enumerate(arrays)})

    @classmethod
    def load(cls, path: str | Path) -> PlacementModel:
        m = cls()
        data = np.load(path)
        slot_keys = data["slot_keys"]
        for i, (src, action, n) in enumerate(slot_keys):
            m._counts[(int(src), int(action), int(n))] = data[f"arr_{i}"]
        return m


# ---------------------------------------------------------------------------
# StartBoardModel
# ---------------------------------------------------------------------------

class StartBoardModel:
    """
    Distribution over starting board configurations.

    Because starting boards are high-dimensional we don't try to build a
    generative model — we simply store observed examples and sample from
    them directly, with real examples weighted higher.

    Each stored entry is (board_4x4, next_tile_candidates, is_real).
    """

    def __init__(self) -> None:
        self._boards:      list[np.ndarray]  = []
        self._next_tiles:  list[list[int]]   = []
        self._is_real:     list[bool]        = []

    # ------------------------------------------------------------------
    # Learning

    def update(self, state: GameState, is_real: bool) -> None:
        """Record a starting board example."""
        self._boards.append(state.board.copy())
        self._next_tiles.append(list(state.next_tile.candidates))
        self._is_real.append(is_real)

    # ------------------------------------------------------------------
    # Inference

    def sample(self, rng: np.random.Generator,
               real_weight: float = 10.0) -> GameState | None:
        """
        Sample a starting state.  Returns None if no examples have been recorded.
        """
        if not self._boards:
            return None
        weights = np.array([real_weight if r else 1.0 for r in self._is_real],
                           dtype=np.float64)
        weights /= weights.sum()
        idx = int(rng.choice(len(self._boards), p=weights))
        return GameState(
            board=self._boards[idx].copy(),
            next_tile=NextTile(list(self._next_tiles[idx])),
        )

    # ------------------------------------------------------------------
    # Stats

    def n_observations(self) -> dict[str, int]:
        real = sum(self._is_real)
        return {"real": real, "simulated": len(self._boards) - real}

    # ------------------------------------------------------------------
    # Persistence

    def save(self, path: str | Path) -> None:
        boards = np.stack(self._boards) if self._boards else np.empty((0, 4, 4), dtype=int)
        # Store next_tiles as a ragged structure: pad to length 3, record actual length
        nt_padded  = np.zeros((len(self._next_tiles), 3), dtype=np.int32)
        nt_lengths = np.zeros(len(self._next_tiles),      dtype=np.int32)
        for i, cands in enumerate(self._next_tiles):
            nt_lengths[i] = len(cands)
            nt_padded[i, :len(cands)] = cands
        np.savez(path,
                 boards=boards,
                 nt_padded=nt_padded,
                 nt_lengths=nt_lengths,
                 is_real=np.array(self._is_real, dtype=bool))

    @classmethod
    def load(cls, path: str | Path) -> StartBoardModel:
        m = cls()
        data = np.load(path)
        boards     = data["boards"]
        nt_padded  = data["nt_padded"]
        nt_lengths = data["nt_lengths"]
        is_real    = data["is_real"]
        for i in range(len(boards)):
            m._boards.append(boards[i].copy())
            n = int(nt_lengths[i])
            m._next_tiles.append(list(map(int, nt_padded[i, :n])))
            m._is_real.append(bool(is_real[i]))
        return m


# ---------------------------------------------------------------------------
# Models bundle
# ---------------------------------------------------------------------------

@dataclass
class Models:
    """Convenience container holding all three models together."""
    next_tile:   NextTileModel   = field(default_factory=NextTileModel)
    placement:   PlacementModel  = field(default_factory=PlacementModel)
    start_board: StartBoardModel = field(default_factory=StartBoardModel)

    def update(self, obs: Observation) -> None:
        """Update all relevant models from a single observation."""
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

    Returns:
      GameState  — the new state after the move and tile placement
      "game_over"— a 12288 tile was created (terminal state)
      None       — the move was invalid (nothing on the board moved)

    The tile placed is drawn from state.next_tile.candidates:
      - 1 candidate: placed directly.
      - 3 candidates (bonus hint): one is sampled uniformly.
        (This will be refined once BonusTileModel is added.)
    The new next tile is sampled from NextTileModel.
    The placement position is sampled from PlacementModel.
    """
    new_board, eligible = _apply_slide(state.board, action)

    # Check for game over first: 12288 created by the slide itself
    if 12288 in new_board:
        return GAME_OVER

    if not eligible:
        return None   # invalid move (nothing moved)

    # Determine which tile to place
    if len(state.next_tile.candidates) == 1:
        tile_to_place = state.next_tile.candidates[0]
    else:
        # Bonus tile: sample uniformly for now
        tile_to_place = int(rng.choice(state.next_tile.candidates))

    # Check for game over before placing (if the merge would create 12288)
    pos = models.placement.sample(action, eligible, rng, real_weight)
    new_board[pos] = tile_to_place

    # Sample the new upcoming tile
    new_next_value = models.next_tile.sample(new_board, rng, real_weight)
    return GameState(new_board, NextTile([new_next_value]))
