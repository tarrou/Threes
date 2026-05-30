"""
Interactive observation entry loop.

Run with:
    python -m threes.entry
    python -m threes.entry --data path/to/data/dir
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from .simulator import (
    GameState, NextTile, Observation,
    parse_state, _apply_slide, ACTION_NAMES, TILE_SET, TILE_VALUES,
)
from .models import Models, TILE_INDEX, REAL, SIM

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "models"

ACTION_ALIASES: dict[str, int] = {
    "up": 0,    "u": 0,
    "right": 1, "r": 1,
    "down": 2,  "d": 2,
    "left": 3,  "l": 3,
}

# For each action: (axis, fixed_index)
# axis 0 = row is fixed (up/down), axis 1 = col is fixed (left/right)
# fixed_index = the row or col number where the new tile always appears
ACTION_TRAILING: dict[int, tuple[str, int]] = {
    0: ("col", 3),   # up:    tile in row 3, ask which col
    1: ("row", 0),   # right: tile in col 0, ask which row
    2: ("col", 0),   # down:  tile in row 0, ask which col
    3: ("row", 3),   # left:  tile in col 3, ask which row
}

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _fmt_board(board: np.ndarray, next_tile: NextTile | None = None) -> str:
    rows = []
    for r in range(4):
        rows.append("  ".join(f"{v:5}" for v in board[r]))
    if next_tile is not None:
        rows.append(f"  next: {next_tile}")
    return "\n".join(rows)


def _show_board(label: str, board: np.ndarray,
                next_tile: NextTile | None = None) -> None:
    print(f"\n{label}:")
    print(_fmt_board(board, next_tile))


def _prompt(msg: str) -> str:
    return input(f"\n{msg}: ").strip()


def _confirm(msg: str) -> bool:
    ans = input(f"{msg} (y/n): ").strip().lower()
    return ans in ("y", "yes")


def _prompt_int(msg: str, lo: int, hi: int) -> int | None:
    """Prompt for an integer in [lo, hi]. Returns None on bad input."""
    raw = _prompt(msg)
    try:
        val = int(raw)
        if lo <= val <= hi:
            return val
        print(f"  Must be between {lo} and {hi}.")
    except ValueError:
        print(f"  Not a valid number.")
    return None


def _enter_next_tile(prompt: str) -> NextTile | None:
    """Prompt for 1 or 3 tile values. Returns None on error."""
    raw = _prompt(prompt)
    if not raw:
        return None
    try:
        vals = list(map(int, raw.split()))
        if len(vals) not in (1, 3):
            print("  Enter 1 value (normal tile) or 3 values (bonus hint).")
            return None
        for v in vals:
            if v not in TILE_SET:
                print(f"  Invalid tile value: {v}")
                return None
        return NextTile(vals)
    except ValueError:
        print("  Could not parse tile values.")
        return None


# ---------------------------------------------------------------------------
# Core observation flow
# ---------------------------------------------------------------------------

def _do_one_observation(models: Models,
                        before: GameState) -> tuple[bool, GameState | None]:
    """
    Carry out one before→action→after observation starting from `before`.

    Returns (recorded, after_state).
    after_state is the completed after GameState (board + next tile),
    or None if the flow was abandoned.
    """
    # --- Action (re-prompt until a valid move is entered or user quits) ---
    while True:
        action = _enter_action()
        if action is None:
            return False, None
        slid_board, eligible = _apply_slide(before.board, action)
        if eligible:
            break
        print(f"  '{ACTION_NAMES[action]}' doesn't move any tiles on this board."
              f" Enter a different action (or q to abort).")

    print(f"  Action: {ACTION_NAMES[action]}")

    print("\n  Board after slide:")
    print(_fmt_board(slid_board))

    # --- Placement ---
    ask_axis, fixed_idx = ACTION_TRAILING[action]

    # Derive the eligible indices along the free axis
    if ask_axis == "row":
        eligible_indices = sorted({r for r, c in eligible})
        axis_label, lo_label, hi_label = "row", "top", "bottom"
    else:
        eligible_indices = sorted({c for r, c in eligible})
        axis_label, lo_label, hi_label = "column", "left", "right"

    if not eligible_indices:
        print("  No eligible positions — this move produced no open slots.")
        return False, None

    if len(eligible_indices) == 1:
        idx = eligible_indices[0]
        print(f"  New tile automatically placed in {axis_label} {idx} (only option).")
    else:
        options_str = " ".join(str(i) for i in eligible_indices)
        while True:
            raw = _prompt(
                f"  Which {axis_label} did the new tile appear in?"
                f" (0={lo_label} … 3={hi_label}, options: {options_str}  |  q=abort)")
            if raw.strip().lower() == "q":
                return False, None
            try:
                idx = int(raw)
            except ValueError:
                print("  Not a valid number.")
                continue
            if idx not in eligible_indices:
                print(f"  Impossible — {idx} is not an eligible {axis_label}."
                      f" Eligible options: {options_str}")
                continue
            break

    placement = (idx, fixed_idx) if ask_axis == "row" else (fixed_idx, idx)

    # --- Which tile was placed? ---
    if before.next_tile.is_bonus():
        cands = before.next_tile.candidates
        label = "  ".join(f"{i}={v}" for i, v in enumerate(cands))
        choice = _prompt_int(f"  Bonus tile — which was placed? ({label})", 0, 2)
        if choice is None:
            return False, None
        tile_placed = cands[choice]
    else:
        tile_placed = before.next_tile.candidates[0]

    # Place tile on board
    after_board = slid_board.copy()
    after_board[placement] = tile_placed

    print("\n  Board with new tile placed:")
    print(_fmt_board(after_board))

    # --- Next tile ---
    next_tile = _enter_next_tile("  Next tile shown now (value, or 3 values for bonus)")
    if next_tile is None:
        return False, None

    after_state = GameState(after_board, next_tile)
    _show_board("  After state", after_board, next_tile)

    # --- Confirm ---
    if not _confirm("  Record this observation?"):
        print("  Skipped.")
        return False, after_state   # still return after_state so caller can chain

    obs = Observation(before, action, after_state, is_real=True)
    models.update(obs)
    return True, after_state


def _enter_action() -> int | None:
    """Prompt for an action. Returns int 0-3, or None if user types q/quit."""
    raw = _prompt("Action (u/r/d/l  or  up/right/down/left  |  q=abort)").lower()
    if raw in ("q", "quit"):
        return None
    action = ACTION_ALIASES.get(raw)
    if action is None:
        print(f"  Unknown action '{raw}'.")
    return action


# ---------------------------------------------------------------------------
# Multi-observation chain
# ---------------------------------------------------------------------------

def _record_chain(models: Models) -> int:
    """
    Enter one or more chained observations.
    Returns the number of observations recorded.
    """
    print("\n--- New Observation ---")

    before = _enter_state("BEFORE state  (16 board values / next tile candidates)")
    if before is None:
        return 0

    total_recorded = 0
    first = True

    while True:
        if not first:
            _show_board("  Current board", before.board, before.next_tile)
        first = False

        recorded, after_state = _do_one_observation(models, before)

        if recorded:
            total_recorded += 1
            models.save(_current_data_dir)
            print("  Saved.")
            print()
            print(models.summary())

        if after_state is None:
            break

        if _confirm("\n  Continue from this state?"):
            before = after_state
        else:
            break

    return total_recorded


def _enter_state(prompt: str) -> GameState | None:
    """Prompt for a state string, parse it, display it, return it or None on error."""
    raw = _prompt(prompt)
    if not raw:
        return None
    try:
        state = parse_state(raw)
    except ValueError as e:
        print(f"  Parse error: {e}")
        return None
    _show_board("  Board", state.board, state.next_tile)
    return state


# ---------------------------------------------------------------------------
# Distribution inspection
# ---------------------------------------------------------------------------

def _show_next(models: Models) -> None:
    """Print the NextTileModel distribution for every bucket that has data."""
    print("\n=== Next Tile Distributions ===")
    m = models.next_tile
    any_data = False
    for bucket_idx, max_tile in enumerate(TILE_VALUES):
        real_counts = m._counts[REAL, bucket_idx]
        sim_counts  = m._counts[SIM,  bucket_idx]
        total = real_counts.sum() + sim_counts.sum()
        if total == 0:
            continue
        any_data = True
        print(f"\n  Max tile on board = {max_tile}"
              f"  (real: {int(real_counts.sum())}, sim: {int(sim_counts.sum())})")
        # Use default real_weight=10 for the displayed PMF
        board_proxy = max_tile * (1 if max_tile > 0 else 0)   # dummy — we use bucket directly
        pmf = (real_counts * 10.0 + sim_counts)
        pmf_sum = pmf.sum()
        if pmf_sum > 0:
            pmf /= pmf_sum
        for vi, tile_val in enumerate(TILE_VALUES):
            if tile_val == 0:
                continue
            p = pmf[vi]
            r = int(real_counts[vi])
            s = int(sim_counts[vi])
            if r + s == 0:
                continue
            bar = "█" * int(p * 30)
            print(f"    {tile_val:6}  {p:5.1%}  {bar}  (real={r}, sim={s})")
    if not any_data:
        print("  No data recorded yet.")


def _show_place(models: Models) -> None:
    """Print the PlacementModel distribution for every (action, n_eligible) with data."""
    print("\n=== Placement Distributions ===")
    m = models.placement
    if not m._counts:
        print("  No data recorded yet.")
        return

    # Group by (action, n_eligible)
    seen = set()
    for src, action, n in sorted(m._counts.keys()):
        seen.add((action, n))

    for action, n in sorted(seen):
        real_arr = m._counts.get((REAL, action, n), None)
        sim_arr  = m._counts.get((SIM,  action, n), None)
        real_counts = real_arr if real_arr is not None else [0] * n
        sim_counts  = sim_arr  if sim_arr  is not None else [0] * n
        total_r = sum(real_counts)
        total_s = sum(sim_counts)
        if total_r + total_s == 0:
            continue

        ask_axis, fixed_idx = ACTION_TRAILING[action]
        axis_label = "row" if ask_axis == "row" else "col"

        print(f"\n  Action={ACTION_NAMES[action]}, eligible slots={n}"
              f"  (real: {int(total_r)}, sim: {int(total_s)})")
        pmf = real_counts * 10.0 + sim_counts
        pmf_total = pmf.sum()
        if pmf_total > 0:
            pmf /= pmf_total
        ordinals = ["1st", "2nd", "3rd", "4th"]
        for i in range(n):
            p = pmf[i]
            r = int(real_counts[i])
            s = int(sim_counts[i])
            bar = "█" * int(p * 30)
            label = f"{ordinals[i]} eligible {axis_label}"
            print(f"    {label}  {p:5.1%}  {bar}  (real={r}, sim={s})")


def _show_start(models: Models) -> None:
    """Print all stored starting boards."""
    print("\n=== Starting Boards ===")
    m = models.start_board
    if not m._boards:
        print("  No starting boards recorded yet.")
        return
    for i, (board, cands, is_real) in enumerate(
            zip(m._boards, m._next_tiles, m._is_real)):
        tag = "real" if is_real else "sim"
        print(f"\n  [{i}] ({tag})  next: {' '.join(str(v) for v in cands)}")
        for r in range(4):
            print("    " + "  ".join(f"{v:5}" for v in board[r]))


def _record_start(models: Models) -> bool:
    """Enter a starting board example."""
    print("\n--- Starting Board ---")
    state = _enter_state("Starting state  (16 board values / next tile candidates)")
    if state is None:
        return False
    if _confirm("  Record as real starting board?"):
        models.start_board.update(state, is_real=True)
        return True
    return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

HELP_TEXT = """
Commands:
  obs   (o)      — record one or more chained game observations
  start (s)      — record a starting board
  show           — show model observation counts
  show next      — next-tile probability distributions by max-tile bucket
  show place     — placement probability distributions by action
  show start     — all stored starting boards
  quit  (q)      — save models and exit
  help  (?)      — show this message
"""

_current_data_dir: Path = DEFAULT_DATA_DIR


def run(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    global _current_data_dir
    _current_data_dir = data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        models = Models.load(data_dir)
        print(f"Models loaded from {data_dir}")
    except Exception:
        models = Models()
        print("No existing models found — starting fresh.")

    print("\n=== Threes Observation Entry ===")
    print(HELP_TEXT)
    print(models.summary())

    while True:
        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            cmd = "quit"

        if cmd in ("obs", "o"):
            _record_chain(models)

        elif cmd in ("start", "s"):
            recorded = _record_start(models)
            if recorded:
                models.save(data_dir)
                print("  Saved.")
                print()
                print(models.summary())

        elif cmd == "show":
            print()
            print(models.summary())
        elif cmd == "show next":
            _show_next(models)
        elif cmd in ("show place", "show placement"):
            _show_place(models)
        elif cmd in ("show start", "show starts"):
            _show_start(models)

        elif cmd in ("quit", "q", "exit"):
            models.save(data_dir)
            print("Models saved. Goodbye.")
            break

        elif cmd in ("help", "?", "h"):
            print(HELP_TEXT)

        elif cmd == "":
            pass

        else:
            print(f"  Unknown command '{cmd}'. Type 'help' for options.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Threes observation entry loop")
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Directory to load/save models (default: {DEFAULT_DATA_DIR})"
    )
    args = parser.parse_args()
    run(data_dir=args.data)


if __name__ == "__main__":
    main()
