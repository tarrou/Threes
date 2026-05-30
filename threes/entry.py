"""
Interactive observation entry loop.

Run with:
    python -m threes.entry
    python -m threes.entry --data path/to/data/dir
"""

from __future__ import annotations
import argparse
from pathlib import Path

from .simulator import (
    GameState, NextTile, Observation,
    parse_state, _apply_slide, ACTION_NAMES,
)
from .models import Models

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

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _fmt_board(state: GameState) -> str:
    rows = []
    for r in range(4):
        rows.append("  ".join(f"{v:5}" for v in state.board[r]))
    rows.append(f"  next: {state.next_tile}")
    return "\n".join(rows)


def _show_state(label: str, state: GameState) -> None:
    print(f"\n{label}:")
    print(_fmt_board(state))


def _prompt(msg: str) -> str:
    return input(f"\n{msg}: ").strip()


def _confirm(msg: str) -> bool:
    ans = input(f"{msg} (y/n): ").strip().lower()
    return ans in ("y", "yes")


# ---------------------------------------------------------------------------
# Sub-flows
# ---------------------------------------------------------------------------

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
    _show_state("  Board", state)
    return state


def _enter_action() -> int | None:
    """Prompt for an action, return int 0-3 or None on error."""
    raw = _prompt("Action (up/right/down/left  or  u/r/d/l)").lower()
    action = ACTION_ALIASES.get(raw)
    if action is None:
        print(f"  Unknown action '{raw}'. Use: up/right/down/left or u/r/d/l.")
    return action


def _record_observation(models: Models) -> bool:
    """
    Walk through entering a full before→action→after observation.
    Returns True if an observation was recorded, False otherwise.
    """
    print("\n--- New Observation ---")

    before = _enter_state("BEFORE state  (16 board values / next tile candidates)")
    if before is None:
        return False

    action = _enter_action()
    if action is None:
        return False

    print(f"  Action: {ACTION_NAMES[action]}")

    # Show what the slide produces so the user can spot inconsistencies
    slid_board, eligible = _apply_slide(before.board, action)
    if not eligible:
        print("  Warning: this action produces no valid move on the entered board.")
        if not _confirm("  Continue anyway?"):
            return False

    after = _enter_state("AFTER state   (16 board values / next tile candidates)")
    if after is None:
        return False

    obs = Observation(before, action, after, is_real=True)

    # Infer and display what was placed
    placed_tile = obs.placed_tile()
    placed_pos  = obs.placed_position()

    print()
    if placed_tile is not None and placed_pos is not None:
        print(f"  Inferred placement: tile {placed_tile} at row {placed_pos[0]}, col {placed_pos[1]}")
        if placed_tile not in before.next_tile.candidates:
            print(f"  Warning: placed tile {placed_tile} is not in next-tile candidates "
                  f"{before.next_tile.candidates}")
        if eligible and placed_pos not in eligible:
            print(f"  Warning: placement position {placed_pos} is not on the eligible "
                  f"trailing edge {eligible}")
    else:
        print("  Could not infer placement (board diff is ambiguous or zero).")
        print("  The observation will still be recorded for next-tile learning.")

    if not _confirm("  Record this observation?"):
        print("  Skipped.")
        return False

    models.update(obs)
    return True


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
  obs   (o)  — record a game observation (before → action → after)
  start (s)  — record a starting board
  show        — show model observation counts
  quit  (q)  — save models and exit
  help  (?)  — show this message
"""


def run(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load or create models
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
            recorded = _record_observation(models)
            if recorded:
                models.save(data_dir)
                print("  Saved.")
                print()
                print(models.summary())

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
