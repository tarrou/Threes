"""
Game runner for Threes.

Wraps the simulator and probability models into a clean interface for
the neural network training loop.

Key design:
  - act(action)    — permanently advances the game state one move
  - rollout(n)     — plays n random moves from the current state to
                     estimate its value, then RESTORES the state
  - valid_actions()— boolean mask so the net never picks an invalid move

Rollout length N=8 is the default; change ROLLOUT_N to adjust.
"""

from __future__ import annotations
import numpy as np

from .simulator import GameState, _apply_slide, score_board, ACTION_NAMES
from .models import Models, GAME_OVER, step

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLLOUT_N = 8   # number of random moves in each evaluation rollout


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class Game:
    """
    Manages a single Threes game session.

    Typical training loop usage:
        game = Game(models, rng)
        state = game.reset()

        while not game.done:
            mask   = game.valid_actions()          # shape (4,) bool
            action = net.choose(state, mask)       # net respects mask
            state, done = game.act(action)
            if not done:
                reward = game.rollout()            # N-step random evaluation
            else:
                reward = score_board(state.board)  # terminal reward
            net.update(action, reward)
    """

    def __init__(self, models: Models, rng: np.random.Generator) -> None:
        self.models = models
        self.rng    = rng
        self.state: GameState = models.start_board.generate(rng)
        self.done:  bool      = False

    # ------------------------------------------------------------------
    # Core interface

    def reset(self) -> GameState:
        """Start a new game and return the initial state."""
        self.state = self.models.start_board.generate(self.rng)
        self.done  = False
        return self.state

    def valid_actions(self) -> np.ndarray:
        """
        Boolean array of length 4.
        True if the action produces at least one eligible spawn position.
        """
        mask = np.zeros(4, dtype=bool)
        for a in range(4):
            _, eligible, _ = _apply_slide(self.state.board, a)
            mask[a] = len(eligible) > 0
        return mask

    def act(self, action: int) -> tuple[GameState, bool]:
        """
        Apply action and permanently advance the game state.

        Returns (new_state, done).
        done is True on game-over (12288 created) or if no valid moves remain.

        Calling act() with an invalid action (mask[action] is False) is a
        no-op: the state is unchanged and done stays False.
        """
        if self.done:
            return self.state, True

        result = step(self.state, action, self.models, self.rng)

        if result is None:
            # Invalid move — no-op (caller should use valid_actions mask)
            return self.state, False

        if result == GAME_OVER:
            self.done = True
            return self.state, True

        self.state = result

        # Also done if no valid moves remain on the new board
        if not self.valid_actions().any():
            self.done = True

        return self.state, self.done

    # ------------------------------------------------------------------
    # Rollout evaluation

    def rollout(self, n: int = ROLLOUT_N) -> float:
        """
        Estimate the value of the current state by playing n random moves,
        then return the board score.  The game state is RESTORED afterward
        so act() history is unaffected.

        If the game ends before n moves the final board score is returned
        immediately.  If no valid moves exist the current score is returned.
        """
        saved_state = self.state.copy()
        saved_done  = self.done

        score = self._run_random(n)

        self.state = saved_state
        self.done  = saved_done
        return score

    def _run_random(self, n: int) -> float:
        """Play up to n random valid moves in-place. Returns final score."""
        for _ in range(n):
            if self.done:
                break
            mask = self.valid_actions()
            if not mask.any():
                self.done = True
                break
            valid   = np.where(mask)[0]
            action  = int(self.rng.choice(valid))
            _, done = self.act(action)
            if done:
                break
        return score_board(self.state.board)

    # ------------------------------------------------------------------
    # Display

    def __str__(self) -> str:
        lines = []
        for r in range(4):
            lines.append("  ".join(f"{v:5}" for v in self.state.board[r]))
        lines.append(f"next: {self.state.next_tile}  "
                     f"score: {score_board(self.state.board):.0f}  "
                     f"{'DONE' if self.done else ''}")
        return "\n".join(lines)
