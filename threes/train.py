"""
Training loop for the Threes policy network.

Each episode:
  1. Reset the game to a generated starting board.
  2. Loop until the game ends or max_moves is reached:
       a. Get valid-action mask from the game.
       b. Network samples an action.
       c. Apply the action permanently (game.act).
       d. Evaluate the resulting state with an N-step random rollout
          (game.rollout) — or use the final board score if the game ended.
       e. Update network weights via REINFORCE.
  3. Log stats.

Run:
    python -m threes.train
    python -m threes.train --episodes 5000 --rollout-n 8 --lr 1e-3
"""

from __future__ import annotations
import argparse
from collections import deque
from pathlib import Path

import numpy as np

from .simulator import score_board
from .models import Models
from .game import Game, ROLLOUT_N
from .network import Network

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_EPISODES   = 2000
DEFAULT_LR         = 1e-3
DEFAULT_ROLLOUT_N  = ROLLOUT_N          # 8
DEFAULT_MAX_MOVES  = 500               # safety cap per episode
DEFAULT_SAVE_EVERY = 100               # save network every N episodes
DEFAULT_NET_DIR    = Path(__file__).parent.parent / "data" / "network"
DEFAULT_MODEL_DIR  = Path(__file__).parent.parent / "data" / "models"


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    n_episodes:   int   = DEFAULT_EPISODES,
    rollout_n:    int   = DEFAULT_ROLLOUT_N,
    max_moves:    int   = DEFAULT_MAX_MOVES,
    lr:           float = DEFAULT_LR,
    baseline_decay: float = 0.99,
    snapshot_every: int = 100,
    save_every:   int   = DEFAULT_SAVE_EVERY,
    net_dir:      Path  = DEFAULT_NET_DIR,
    model_dir:    Path  = DEFAULT_MODEL_DIR,
    seed:         int   = 42,
    resume:       bool  = True,
) -> Network:
    """
    Train the policy network and return it.

    Parameters
    ----------
    n_episodes    : number of training episodes
    rollout_n     : random moves per reward evaluation (N-step lookahead)
    max_moves     : max moves per episode (safety cap)
    lr            : network learning rate
    baseline_decay: EMA decay for REINFORCE baseline
    snapshot_every: weight snapshot interval (for history analysis)
    save_every    : save network to disk every N episodes
    net_dir       : directory to save/load the network
    model_dir     : directory holding the probability models
    seed          : RNG seed
    resume        : if True and a saved network exists, load and continue
    """
    rng     = np.random.default_rng(seed)
    net_dir = Path(net_dir)
    net_dir.mkdir(parents=True, exist_ok=True)
    net_path = net_dir / "net.npz"

    # Load models
    models = Models.load(model_dir)

    # Load or create network
    if resume and net_path.exists():
        net = Network.load(net_path)
        print(f"Resumed network from {net_path}  ({net.n_updates} updates so far)")
    else:
        net = Network(lr=lr, baseline_decay=baseline_decay,
                      snapshot_every=snapshot_every,
                      rng=np.random.default_rng(seed + 1))
        print(f"New network  lr={lr}  rollout_n={rollout_n}")

    game = Game(models, rng)

    # Running stats (last 100 episodes)
    recent_scores:    deque[float] = deque(maxlen=100)
    recent_max_tiles: deque[int]   = deque(maxlen=100)

    print(f"\nTraining for {n_episodes} episodes  "
          f"(rollout_n={rollout_n}, max_moves={max_moves})\n")
    print(f"{'Episode':>8}  {'Score':>8}  {'MaxTile':>8}  "
          f"{'Moves':>6}  {'Avg100':>8}  {'Baseline':>9}")
    print("-" * 60)

    for ep in range(1, n_episodes + 1):
        state = game.reset()
        episode_moves = 0

        while not game.done and episode_moves < max_moves:
            mask   = game.valid_actions()
            action = net.choose(state, mask, rng)

            prev_state = state
            state, done = game.act(action)
            episode_moves += 1

            if done:
                reward = score_board(game.state.board)
            else:
                reward = game.rollout(n=rollout_n)

            net.update(prev_state, action, reward, mask)

        final_score = score_board(game.state.board)
        max_tile    = int(game.state.board.max())
        recent_scores.append(final_score)
        recent_max_tiles.append(max_tile)

        # Periodic console output
        if ep % 10 == 0 or ep == 1:
            avg = np.mean(recent_scores)
            print(f"{ep:>8}  {final_score:>8.0f}  {max_tile:>8}  "
                  f"{episode_moves:>6}  {avg:>8.0f}  {net.baseline:>9.1f}")

        # Periodic save
        if ep % save_every == 0:
            net.save(net_path)

    # Final save
    net.save(net_path)
    print(f"\nDone. Network saved to {net_path}")
    print(f"Final avg score (last 100): {np.mean(recent_scores):.0f}")
    print(f"Max tile seen  (last 100): {max(recent_max_tiles)}")
    print(net.summary())

    return net


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Train the Threes policy network")
    p.add_argument("--episodes",       type=int,   default=DEFAULT_EPISODES)
    p.add_argument("--rollout-n",      type=int,   default=DEFAULT_ROLLOUT_N)
    p.add_argument("--max-moves",      type=int,   default=DEFAULT_MAX_MOVES)
    p.add_argument("--lr",             type=float, default=DEFAULT_LR)
    p.add_argument("--baseline-decay", type=float, default=0.99)
    p.add_argument("--snapshot-every", type=int,   default=100)
    p.add_argument("--save-every",     type=int,   default=DEFAULT_SAVE_EVERY)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--no-resume",      action="store_true")
    p.add_argument("--net-dir",        type=Path,  default=DEFAULT_NET_DIR)
    p.add_argument("--model-dir",      type=Path,  default=DEFAULT_MODEL_DIR)
    args = p.parse_args()

    train(
        n_episodes    = args.episodes,
        rollout_n     = args.rollout_n,
        max_moves     = args.max_moves,
        lr            = args.lr,
        baseline_decay= args.baseline_decay,
        snapshot_every= args.snapshot_every,
        save_every    = args.save_every,
        net_dir       = args.net_dir,
        model_dir     = args.model_dir,
        seed          = args.seed,
        resume        = not args.no_resume,
    )


if __name__ == "__main__":
    main()
