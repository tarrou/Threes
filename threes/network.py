"""
Policy network for Threes, implemented in pure NumPy.

Architecture: fully-connected feedforward network.
  Input  : 19 floats  (encode_for_net output)
  Hidden : one or more ReLU layers (default: [64])
  Output : 4 logits  → masked softmax → action probabilities

Training: REINFORCE (policy gradient) with a running baseline.

  For each move:
    1. Forward pass → probabilities over valid actions
    2. Sample action
    3. Run Game.rollout() to get reward R
    4. Advantage = R − baseline
    5. Gradient ascent on log π(a|s) · advantage
    6. Update baseline (exponential moving average)

Weight history: snapshots are saved every `snapshot_every` updates so
  the evolution of learning can be inspected later.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from .simulator import GameState, encode_for_net

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SIZE    = 19
OUTPUT_SIZE   = 4
DEFAULT_HIDDEN = [64]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Network:
    """
    Policy network with configurable hidden layers.

    Parameters
    ----------
    hidden_sizes   : list of hidden layer widths, e.g. [64] or [64, 32]
    lr             : learning rate for gradient ascent
    baseline_decay : EMA decay for the reward baseline (0 = no baseline)
    snapshot_every : save a weight snapshot every N updates (0 = never)
    rng            : numpy random generator for weight init
    """

    def __init__(self,
                 hidden_sizes:   list[int]           = DEFAULT_HIDDEN,
                 lr:             float                = 1e-3,
                 baseline_decay: float                = 0.99,
                 snapshot_every: int                  = 100,
                 rng:            np.random.Generator  | None = None) -> None:

        if rng is None:
            rng = np.random.default_rng()

        self.lr             = lr
        self.baseline_decay = baseline_decay
        self.snapshot_every = snapshot_every
        self.n_updates      = 0
        self.baseline       = 0.0

        # Build layers: list of (W, b) pairs
        sizes = [INPUT_SIZE] + hidden_sizes + [OUTPUT_SIZE]
        self.weights: list[np.ndarray] = []
        self.biases:  list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            # Xavier initialisation
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(rng.normal(0.0, scale, (fan_out, fan_in)))
            self.biases.append(np.zeros(fan_out))

        # History: list of dicts with weight snapshots
        self.history: list[dict] = []
        self._maybe_snapshot()   # record initial weights

    # ------------------------------------------------------------------
    # Forward pass

    def forward(self, x: np.ndarray,
                mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Compute action probabilities.

        Parameters
        ----------
        x    : (19,) encoded state
        mask : (4,) bool — True for valid actions

        Returns
        -------
        probs      : (4,) softmax probabilities (invalid actions are 0)
        activations: list of pre-activation outputs per layer (for backprop)
        """
        activations: list[np.ndarray] = [x]
        h = x
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.maximum(0.0, W @ h + b)   # ReLU hidden layer
            activations.append(h)

        # Output layer — no activation yet, just logits
        logits = self.weights[-1] @ h + self.biases[-1]

        # Mask invalid actions before softmax
        masked = logits.copy()
        masked[~mask] = -1e9

        # Numerically stable softmax
        masked -= masked.max()
        exp = np.exp(masked)
        probs = exp / exp.sum()

        return probs, activations

    # ------------------------------------------------------------------
    # Action selection

    def choose(self, state: GameState, mask: np.ndarray,
               rng: np.random.Generator,
               greedy: bool = False) -> int:
        """
        Sample (or greedily select) an action.

        Parameters
        ----------
        greedy : if True, return argmax instead of sampling
        """
        x     = encode_for_net(state)
        probs, _ = self.forward(x, mask)
        if greedy:
            return int(np.argmax(probs))
        return int(rng.choice(OUTPUT_SIZE, p=probs))

    # ------------------------------------------------------------------
    # REINFORCE update

    def update(self, state: GameState, action: int,
               reward: float, mask: np.ndarray) -> float:
        """
        Apply one REINFORCE gradient step.

        Returns the advantage (reward − baseline) used for this update.
        """
        x            = encode_for_net(state)
        probs, acts  = self.forward(x, mask)

        # Update baseline (EMA)
        self.baseline = (self.baseline_decay * self.baseline
                         + (1.0 - self.baseline_decay) * reward)
        advantage = reward - self.baseline

        # ∂ log π(a|s) / ∂ logits = one_hot(a) − probs
        grad_logits = -probs.copy()
        grad_logits[action] += 1.0
        grad_logits *= advantage   # scale by advantage

        # Backpropagate through layers
        # Output layer gradient
        dW = np.outer(grad_logits, acts[-1])
        db = grad_logits.copy()
        self.weights[-1] += self.lr * dW
        self.biases[-1]  += self.lr * db

        # Hidden layers (reverse order)
        delta = self.weights[-1].T @ grad_logits
        for i in range(len(self.weights) - 2, -1, -1):
            h    = acts[i + 1]
            # ReLU derivative
            delta = delta * (h > 0)
            dW = np.outer(delta, acts[i])
            db = delta.copy()
            self.weights[i] += self.lr * dW
            self.biases[i]  += self.lr * db
            if i > 0:
                delta = self.weights[i].T @ delta

        self.n_updates += 1
        self._maybe_snapshot()
        return advantage

    # ------------------------------------------------------------------
    # Weight history

    def _maybe_snapshot(self) -> None:
        if self.snapshot_every > 0 and self.n_updates % self.snapshot_every == 0:
            self.history.append({
                "step":    self.n_updates,
                "weights": [W.copy() for W in self.weights],
                "biases":  [b.copy() for b in self.biases],
            })

    def weight_history(self) -> dict[str, np.ndarray]:
        """
        Return the weight history as stacked arrays for easy analysis.

        Returns a dict:
          "steps"     : (n_snapshots,)
          "W{i}"      : (n_snapshots, fan_out, fan_in) for each layer i
          "b{i}"      : (n_snapshots, fan_out)         for each layer i
        """
        if not self.history:
            return {}
        steps = np.array([s["step"] for s in self.history])
        result: dict[str, np.ndarray] = {"steps": steps}
        n_layers = len(self.weights)
        for i in range(n_layers):
            result[f"W{i}"] = np.stack([s["weights"][i] for s in self.history])
            result[f"b{i}"] = np.stack([s["biases"][i]  for s in self.history])
        return result

    # ------------------------------------------------------------------
    # Persistence

    def save(self, path: str | Path) -> None:
        """Save current weights and full history to a .npz file."""
        arrays: dict[str, np.ndarray] = {}
        n = len(self.weights)
        arrays["n_layers"]      = np.array(n)
        arrays["lr"]            = np.array(self.lr)
        arrays["baseline_decay"]= np.array(self.baseline_decay)
        arrays["snapshot_every"]= np.array(self.snapshot_every)
        arrays["n_updates"]     = np.array(self.n_updates)
        arrays["baseline"]      = np.array(self.baseline)
        for i in range(n):
            arrays[f"W{i}"] = self.weights[i]
            arrays[f"b{i}"] = self.biases[i]
        # History
        arrays["n_history"] = np.array(len(self.history))
        for hi, snap in enumerate(self.history):
            arrays[f"h_step_{hi}"] = np.array(snap["step"])
            for i in range(n):
                arrays[f"h_W{i}_{hi}"] = snap["weights"][i]
                arrays[f"h_b{i}_{hi}"] = snap["biases"][i]
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Network:
        data = np.load(path)
        n    = int(data["n_layers"])
        net  = cls.__new__(cls)
        net.lr             = float(data["lr"])
        net.baseline_decay = float(data["baseline_decay"])
        net.snapshot_every = int(data["snapshot_every"])
        net.n_updates      = int(data["n_updates"])
        net.baseline       = float(data["baseline"])
        net.weights        = [data[f"W{i}"] for i in range(n)]
        net.biases         = [data[f"b{i}"] for i in range(n)]
        n_history          = int(data["n_history"])
        net.history        = []
        for hi in range(n_history):
            net.history.append({
                "step":    int(data[f"h_step_{hi}"]),
                "weights": [data[f"h_W{i}_{hi}"] for i in range(n)],
                "biases":  [data[f"h_b{i}_{hi}"] for i in range(n)],
            })
        return net

    # ------------------------------------------------------------------
    # Display

    def summary(self) -> str:
        shapes = " → ".join(
            f"{W.shape[1]}" for W in self.weights
        ) + f" → {self.weights[-1].shape[0]}"
        return (f"Network  [{shapes}]  "
                f"lr={self.lr}  updates={self.n_updates}  "
                f"snapshots={len(self.history)}  baseline={self.baseline:.1f}")
