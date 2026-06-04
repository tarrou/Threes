import tempfile
from pathlib import Path
import numpy as np
from threes.train import train
from threes.network import Network


def test_train_runs_and_saves():
    with tempfile.TemporaryDirectory() as td:
        net_dir   = Path(td) / "network"
        model_dir = Path(__file__).parent.parent / "data" / "models"
        net = train(
            n_episodes=3,
            rollout_n=2,
            max_moves=20,
            save_every=3,
            snapshot_every=1,
            net_dir=net_dir,
            model_dir=model_dir,
            resume=False,
        )
        assert isinstance(net, Network)
        assert net.n_updates > 0
        assert (net_dir / "net.npz").exists()


def test_train_resume():
    with tempfile.TemporaryDirectory() as td:
        net_dir   = Path(td) / "network"
        model_dir = Path(__file__).parent.parent / "data" / "models"
        kwargs = dict(n_episodes=3, rollout_n=2, max_moves=10,
                      save_every=3, net_dir=net_dir, model_dir=model_dir)
        net1 = train(**kwargs, resume=False)
        updates_after_first = net1.n_updates
        net2 = train(**kwargs, resume=True)
        assert net2.n_updates > updates_after_first
