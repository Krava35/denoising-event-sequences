import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d", seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")

    set_seed(42)
    a = torch.rand(4)
    set_seed(42)
    b = torch.rand(4)
    assert torch.equal(a, b), "set_seed is not deterministic"
    print(f"Reproducibility check passed: {a.tolist()}")

    device = get_device()
    print(f"Selected device: {device}")
