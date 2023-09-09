import wandb
from pydantic import BaseModel
from typing import Dict, Optional
import os


class WandBConfig(BaseModel):
    project: str = "protein_lm_scaling"
    name: str
    # directory to save to
    dir: Optional[str] = None


def setup_wandb(full_config_dict: Dict) -> None:
    """
    Sets up logging via wieghts and biases
    Args:
        full_config_dict: contains the full config, not just
    the part corresponding to wandb, so that it can be logged
    """
    assert "wandb" in full_config_dict, f"If using wandb, need wandb section in config"
    wandb_config = WandBConfig(**full_config_dict["wandb"])
    if wandb_config.dir is not None:
        if not os.path.isdir(wandb_config.dir):
            print(f"creating wandb directory at {wandb_config.dir}")
            os.makedirs(wandb_config.dir)

    wandb.init(**dict(wandb_config), config=full_config_dict)
