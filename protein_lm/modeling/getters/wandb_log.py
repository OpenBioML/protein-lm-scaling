import wandb
from pydantic import BaseModel
from typing import Dict, Optional
import os


class WandBConfig(BaseModel):
    project: str = "protein_lm_scaling"
    name: str
    # directory to save to
    dir: Optional[str] = None


def setup_wandb(config_dict: Dict) -> None:
    config = WandBConfig(**config_dict)
    if config.dir is not None:
        if not os.path.isdir(config.dir):
            print(f"creating wandb directory at {config.dir}")
            os.makedirs(config.dir)

    os.environ["WANDB_PROJECT"] = config.project
    os.environ["WANDB_NAME"] = config.name
    os.environ["WANDB_DIR"] = config.dir
