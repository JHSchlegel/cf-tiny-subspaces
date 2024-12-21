import wandb
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any


def setup_wandb(
    dir: Optional[str],
    project: str = "ETHZ AS2024 Deep Learning/cf-tiny-subspaces",
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Setup Weights & Biases logging.

    Args:
        dir (Optional[str]): W&B logging directory
        project (str): Name of the wandb project
        config (Optional[Dict[str, Any]]): Configuration dictionary (optional)
    """
    # load_dotenv()

    # api_key = os.getenv('WANDB_API_KEY')

    # if api_key is None:
    #     raise ValueError("WANDB_API_KEY not found! Please make sure you have access key to W&B.")

    # wandb.login(api_key)

    if wandb.run is None:
        wandb.init(dir=dir, project=project, config=config)

    return True, "Logged into Weights & Biases!"
