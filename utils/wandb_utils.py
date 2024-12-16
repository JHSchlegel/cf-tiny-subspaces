import wandb
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any


def setup_wandb(
    project: str = "ETHZ AS2024 Deep Learning/cf-tiny-subspaces",
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Setup Weights & Biases logging.
    
    Args:
        project: Name of the wandb project
        name: Name of the run (optional)
        config: Configuration dictionary (optional)
    """
    load_dotenv()
    
    api_key = os.getenv('WANDB_API_KEY')

    if api_key is None:
        raise ValueError("WANDB_API_KEY not found! Please make sure you have access key to W&B.")
    
    wandb.login(api_key)
    
    if wandb.run is None:
        wandb.init(
            project=project,
            name=name,
            config=config
        )

    return True, "Logged into Weights & Biases!"
