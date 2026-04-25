from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class MovingTargetAction(BaseModel):
    """Action for the Moving Target environment."""
    tool: str = Field(description="Tool to call: 'ask_watchdog' or 'place_order'")
    merchant_name: str = Field(description="Name of the merchant")
    payload: Optional[Dict[str, Any]] = Field(default=None, description="Order payload (for place_order)")


class MovingTargetObservation(BaseModel):
    """Observation returned by the Moving Target environment."""
    data: str = Field(default="", description="Response data from the environment")
    status: int = Field(default=200, description="HTTP-like status code")
    reward: Optional[float] = Field(default=None, description="Reward scalar for RL.")
    done: bool = Field(default=False, description="Flag indicating if the episode is finished.")

class MovingTargetEnvironmentState(BaseModel):
    """The literal state of the environment server."""
    episode_id: Optional[str] = Field(default=None)
    step_count: int = Field(default=0)