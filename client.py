"""OpenEnv client entry required by some OpenEnv validators."""

from openenv.core.env_client import EnvClient

from models import (
    MovingTargetAction,
    MovingTargetEnvironmentState,
    MovingTargetObservation,
)


class MovingTargetClient(
    EnvClient[MovingTargetAction, MovingTargetObservation, MovingTargetEnvironmentState]
):
    """Typed client for the Moving Target environment."""


__all__ = ["MovingTargetClient"]
