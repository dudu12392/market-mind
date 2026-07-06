"""Abstract base class for market agents."""

from abc import ABC, abstractmethod

from src.environment import Action, Observation


class BaseAgent(ABC):
    """Every agent must implement `act(obs) -> Action`."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id: str = agent_id

    @abstractmethod
    def act(self, obs: Observation) -> Action:
        """Receive observation, return action."""
        ...
