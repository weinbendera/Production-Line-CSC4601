from abc import ABC, abstractmethod 
from typing import List, Dict, Optional
from backend.models import Action, Job


class Scheduler(ABC):
    """
    Abstract base class (interface) for all schedulers.

    Schedulers are used to choose actions for the factory's machines.
    Scheduler chooses the action -> factory runs the action -> factory updates its state -> scheduler chooses the next action.
    """
    def __init__(self):
        self.scheduled_actions = [] # list of actions that have been scheduled

class OnlineScheduler(Scheduler):
    """
    Abstract base class (interface) for all online schedulers.
    All concrete online scheduler implementations must implement the choose method.
    Online schedulers are used to choose the next actions for each ofthe factory's machines, one step at a time.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def choose(self, factory_state) -> Dict[str, Action | None]:
        """
        Chooses the next actions for each of the factory's machines.
        Returns the actions to be taken by the scheduler for each machine, or None if no actions were taken.
        """
        raise NotImplementedError("Subclasses must implement this method")

class OfflineScheduler(Scheduler):
    """
    Abstract base class (interface) for all offline schedulers.
    All concrete offline scheduler implementations must implement the schedule method.
    Offline schedulers are used to schedule all of the given jobs at once.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def schedule(self, factory_state) -> List[Action]:
        """
        Schedules the given jobs for all machines.
        Returns the scheduled actions taken by the scheduler.
        """
        raise NotImplementedError("Subclasses must implement this method")