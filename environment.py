import random
import uuid
import numpy as np
from typing import Any,List,Union,Iterable

class Environment:
    """Base class for all classes that have a spatio-temporal dimension.
    Parameters
    ----------
    seconds_per_time_step (float): Number of seconds in 1 `time step` and must >= 1,default to be 900 (15min)
    random_seed (int):Pseudorandom number generator seed for repeatable results.
    """
    def __init__(self,seconds_per_time_step:float = 900,random_seed:float = 2023):
        self.__seconds_per_time_step = seconds_per_time_step
        self.__random_seed = random_seed
        self.__uid = uuid.uuid4().hex
        self.__time_step = None
        self.reset()
    @property
    def uid(self) -> str:
        """Unique environment id."""
        return self.__uid
    @property
    def random_seed(self) -> int:
        """Pseudorandom number generator seed for repeatable results."""
        return self.__random_seed
    @property
    def time_step(self) -> float:
        """Current environment time step. It should be counted from 0 to 4(4 time steps in an hour) x 24(24 hours in a day)"""
        return self.__time_step
    @property
    def seconds_per_time_step(self) -> float:
        return self.__seconds_per_time_step

    @seconds_per_time_step.setter
    def seconds_per_time_step(self,seconds_per_time_step:float):
        assert seconds_per_time_step >= 1
        self.__seconds_per_time_step = seconds_per_time_step

    def next_time_step(self):
        """Advance to next 'time_step' value
        Notes
        -----
        Override in subclass for custom implementation when advancing to next 'time_step'
        """
        self.__time_step += 1
    def reset(self):
        """Reset environment to initial state.
        Call 'reset_time_step'
        Notes
        -----
        Override in subclass for customizing implementation when resetting environment.
        """
        self.reset_time_step()
    def reset_time_step(self):
        """Reset 'time_step' to initial state(0).
        'time_step' is set to zero.
        """
        self.__time_step = 0