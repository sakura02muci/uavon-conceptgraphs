from enum import Enum
from typing import Dict, Union
import attr
import argparse
from common.param import args


class AirsimActions(Enum):
    """AirSim action enumeration"""
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    UP = 4
    DOWN = 5
    TURN_LEFT = 6
    TURN_RIGHT = 7
    STOP = 8


class Singleton(type):
    _instances: Dict["Singleton", "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]

class _DefaultAirsimActionSettings(Dict):
    FORWARD_STEP_SIZE = args.xOy_step_size
    UP_DOWN_STEP_SIZE = args.z_step_size
    LEFT_RIGHT_STEP_SIZE = args.xOy_step_size
    TURN_ANGLE = args.rotateAngle
   
AirsimActionSettings: _DefaultAirsimActionSettings = _DefaultAirsimActionSettings()
