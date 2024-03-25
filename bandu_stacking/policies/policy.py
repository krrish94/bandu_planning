from __future__ import print_function

from dataclasses import dataclass, field
from typing import Dict, List

from bandu_stacking.env_utils import Conf
from bandu_stacking.pb_utils import Pose


@dataclass
class State:
    block_ids: List[int] = field(default_factory=lambda: [])
    block_poses: Dict[int, Pose] = field(default_factory=lambda: {})
    foundation: int = None
    robot_conf: Conf = None

    def __repr__(self):
        return f"State(block_ids={self.block_ids}, block_poses={self.block_poses})"


class Action:
    def __init__(self, grasp_block, target_block, pose):
        self.grasp_block = grasp_block
        self.target_block = target_block
        self.pose = pose

    def __repr__(self):
        return f"Action(grasp_block={self.grasp_block}, target_block={self.target_block}, pose={self.pose})"


class Policy:
    def __init__(self):
        pass

    def get_action(self, initial_state):
        raise NotImplementedError
