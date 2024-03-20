from __future__ import print_function

from policies.planning.planners.main import Planner
from policies.simulation.environment import (
    create_ycb,
)
from robots.panda.panda_utils import PANDA_PATH, PandaRobot

from pybullet_planning.pybullet_tools.utils import set_pose


def reset_robot(robot):
    conf = robot.get_default_conf()
    for group, positions in conf.items():
        robot.set_group_positions(group, positions)


class Hardcode(Planner):
    def __init__(self):
        pass

    def plan(self, robot, belief, goal):
        local_robot, table, client = self.world_twin(robot, belief)

        # Add in the objects
        segments = belief.segments[0]
        objects = []
        for segment in segments:
            category, pose = segment.category, segment.pose
            obj = create_ycb(category, client=client)
            set_pose(obj, pose, client=client)
            objects.append(obj)

        obj = objects[0]

        return self.get_pick_plan(local_robot, obj, client), belief
