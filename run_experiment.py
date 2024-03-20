import argparse

import numpy as np
import pybullet as p
from pybullet_tools.utils import wait_if_gui

from bandu_stacking.env import StackingEnvironment
from bandu_stacking.policies.random_policy import RandomPolicy

# from bandu_stacking.policies.skeleton_planner import SkeletonPlanner
from bandu_stacking.policies.skeleton_planner_improved import SkeletonPlannerImproved

algorithms = {"random": RandomPolicy, "skeleton_planner": SkeletonPlannerImproved}

object_sets = ["blocks", "bandu", "random"]


def create_args():
    """Creates the arguments for the experiment."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-exps", default=100, type=int)
    parser.add_argument("--max-steps", default=5, type=int)
    parser.add_argument("--num-blocks", default=3, type=int)
    parser.add_argument("--sim-freq", default=0.01, type=float)
    parser.add_argument(
        "--use-robot",
        action="store_true",
        help="Find plans with IK, grasp, and collision constraints",
    )
    parser.add_argument(
        "--real-execute", action="store_true", help="Execute on the real robot"
    )
    parser.add_argument(
        "--real-camera", action="store_true", help="Use real camera data"
    )
    parser.add_argument(
        "--vis", action="store_false", help="View the pybullet gui when planning"
    )
    parser.add_argument("--object-set", default="blocks", choices=object_sets)
    parser.add_argument(
        "--algorithm", default="skeleton_planner", choices=list(algorithms.keys())
    )
    args = parser.parse_args()
    return args


def main():
    args = create_args()

    if args.real_execute or args.real_camera:
        raise NotImplementedError

    env = StackingEnvironment(
        object_set=args.object_set, num_blocks=args.num_blocks, gui=args.vis
    )

    policy = algorithms[args.algorithm](env)
    env.client.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=90,
        cameraPitch=-15,
        # cameraTargetPosition=[-0.5, 0, 0.3],
        cameraTargetPosition=[-0.5, 0, 0],
    )
    wait_if_gui(client=env.client)

    rewards = []
    for _ in range(args.num_exps):
        s = env.reset()
        for _ in range(100):
            env.client.stepSimulation()
        for _ in range(args.max_steps):
            env.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            a = policy.get_action(s)
            env.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

            if a is None:
                break

            s = env.step(s, a, sim_freq=args.sim_freq)
        rewards.append(env.reward(s))

    print("====================")
    print("Experiments finished")
    print("Method: " + str(args.algorithm))
    print("Reward Mean: " + str(np.mean(rewards)))
    print("Reward Std: " + str(np.std(rewards)))
    print("====================")


if __name__ == "__main__":
    main()
