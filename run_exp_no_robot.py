import sys

sys.path.extend(["pybullet_planning"])

import argparse
import random

import numpy as np
import pybullet as p
from pybullet_tools.utils import wait_if_gui

from bandu_stacking.env_no_robot import StackingEnvironment
from bandu_stacking.policies.random_policy import RandomPolicy

# from bandu_stacking.policies.skeleton_planner import SkeletonPlanner
from bandu_stacking.policies.skeleton_planner_improved import SkeletonPlannerImproved

algorithms = {"random": RandomPolicy, "skeleton_planner": SkeletonPlannerImproved}

object_sets = ["blocks", "bandu", "random"]


def create_args():
    """Creates the arguments for the experiment."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num-exps", default=1, type=int)
    parser.add_argument("--max-steps", default=5, type=int)
    parser.add_argument("--num-blocks", default=3, type=int)
    parser.add_argument("--sim-freq", default=0.01, type=float)
    parser.add_argument(
        "--use-robot",
        action="store_true",
        help="Find plans with IK, grasp, and collision constraints",
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

    # Set random seed for python and numpy
    random.seed(args.seed)
    np.random.seed(args.seed)

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
    # wait_if_gui(client=env.client)

    rewards = []
    for _ in range(args.num_exps):
        s = env.reset()
        # for _ in range(100):
        #     env.client.stepSimulation()
        # For the planning timestep (i.e., if policy.plan is None), load simplified object meshes for planning.
        # Else, load the full object meshes for execution.
        """
        Plan using simplified meshes
        """
        env.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        plan = policy.get_plan(s)
        policy.plan = plan
        env.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        """
        Load fullres meshes for execution
        """
        env.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # Sample mass, friction, restitution randomly
        sampled_mass = np.random.uniform(0.5, 1)
        sampled_friction = np.random.uniform(0.25, 0.8)
        sampled_restitution = np.random.uniform(0.1, 0.4)
        s = env.replace_with_fullres_meshes(
            sampled_mass, sampled_friction, sampled_restitution
        )
        env.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        for _ in range(args.max_steps):
            env.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            a = policy.get_action(s)
            env.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

            if a is None:
                break

            s = env.step(s, a, sim_freq=args.sim_freq)
        rewards.append(env.reward(s))

    # print("====================")
    # print("Experiments finished")
    # print("Method: " + str(args.algorithm))
    # print("Reward Mean: " + str(np.mean(rewards)))
    # print("Reward Std: " + str(np.std(rewards)))
    # print("====================")


if __name__ == "__main__":
    main()
