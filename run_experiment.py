import argparse
import datetime

import numpy as np
import pybullet as p

from bandu_stacking.env import StackingEnvironment
from bandu_stacking.pb_utils import wait_if_gui
from bandu_stacking.policies.random_policy import RandomPolicy
from bandu_stacking.policies.skeleton_planner import SkeletonPlanner

algorithms = {"random": RandomPolicy, "skeleton_planner": SkeletonPlanner}

object_sets = ["blocks", "bandu", "random"]


def create_args():
    """Creates the arguments for the experiment."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-exps", default=100, type=int)
    parser.add_argument("--max-steps", default=5, type=int)
    parser.add_argument("--num-blocks", default=3, type=int)
    parser.add_argument("--sim-freq", default=0.01, type=float)
    parser.add_argument(
        "--disable-robot",
        action="store_true",
        help="Do not load in the robot or do any constraint satisfaction planning",
    )
    parser.add_argument(
        "--real-execute", action="store_true", help="Execute on the real robot"
    )
    parser.add_argument(
        "--real-camera", action="store_true", help="Use real camera data"
    )
    parser.add_argument(
        "--disable-gui", action="store_true", help="View the pybullet gui when planning"
    )
    parser.add_argument("--object-set", default="blocks", choices=object_sets)
    parser.add_argument(
        "--algorithm", default="skeleton_planner", choices=list(algorithms.keys())
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record the execution and store as an MP4",
    )
    args = parser.parse_args()
    return args


def main():
    args = create_args()

    if args.real_execute or args.real_camera:
        raise NotImplementedError

    env = StackingEnvironment(
        object_set=args.object_set,
        num_blocks=args.num_blocks,
        disable_gui=args.disable_gui,
        disable_robot=args.disable_robot,
    )

    policy = algorithms[args.algorithm](env)
    env.client.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=90,
        cameraPitch=-15,
        cameraTargetPosition=[-0.5, 0, 0],
    )

    rewards = []

    if args.record_video:
        # Get the current date and time
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create a video recorder object
        video_recorder = env.client.startStateLogging(
            p.STATE_LOGGING_VIDEO_MP4, f"generated_videos/{timestamp}.mp4"
        )

    for _ in range(args.num_exps):
        s = env.reset()
        wait_if_gui(client=env.client)

        for _ in range(100):
            env.client.stepSimulation()
        for _ in range(args.max_steps):
            # env.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            a = policy.get_action(s)
            # env.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

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

    env.client.stopStateLogging(video_recorder)


if __name__ == "__main__":
    main()
