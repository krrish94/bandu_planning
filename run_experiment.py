import argparse
import datetime
import logging
import os
import sys
import time

import numpy as np
import pybullet as p

from bandu_stacking.env import StackingEnvironment
from bandu_stacking.pb_utils import wait_if_gui
from bandu_stacking.policies.random_policy import RandomPolicy
from bandu_stacking.policies.skeleton_planner import SkeletonPlanner

algorithms = {"random": RandomPolicy, "skeleton_planner": SkeletonPlanner}

object_sets = ["blocks", "bandu", "random"]


class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def setup_logger(save_dir):
    # Create a logs folder if it doesn't exist
    if not os.path.exists(os.path.join(save_dir, "logs")):
        os.makedirs(os.path.join(save_dir, "logs"))

    log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(save_dir, "logs", f"{time.time()}.log"))
        ],
    )

    logger = logging.getLogger()

    # Add StreamHandler to logger to output logs to stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Redirect stdout and stderr
    sys.stdout = StreamToLogger(logger, log_level)
    sys.stderr = StreamToLogger(logger, logging.ERROR)


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
        "--save-dir",
        help="File to load from",
        default="logs/run{}".format(str(time.time())),
    )
    parser.add_argument(
        "--use-previous-pointclouds",
        action="store_true",
        help="Execute on the real robot",
    )
    parser.add_argument(
        "--use-sbi", action="store_true", help="Execute on the real robot"
    )
    parser.add_argument(
        "--real-camera", action="store_true", help="Use real camera data"
    )
    parser.add_argument(
        "--real-execute", action="store_true", help="Execute on the real robot"
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

    setup_logger(args.save_dir)

    env = StackingEnvironment(
        object_set=args.object_set,
        num_blocks=args.num_blocks,
        disable_gui=args.disable_gui,
        disable_robot=args.disable_robot,
        real_camera=args.real_camera,
        real_execute=args.real_execute,
        use_previous_pointclouds=args.use_previous_pointclouds,
        save_dir=args.save_dir,
    )

    policy = algorithms[args.algorithm](env, use_sbi=args.use_sbi)
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
            p.STATE_LOGGING_VIDEO_MP4,
            os.path.join(args.save_dir, f"generated_videos/{timestamp}.mp4"),
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
