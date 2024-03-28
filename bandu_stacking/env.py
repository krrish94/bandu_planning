from __future__ import print_function

import copy
import os
import random
import time
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import bandu_stacking.pb_utils as pbu
from bandu_stacking.env_utils import (
    ARM_GROUP,
    PANDA_PATH,
    TABLE_AABB,
    TABLE_POSE,
    PandaRobot,
    WorldState,
    create_default_env,
    create_pybullet_block,
    get_absolute_pose,
    get_current_confs,
)
from bandu_stacking.policies.policy import State
from bandu_stacking.realsense_utils import CALIB_DIR, CAMERA_SNS, get_camera_image
from bandu_stacking.vision_utils import save_camera_images, get_seg_sam,mask_roi, load_sam

BANDU_PATH = os.path.join(os.path.dirname(__file__), "models", "bandu_simplified")

DEFAULT_TS = 5e-3
BANDU_SCALING = 0.002


def iterate_sequence(
    state, sequence, time_step=DEFAULT_TS, teleport=False, **kwargs
):  # None | INF
    assert sequence is not None
    for i, _ in enumerate(sequence.iterate(state, teleport=teleport, **kwargs)):
        state.propagate(**kwargs)
        if time_step is None:
            pbu.wait_if_gui(**kwargs)
        else:
            pbu.wait_for_duration(time_step)
    return state


class StackingEnvironment:
    def __init__(
        self,
        object_set="blocks",
        num_blocks=5,
        disable_gui=True,
        disable_robot=False,
        real_camera=False,
        real_execute=False,
    ):

        self.num_blocks = num_blocks
        self.object_set = object_set
        self.disable_robot = disable_robot
        self.disable_gui = disable_gui
        self.real_camera = real_camera

        if not self.disable_gui:
            self.client = bc.BulletClient(connection_mode=p.GUI)
        else:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)

        self.client.setGravity(0, 0, -9.8)

        if not self.disable_robot:
            robot_body = pbu.load_pybullet(
                PANDA_PATH, fixed_base=True, client=self.client
            )
            self.robot = PandaRobot(
                robot_body,
                real_execute=real_execute,
                real_camera=real_camera,
                client=self.client,
            )

        self.client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.client.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        self.table, self.obstacles = create_default_env(client=self.client)
        self.table_width, self.table_length, self.table_height = pbu.get_aabb_extent(
            TABLE_AABB
        )

        # Option parameters.
        self._offset_z = 0.01

        # Object parameters.
        self._obj_mass = 0.5
        self._obj_friction = 1.2
        self._obj_restitution = 0.8
        self._obj_colors = [
            (0.95, 0.05, 0.1, 1.0),
            (0.05, 0.95, 0.1, 1.0),
            (0.1, 0.05, 0.95, 1.0),
            (0.4, 0.05, 0.6, 1.0),
            (0.6, 0.4, 0.05, 1.0),
            (0.05, 0.04, 0.6, 1.0),
            (0.95, 0.95, 0.1, 1.0),
            (0.95, 0.05, 0.95, 1.0),
            (0.05, 0.95, 0.95, 1.0),
        ]
        self._default_orn = [0.0, 0.0, 0.0, 1.0]
        self.offset_size = 0.04

        self.block_ids = []

        self.foundation = self.client.loadURDF(
            os.path.join(BANDU_PATH, "foundation.urdf"),
            globalScaling=BANDU_SCALING,
            useFixedBase=True,
        )

        self.block_size = 0.045
        if self.real_camera:
            self.sam = load_sam()

            for camera_sn in CAMERA_SNS:
                base_T_camera = np.load(
                    os.path.join(CALIB_DIR, f"{camera_sn}/pose.npy")
                )
                camera_image = get_camera_image(camera_sn, base_T_camera)

                
                camera_image = mask_roi(camera_sn, camera_image)

                masks = get_seg_sam(self.sam, camera_image.rgbPixels)
                save_camera_images(camera_image, prefix=camera_sn)

        elif object_set == "blocks":
            for i in range(self.num_blocks):
                color = self._obj_colors[i % len(self._obj_colors)]
                half_extents = (
                    self.block_size / 2.0,
                    self.block_size / 2.0,
                    self.block_size / 2.0,
                )
                block = create_pybullet_block(
                    color,
                    half_extents,
                    self._obj_mass,
                    self._obj_friction,
                    self._obj_restitution,
                    self._default_orn,
                    client=self.client,
                )
                self.block_ids.append(block)

        elif object_set == "bandu":
            self.block_ids = self.add_bandu_objects()
        elif object_set == "random":
            self.block_ids = self.add_random_objects()

        # Camera parameters.
        self._camera_distance = 0.8
        self._camera_yaw = 90.0
        self._camera_pitch = -24
        self._camera_target = (1.65, 0.75, 0.42)
        self._debug_text_position = (1.65, 0.25, 0.75)
        self.client.resetDebugVisualizerCamera(
            self._camera_distance,
            self._camera_yaw,
            self._camera_pitch,
            self._camera_target,
        )

    def add_bandu_objects(self):
        block_ids = []

        bandu_filenames = [
            f for f in listdir(BANDU_PATH) if isfile(join(BANDU_PATH, f))
        ]
        bandu_urdfs = [
            f
            for f in bandu_filenames
            if f.endswith("urdf") and "original" not in f and "foundation" not in f
        ]
        for i in range(self.num_blocks):
            bandu_urdf = os.path.join(BANDU_PATH, random.choice(bandu_urdfs))
            obj = self.client.loadURDF(bandu_urdf, globalScaling=BANDU_SCALING)
            block_ids.append(obj)
        return block_ids

    def add_random_objects(self):

        block_ids = []

        random_model_path = os.path.join(
            os.path.dirname(__file__), "models", "random_models"
        )
        random_filenames = [
            f for f in listdir(random_model_path) if isfile(join(random_model_path, f))
        ]
        random_urdfs = [f for f in random_filenames if f.endswith("urdf")]
        for i in range(self.num_blocks):
            random_urdf = os.path.join(random_model_path, random.choice(random_urdfs))
            obj = self.client.loadURDF(random_urdf, globalScaling=0.1)
            block_ids.append(obj)
        return block_ids

    def sample_action(self, mesh_dicts):
        blocks = random.sample(self.block_ids, 2)
        return self.sample_constrained_action(*blocks, mesh_dicts)

    def sample_state(self):

        # Add foundation object
        self.client.removeBody(self.foundation)
        self.foundation = self.client.loadURDF(
            os.path.join(BANDU_PATH, "foundation.urdf"),
            globalScaling=BANDU_SCALING,
            useFixedBase=True,
        )
        self.foundation_pose = list(copy.deepcopy(TABLE_POSE))
        self.foundation_pose[0] = list(self.foundation_pose[0])
        self.foundation_pose[0][2] = (
            pbu.get_aabb_extent(pbu.get_aabb(self.foundation, client=self.client))[2]
            / 2.0
        )
        self.foundation_pose[0][0] += 0.05
        self.foundation_pose[1] = pbu.quat_from_euler(pbu.Euler(yaw=np.pi / 2.0))
        pbu.set_pose(self.foundation, self.foundation_pose, client=self.client)
        pbu.wait_if_gui(client=self.client)

        # Add additional objects
        if self.object_set == "bandu":
            for object_id in self.block_ids:
                self.client.removeBody(object_id)
            self.block_ids = self.add_bandu_objects()
        elif self.object_set == "random":
            for object_id in self.block_ids:
                self.client.removeBody(object_id)
            self.block_ids = self.add_random_objects()

        for block_index, block_id in enumerate(self.block_ids):
            found_collision_free = False
            timeout = 10
            x_padding = 0.15
            y_padding = 0.2
            while not found_collision_free or timeout > 0:
                timeout -= 1

                rx = random.uniform(
                    TABLE_POSE[0][0] - self.table_width / 2.0 + x_padding,
                    TABLE_POSE[0][0] + self.table_width / 2.0 - x_padding,
                )
                ry = random.uniform(
                    TABLE_POSE[0][1] - self.table_length / 2.0 + y_padding,
                    TABLE_POSE[0][1] + self.table_length / 2.0 - y_padding,
                )
                self.client.resetBasePositionAndOrientation(
                    block_id,
                    [rx, ry, self.table_height + self.block_size / 2.0],
                    self._default_orn,
                )
                collision = False
                for placed_block in self.block_ids[:block_index] + [self.foundation]:
                    if pbu.pairwise_collision(
                        block_id, placed_block, client=self.client, max_distance=0.03
                    ):
                        collision = True
                        break
                if not collision:
                    break

            if timeout <= 0:
                print("Timeout setting block positions")
                assert False

        return self.state_from_sim()

    def state_from_sim(self):
        block_poses = {self.foundation: self.foundation_pose}
        for block in self.block_ids:
            pose = self.client.getBasePositionAndOrientation(block)
            block_poses[block] = pose

        current_conf = get_current_confs(self.robot, client=self.client)[ARM_GROUP]
        return State(
            self.block_ids,
            block_poses,
            foundation=self.foundation,
            robot_conf=current_conf,
        )

    def set_sim_state(self, state: State):
        for block_id, block_pose in state.block_poses.items():
            (point, quat) = block_pose
            self.client.resetBasePositionAndOrientation(block_id, point, quat)
        if state.robot_conf is not None:
            state.robot_conf.assign(client=self.client)

    def state_diff(self, s1, s2):
        return sum(
            [
                np.linalg.norm(
                    np.array(s1.block_poses[k][0]) - np.array(s2.block_poses[k][0])
                )
                for k in s1.block_poses.keys()
            ]
        )

    def simulate_until_static(self, sim_freq=0, max_iter=50):
        state_diff_thresh = 5e-4

        prev_state = self.state_from_sim()
        state_diff = float("inf")
        count = 0
        while state_diff > state_diff_thresh:
            for _ in range(10):
                time.sleep(sim_freq)
                self.client.stepSimulation()
            current_state = self.state_from_sim()
            state_diff = self.state_diff(prev_state, current_state)
            prev_state = current_state
            count += 1
            if count >= max_iter:
                break

    def in_contact(self, grasp_block):
        for placed_block in self.block_ids:
            if placed_block != grasp_block and pbu.pairwise_collision(
                grasp_block, placed_block, client=self.client
            ):
                return True
        return False

    def execute(self, action, sim_freq=0.0):
        # Don't execute the action if the block starts in contact with another block
        if self.in_contact(action.grasp_block):
            return

        target_pose = self.client.getBasePositionAndOrientation(action.target_block)
        pose = get_absolute_pose(target_pose, action)
        self.client.resetBasePositionAndOrientation(action.grasp_block, *pose)
        self.simulate_until_static(sim_freq=sim_freq)

    def step_abstract(self, state, abstract_action, sim_freq=0):
        self.set_sim_state(state)
        self.execute(abstract_action, sim_freq=sim_freq)
        next_state = self.state_from_sim()
        return next_state

    def step(self, state, action, sim_freq=0):
        self.set_sim_state(state)
        self.world_state.assign()
        iterate_sequence(self.world_state, action, client=self.client)
        self.simulate_until_static(sim_freq=1e-2)
        next_state = self.state_from_sim()
        self.world_state = WorldState(client=self.client)
        return next_state

    def reset(self):
        self.robot.reset(client=self.client)
        initial_state = self.sample_state()
        self.set_sim_state(initial_state)
        self.simulate_until_static()
        state = self.state_from_sim()
        self.world_state = WorldState(client=self.client)
        return state

    def reward(self, state):
        """Max z value of any object."""
        return max([bp[0][-1] for bp in state.block_poses.values()])
