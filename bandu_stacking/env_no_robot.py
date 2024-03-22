import os
import random
import time
from os import listdir
from os.path import isfile, join

import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc

from bandu_stacking.bandu_utils import (
    create_pybullet_block,
    get_absolute_pose,
    pairwise_collision,
)
from bandu_stacking.env_utils import WorldState, create_default_env
from bandu_stacking.pb_utils import (
    AABB,
    Pose,
    get_aabb,
    get_aabb_extent,
    remove_body,
    wait_for_duration,
    wait_if_gui,
)
from bandu_stacking.policies.policy import State

TABLE_AABB = AABB(
    lower=(-1.53 / 2.0, -1.22 / 2.0, -0.03 / 2.0),
    upper=(1.53 / 2.0, 1.22 / 2.0, 0.03 / 2.0),
)
TABLE_POSE = Pose((0.1, 0, -TABLE_AABB.upper[2]))
DEFAULT_TS = 5e-3


def iterate_sequence(
    state, sequence, time_step=DEFAULT_TS, teleport=False, **kwargs
):  # None | INF
    assert sequence is not None
    for i, _ in enumerate(sequence.iterate(state, teleport=teleport, **kwargs)):
        state.propagate()
        if time_step is None:
            wait_if_gui()
        else:
            wait_for_duration(time_step)
    return state


def setup_client_pybullet():
    client = bc.BulletClient(connection_mode=p.GUI)

    return client


class StackingEnvironment:
    def __init__(
        self,
        object_set="blocks",
        num_blocks=5,
        gui=False,
        real_camera=False,
        real_execute=False,
    ):
        self.num_blocks = num_blocks
        self.object_set = object_set

        self.client = setup_client_pybullet()

        # self.robot = PandaRobot(
        #     robot_body,
        #     real_execute=real_execute,
        #     real_camera=real_camera,
        #     client=self.client,
        # )

        self.client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.client.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        self.table, self.obstacles = create_default_env(client=self.client)
        self.table_width, self.table_length, self.table_height = get_aabb_extent(
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
        self.offset_size = 0.0

        self.block_ids = []
        self.bounding_boxes = []
        self.urdf_filenames = []
        self.fullres_urdf_filenames = []

        self.block_size = 0.045

        if object_set == "blocks":
            for i in range(self.num_blocks):
                color = self._obj_colors[i % len(self._obj_colors)]
                half_extents = (
                    self.block_size / 2.0,
                    self.block_size / 2.0,
                    self.block_size / 2.0,
                )
                self.block_ids.append(
                    create_pybullet_block(
                        color,
                        half_extents,
                        self._obj_mass,
                        self._obj_friction,
                        self._obj_restitution,
                        self._default_orn,
                        client=self.client,
                    )
                )
                self.bounding_boxes.append(
                    get_aabb(self.block_ids[-1], client=self.client)
                )
        elif object_set == "bandu":
            self.block_ids, self.bounding_boxes = self.add_bandu_objects()
        elif object_set == "random":
            self.block_ids, self.bounding_boxes = self.add_random_objects()

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
        bounding_boxes = []
        block_ids = []
        bandu_model_path = os.path.join(
            os.path.dirname(__file__), "models", "bandu_simplified"
        )
        bandu_filenames = [
            f for f in listdir(bandu_model_path) if isfile(join(bandu_model_path, f))
        ]
        original_bandu_urdfs = [
            f for f in bandu_filenames if f.endswith("_original.urdf")
        ]
        bandu_urdfs = [
            f
            for f in bandu_filenames
            if f.endswith("urdf") and f not in original_bandu_urdfs
        ]
        for _ in range(self.num_blocks):
            bandu_urdf = os.path.join(bandu_model_path, random.choice(bandu_urdfs))
            obj = self.client.loadURDF(bandu_urdf, globalScaling=0.002)
            block_ids.append(obj)
            bounding_boxes.append(get_aabb(block_ids[-1]))
            self.urdf_filenames.append(bandu_urdf)
            # Also store the corresponding full resolution urdf filename for later use
            # (replace "_simplified" with "" in the filename)
            fullres_urdf = bandu_urdf.replace(".urdf", "_original.urdf")
            self.fullres_urdf_filenames.append(fullres_urdf)
            self.client.changeDynamics(
                block_ids[-1],
                -1,
                mass=self._obj_mass,
                lateralFriction=self._obj_friction,
                restitution=self._obj_restitution,
            )
        return block_ids, bounding_boxes

    def add_random_objects(self):
        bounding_boxes = []
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
            bounding_boxes.append(get_aabb(block_ids[-1], client=self.client))
        return block_ids, bounding_boxes

    def sample_action(self, mesh_dicts):
        blocks = random.sample(self.block_ids, 2)
        return self.sample_constrained_action(*blocks, mesh_dicts, best_face=True)

    def sample_state(self):
        if self.object_set == "bandu":
            for object_id in self.block_ids:
                remove_body(object_id, client=self.client)
            self.urdf_filenames = []
            self.fullres_urdf_filenames = []
            self.block_ids, self.bounding_boxes = self.add_bandu_objects()
        elif self.object_set == "random":
            for object_id in self.block_ids:
                self.client.removeBody(object_id)
            self.block_ids, self.bounding_boxes = self.add_random_objects()

        for block_index, block_id in enumerate(self.block_ids):
            found_collision_free = False
            timeout = 10
            padding = 0.1
            while not found_collision_free or timeout > 0:
                timeout -= 1

                rx = random.uniform(
                    self.table_width / 6.0, TABLE_POSE[0][0] + self.table_width / 3.0
                )
                ry = random.uniform(
                    TABLE_POSE[0][1] - self.table_length / 2.0 + padding,
                    TABLE_POSE[0][1] + self.table_length / 2.0 - padding,
                )
                self.client.resetBasePositionAndOrientation(
                    block_id,
                    [rx, ry, self.table_height + self.block_size / 2.0],
                    self._default_orn,
                )
                collision = False
                for placed_block in self.block_ids[:block_index]:
                    if pairwise_collision(
                        block_id, placed_block, client=self.client, max_distance=1e-2
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
        block_poses = []
        for block in self.block_ids:
            pose = self.client.getBasePositionAndOrientation(block)
            block_poses.append(pose)
        return State(self.block_ids, block_poses)

    def set_sim_state(self, state):
        for block_id, block_pose in zip(state.block_ids, state.block_poses):
            (point, quat) = block_pose
            self.client.resetBasePositionAndOrientation(block_id, point, quat)

    def state_diff(self, s1, s2):
        return sum(
            [
                np.linalg.norm(
                    np.array(s1.block_poses[i][0]) - np.array(s2.block_poses[i][0])
                )
                for i in range(self.num_blocks)
            ]
        )

    def simulate_until_static(self, sim_freq=0, max_iter=100):
        state_diff_thresh = 5e-4

        prev_state = self.state_from_sim()
        state_diff = float("inf")
        count = 0
        while state_diff > state_diff_thresh:
            for _ in range(5):
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
            if placed_block != grasp_block and pairwise_collision(
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

        print(action)
        target_pose = p.getBasePositionAndOrientation(action.target_block)
        rx = target_pose[0][0] + action.pose[0][0]
        ry = target_pose[0][1] + action.pose[0][1]
        rz = self.table_height + target_pose[0][2] + 0.1  # self.block_size / 2.0 + 0.1
        p.resetBasePositionAndOrientation(
            action.grasp_block, [rx, ry, rz], self._default_orn
        )
        self.simulate_until_static(sim_freq=sim_freq)

        # iterate_sequence(self.world_state, action)
        # self.simulate_until_static(sim_freq=1e-2)
        next_state = self.state_from_sim()
        self.world_state = WorldState(client=self.client)
        return next_state

    def replace_with_fullres_meshes(self, mass=0.5, friction=0.25, restitution=1.0):
        """Replace the simplified meshes with the full resolution meshes."""

        initial_state = self.state_from_sim()
        dynamics_properties = []
        positions_orientations = []
        for block_id in self.block_ids:
            pos, ori = self.client.getBasePositionAndOrientation(block_id)
            positions_orientations.append((pos, ori))
            dynamics_properties.append(self.client.getDynamicsInfo(block_id, -1))
        for obj_id in reversed(self.block_ids):
            self.client.removeBody(obj_id)
            # remove bounding box info
            self.bounding_boxes.pop(-1)
        for fullres_urdf in self.fullres_urdf_filenames:
            properties = dynamics_properties.pop(0)
            pos_ori = positions_orientations.pop(0)
            block_id = self.client.loadURDF(
                fullres_urdf,
                globalScaling=0.002,
                basePosition=pos_ori[0],
                baseOrientation=pos_ori[1],
            )
            self.block_ids.append(block_id)
            self.bounding_boxes.append(get_aabb(self.block_ids[-1], client=self.client))

            self.client.changeDynamics(
                self.block_ids[-1],
                -1,
                mass=mass,
                lateralFriction=friction,
                restitution=restitution,
            )
        self.simulate_until_static()
        self.world_state = WorldState(client=self.client)
        state = self.state_from_sim()
        return state

    def reset(self):
        # self.robot.reset()
        initial_state = self.sample_state()
        self.set_sim_state(initial_state)
        self.simulate_until_static()
        state = self.state_from_sim()
        self.world_state = WorldState(client=self.client)
        return state

    def reward(self, state):
        """Max z value of any object."""
        return max([bp[0][-1] for bp in state.block_poses])
