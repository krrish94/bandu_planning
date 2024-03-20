import math
import random
from collections import defaultdict, namedtuple

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from bandu_stacking.bandu_utils import check_collision, mesh_from_obj
from bandu_stacking.env import TABLE_AABB, get_absolute_pose
from bandu_stacking.env_utils import (
    GroupConf,
    RelativePose,
    Sequence,
)
from bandu_stacking.pb_utils import (
    Euler,
    Point,
    Pose,
    get_aabb,
    get_pose,
    stable_z_on_aabb,
)
from bandu_stacking.policies.policy import Action, Policy
from bandu_stacking.streams import (
    get_grasp_gen_fn,
    get_plan_motion_fn,
    get_plan_pick_fn,
    get_plan_place_fn,
)

OOBB = namedtuple("OOBB", ["aabb", "pose"])


def aabb_height(aabb):
    return aabb[1][2] - aabb[0][2]


def get_initial_configurations(robot, client):
    return {
        group: GroupConf(robot, group, important=True, client=client)
        for group in robot.groups
    }


def get_random_placement_pose(obj, surface_aabb, client):
    return Pose(
        point=Point(
            x=random.uniform(surface_aabb.lower[0], surface_aabb.upper[0]),
            y=random.uniform(surface_aabb.upper[1], surface_aabb.upper[1]),
            z=stable_z_on_aabb(obj, surface_aabb, client=client),
        ),
        euler=Euler(yaw=random.uniform(0, math.pi * 2)),
    )


def find_pick_plan(GROUP, obj, pose, grasp, base_conf, pick_planner):
    pick = None
    while not pick:
        pick = pick_planner(GROUP, obj, pose, grasp, base_conf)
    return pick


def find_place_plan(GROUP, obj, place_pose, grasp, base_conf, place_planner, client):
    place = None
    while not place:
        place_rp = RelativePose(
            obj, parent=None, relative_pose=place_pose, client=client
        )
        place = place_planner(GROUP, obj, place_rp, grasp, base_conf)
    return place


def find_motion_plan(GROUP, start_conf, end_conf, motion_planner, attachments=[]):
    motion_plan = None
    while not motion_plan:
        (motion_plan,) = motion_planner(
            GROUP, start_conf, end_conf, attachments=attachments
        )
    return motion_plan


def get_pick_place_plan(abstract_action, env):
    client, robot = env.client, env.robot
    surface_aabb = TABLE_AABB
    GROUP = "main_arm"
    obj = abstract_action.grasp_block
    obj_aabb, obj_pose = get_aabb(obj, client=client), get_pose(obj, client=client)

    target_pose = client.getBasePositionAndOrientation(abstract_action.target_block)
    placement_pose = get_absolute_pose(target_pose, abstract_action)

    motion_planner = get_plan_motion_fn(robot, environment=env.block_ids, client=client)
    pick_planner, place_planner = get_plan_pick_fn(
        robot, client=client
    ), get_plan_place_fn(robot, client=client)
    grasp_finder = get_grasp_gen_fn(
        robot, env.block_ids, grasp_mode="top", client=client
    )

    init_confs = get_initial_configurations(robot, client)
    pose, base_conf = RelativePose(obj, client=client), init_confs["base"]
    q1 = init_confs[GROUP]

    (grasp,) = next(grasp_finder(GROUP, obj, obj_aabb, obj_pose))

    pick = find_pick_plan(GROUP, obj, pose, grasp, base_conf, pick_planner)
    q2, at1 = pick

    if placement_pose is None:
        placement_pose = get_random_placement_pose(obj, surface_aabb, client)

    place = find_place_plan(
        GROUP, obj, placement_pose, grasp, base_conf, place_planner, client
    )
    q3, at2 = place

    _, _, tool_name = robot.manipulators[robot.side_from_arm(GROUP)]
    attachment = grasp.create_attachment(robot, link=robot.link_from_name(tool_name))

    motion_plan1 = find_motion_plan(GROUP, q1, q2, motion_planner)
    motion_plan2 = find_motion_plan(
        GROUP, q2, q3, motion_planner, attachments=[attachment]
    )

    env.robot.remove_components()

    return placement_pose, Sequence([motion_plan1, at1, motion_plan2, at2])


class SkeletonPlanner(Policy):
    def __init__(self, env):
        self.env = env
        self.plan = None
        super(SkeletonPlanner, self).__init__()

    def planning_heuristic(self, initial_state, plan):
        """Get the height of the tower if this plan were to succeed."""
        TABLE = -1
        on_dict = defaultdict(
            lambda: TABLE, {action.grasp_block: action.target_block for action in plan}
        )
        height_dict = {TABLE: self.env.table_height}

        def get_height(obj):
            if obj in height_dict.keys():
                return height_dict[obj]
            else:
                height = aabb_height(
                    self.env.bounding_boxes[self.env.block_ids.index(obj)]
                )
                height_dict[obj] = get_height(on_dict[obj]) + height
                return height_dict[obj]

        return max([get_height(obj) for obj in initial_state.block_ids])

    def get_plan_skeletons(self, initial_state, mesh_info, num_skeletons=1000):
        """This function is typically implemented pddl+fast downward top-k
        planner, but is hardcoded here with a top-k random planner for
        simplicity.

        A plan skeleton is a sequence of symbolic block stacking
        actions. Multiple objects can be stacked on a single object, and
        even if one object is symbolically stacked on another object, it
        can be resting on two objects based on the continuous
        parameters.
        """

        plan_skeletons = []
        while len(plan_skeletons) < num_skeletons:
            # Sample a random plan length
            plan_length = random.choice(range(1, len(initial_state.block_ids)))
            skeleton = []
            state = initial_state
            collision = False
            for _ in range(plan_length):
                # sample target object
                assert len(initial_state.block_ids) >= 2
                target_object = random.choice(initial_state.block_ids)

                source_options = list(
                    set(initial_state.block_ids)
                    - set(
                        [target_object]
                        + [p.grasp_block for p in skeleton]
                        + [p.target_block for p in skeleton]
                    )
                )
                if len(source_options) == 0:
                    break

                source_object = random.choice(source_options)

                action = self.sample_constrained_action(
                    source_object, target_object, mesh_info
                )
                if check_collision(state, action, client=self.env.client):
                    collision = True
                    break

                skeleton.append(action)

            if not collision:
                plan_skeletons.append(skeleton)
        return plan_skeletons

    def sample_constrained_action(
        self, source_obj, target_obj, mesh_info, best_face=False
    ):
        if best_face:
            # Place the object on its largest face
            mesh_dict = mesh_info[source_obj]
            best_face = mesh_dict["face_sizes"][0][0]
            normal = mesh_dict["mesh"].face_normals[best_face]
        else:
            # Place all objects facing upward
            normal = [0, 0, 1]

        def get_rotation_matrix(vec2, vec1=np.array([1, 0, 0])):
            """Get rotation matrix between two vectors using scipy."""
            vec1 = np.reshape(vec1, (1, -1))
            vec2 = np.reshape(vec2, (1, -1))
            r = R.align_vectors(vec2, vec1)
            return r[0].as_quat()

        vec2 = [0, 0, 1]
        r = get_rotation_matrix(normal, vec2)

        x_offset = random.uniform(-self.env.offset_size, self.env.offset_size)
        y_offset = random.uniform(-self.env.offset_size, self.env.offset_size)
        pose = [(x_offset, y_offset, self.env.block_size + 0.01), r]
        return Action(source_obj, target_obj, pose)

    def get_plan(self, initial_state):
        # Cache all of the bandu objects, their meshes, and sort the faces sizes
        mesh_info = {}
        for block_id in initial_state.block_ids:
            block_dict = {}
            vertices, faces = mesh_from_obj(block_id, client=self.env.client)
            mesh = trimesh.Trimesh(vertices, faces)
            mesh.fix_normals()
            face_sizes = sorted(
                [(i, area) for (i, area) in enumerate(mesh.area_faces)],
                key=lambda x: x[1],
                reverse=True,
            )
            block_dict["face_sizes"] = face_sizes
            block_dict["mesh"] = mesh
            block_dict["vertices"] = vertices
            block_dict["faces"] = faces
            mesh_info[block_id] = block_dict

        # Find k plan skeletons with correct collision free bounding boxes
        plan_skeletons = self.get_plan_skeletons(initial_state, mesh_info)

        # Sort by final state reward
        plan_skeletons = sorted(
            plan_skeletons,
            key=lambda plan: self.planning_heuristic(initial_state, plan),
            reverse=True,
        )

        # For each skeleton, sample action parameters and evaluate in simulation
        for plan_skeleton in tqdm(plan_skeletons):
            self.env.set_sim_state(initial_state)
            concrete_plan = []
            diff_thresh = 0.005
            fail = False
            # Sample continuous parameters, test tower height, fail if height not expected
            for ai, action in enumerate(plan_skeleton):
                self.env.execute(action)
                new_state = self.env.state_from_sim()
                bid = initial_state.block_ids.index(action.grasp_block)
                tid = initial_state.block_ids.index(action.target_block)

                np.array(new_state.block_poses[bid][0][:2])
                adiff = np.linalg.norm(
                    np.array(new_state.block_poses[tid][0][:2])
                    + np.array(action.pose[0][:2])
                    - np.array(new_state.block_poses[bid][0][:2])
                )
                if adiff > diff_thresh:
                    fail = True
                    break
                else:
                    concrete_plan.append(action)

            if not fail:
                self.env.set_sim_state(initial_state)
                return concrete_plan

    def get_action(self, initial_state):
        if self.plan == None:
            abstract_actions = self.get_plan(initial_state)
            current_state = initial_state
            self.plan = []
            for aa in abstract_actions:
                self.plan.append(get_pick_place_plan(aa, env=self.env)[1])
                current_state = self.env.step_abstract(current_state, aa)

        elif len(self.plan) == 0:
            # No more actions left, terminate the policy
            self.plan = None
            return None

        next_action = self.plan[0]
        self.plan = self.plan[1:]
        return next_action
