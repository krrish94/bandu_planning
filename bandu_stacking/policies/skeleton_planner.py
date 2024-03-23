from __future__ import print_function

import math
import random
import sys
import time

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import bandu_stacking.pb_utils as pbu
from bandu_stacking.env import TABLE_AABB, get_absolute_pose
from bandu_stacking.env_utils import (
    ARM_GROUP,
    PANDA_GROUPS,
    PANDA_TOOL_TIP,
    GroupConf,
    PandaRobot,
    RelativePose,
    Sequence,
    check_collision,
    mesh_from_obj,
)
from bandu_stacking.policies.policy import Action, Policy
from bandu_stacking.streams import (
    get_grasp_gen_fn,
    get_plan_motion_fn,
    get_plan_pick_fn,
    get_plan_place_fn,
)


def aabb_height(aabb: pbu.AABB):
    return aabb.upper[2] - aabb.lower[2]


def get_current_confs(robot: PandaRobot, **kwargs):

    return {
        group: GroupConf(
            robot,
            group,
            pbu.get_joint_positions(
                robot, robot.get_group_joints(group, **kwargs), **kwargs
            ),
            important=True,
            **kwargs,
        )
        for group in PANDA_GROUPS
    }


def get_random_placement_pose(obj, surface_aabb, client):
    return pbu.Pose(
        point=pbu.Point(
            x=random.uniform(surface_aabb.lower[0], surface_aabb.upper[0]),
            y=random.uniform(surface_aabb.upper[1], surface_aabb.upper[1]),
            z=pbu.stable_z_on_aabb(obj, surface_aabb, client=client),
        ),
        euler=pbu.Euler(yaw=random.uniform(0, math.pi * 2)),
    )


def get_pick_place_plan(abstract_action, env):

    MAX_GRASP_ATTEMPTS = 10
    MAX_PICK_ATTEMPTS = 10
    MAX_PLACE_ATTEMPTS = 10

    client, robot = env.client, env.robot
    client = env.client
    surface_aabb = TABLE_AABB

    obj = abstract_action.grasp_block
    obj_aabb, obj_pose = pbu.get_aabb(obj, client=client), pbu.get_pose(
        obj, client=client
    )

    target_pose = client.getBasePositionAndOrientation(abstract_action.target_block)
    placement_pose = get_absolute_pose(target_pose, abstract_action)

    obstacles = env.block_ids + [env.table]

    motion_planner = get_plan_motion_fn(robot, environment=obstacles, client=client)
    pick_planner = get_plan_pick_fn(
        robot, environment=obstacles, max_attempts=MAX_PICK_ATTEMPTS, client=client
    )
    place_planner = get_plan_place_fn(
        robot, environment=obstacles, max_attempts=MAX_PLACE_ATTEMPTS, client=client
    )

    grasp_finder = get_grasp_gen_fn(
        robot, environment=obstacles, grasp_mode="mesh", client=client
    )

    init_confs = get_current_confs(robot, client=client)
    pose, base_conf = RelativePose(obj, client=client), init_confs["base"]
    q1 = init_confs[ARM_GROUP]
    pose = RelativePose(obj, client=client)

    for gi in range(MAX_GRASP_ATTEMPTS):
        print("[Planner] grasp attempt " + str(gi))
        (grasp,) = next(grasp_finder(obj, obj_aabb, obj_pose))

        print("[Planner] finding pick plan")
        for _ in range(MAX_PICK_ATTEMPTS):
            pick = pick_planner(obj, pose, grasp, base_conf)

        if pick is None:
            continue

        q2, at1 = pick

        print("[Planner] finding place plan")
        for _ in range(MAX_PLACE_ATTEMPTS):
            if placement_pose is None:
                placement_pose = get_random_placement_pose(obj, surface_aabb, client)
            place_rp = RelativePose(
                obj, parent=None, relative_pose=placement_pose, client=client
            )
            print("[Place Planner] Placement for pose: " + str(place_rp))
            place = place_planner(obj, place_rp, grasp, base_conf)

            if place is not None:
                break

        if place is None:
            continue

        q3, at2 = place

        attachment = grasp.create_attachment(
            robot, link=pbu.link_from_name(robot, PANDA_TOOL_TIP, client=client)
        )

        print("[Planner] finding pick motion plan")
        motion_plan1 = motion_planner(q1, q2)
        if motion_plan1 is None:
            continue

        print("[Planner] finding place motion plan")
        motion_plan2 = motion_planner(q2, q3, attachments=[attachment])
        if motion_plan2 is None:
            continue

        env.robot.remove_components(client=client)

        return Sequence([motion_plan1, at1, motion_plan2, at2])
    return None


class SkeletonPlanner(Policy):
    def __init__(self, env):
        self.env = env
        self.plan = None
        self.use_sbi = False
        super().__init__()

    def planning_heuristic(self, initial_state, plan):
        """Get the height of the tower if this plan were to succeed."""
        return 1

    def sample_constrained_action(
        self, state, source_obj, target_obj, mesh_info, best_face=False, **kwargs
    ):
        """Sample a pick and place action with the source object on the target
        object."""
        if best_face:
            # Place the object on its largest face
            mesh_dict = mesh_info[source_obj]
            best_face = mesh_dict["face_sizes"][0][0]
            normal = mesh_dict["mesh"].face_normals[best_face]
        else:
            # Place all objects facing upward
            normal = [0, 0, 1]

        # Pick a random face among the top (upto) 10 largest faces
        mesh_dict = mesh_info[source_obj]
        # recall that face_sizes may not contain upto 10 faces
        # if the mesh has less than 10 faces
        num_faces = min(10, len(mesh_dict["face_sizes"]))
        # face_sizes is sorted in descending order
        # so we pick a random face from the num_faces largest faces
        face_index = random.randint(0, num_faces - 1)
        face = mesh_dict["face_sizes"][face_index][0]
        normal = mesh_dict["mesh"].face_normals[face]

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

        pbu.set_pose(source_obj, [pbu.unit_point(), r], **kwargs)
        pbu.set_pose(target_obj, state.block_poses[target_obj], **kwargs)
        ae1 = pbu.get_aabb_extent(pbu.get_aabb(source_obj, **kwargs))
        ae2 = pbu.get_aabb(target_obj, **kwargs)
        pose = [(x_offset, y_offset, ae2.upper[2] + ae1[2] / 2.0), r]

        return Action(source_obj, target_obj, pose)

    def get_plan_skeletons(self, initial_state, mesh_info, num_skeletons=1000):
        """A plan skeleton is a sequence of symbolic block stacking actions.
        Multiple objects can be stacked on a single object, and even if one
        object is symbolically stacked on another object, it can be resting on
        two objects based on the continuous parameters.

        Several ways to generate this. The simplest is to sample a set of random
        skeletons by choosing source and target objects at random, iterating until
        all objects are used.

        Other ways to generate this include:
        1. Choosing blocks in an order so that they have a stable base (e.g., computing
        block 'eccentricities' and choosing blocks that are more 'stable' first, i.e.,
        blocks that have lower centers of mass).
        2. Choosing blocks in decreasing orders of face areas of bottom faces.
        3. Choosing blocks so that the hardest-to-place blocks are chosen last (e.g., blocks
        that have high centers of mass, or blocks that have a high center of mass and a small
        base area).
        4. Choosing blocks so that the heavier blocks are at the bottom (i.e., chosen first).
        5. Choosing blocks so that the blocks with the smallest base area are chosen last.
        """

        strategy = "random"  # other choices: "bottom_area", "eccentricity", "face_area"
        # strategy = "bottom_area"
        # strategy = "eccentricity"
        print("[Planner] get_plan_skeletons")

        if strategy == "random":
            plan_skeletons = []
            while len(plan_skeletons) < num_skeletons:
                # # Sample a random plan length (unused here; preserved for legacy purposes)
                # plan_length = random.choice(range(1, len(initial_state.block_ids)))
                # Use all blocks
                plan_length = len(initial_state.block_ids)
                skeleton = []
                state = initial_state
                collision = False
                current_tower = [initial_state.foundation]
                for _ in range(plan_length):
                    # sample target object
                    assert len(initial_state.block_ids) >= 2

                    target_object = random.choice(current_tower)
                    source_options = set(initial_state.block_ids) - set(current_tower)
                    source_object = random.choice(list(source_options))

                    action = self.sample_constrained_action(
                        state,
                        source_object,
                        target_object,
                        mesh_info,
                        client=self.env.client,
                    )
                    if check_collision(state, action, client=self.env.client):
                        collision = True
                        break

                    current_tower.append(source_object)
                    skeleton.append(action)

                if not collision:
                    plan_skeletons.append(skeleton)
        elif strategy == "bottom_area":
            # Sort blocks in decreasing order of bottom face areas
            sorted_blocks = sorted(
                mesh_info.items(),
                key=lambda x: x[1]["bottom_face_areas"][0][1],
                reverse=True,
            )
            plan_skeletons = []
            while len(plan_skeletons) < num_skeletons:
                skeleton = []
                state = initial_state
                collision = False
                for block_id, block_dict in sorted_blocks:
                    for target_block_id, _ in sorted_blocks:
                        if block_id == target_block_id:
                            continue
                        action = self.sample_constrained_action(
                            block_id, target_block_id, mesh_info
                        )
                        if check_collision(state, action, client=self.env.client):
                            collision = True
                            break
                        plan_skeletons.append([action])
                if not collision:
                    plan_skeletons.append(skeleton)
        elif strategy == "eccentricity":
            # Sort blocks in increasing order of eccentricity
            sorted_blocks = sorted(
                mesh_info.items(), key=lambda x: x[1]["eccentricity"]
            )
            plan_skeletons = []
            while len(plan_skeletons) < num_skeletons:
                skeleton = []
                state = initial_state
                collision = False
                for block_id, block_dict in sorted_blocks:
                    for target_block_id, _ in sorted_blocks:
                        if block_id == target_block_id:
                            continue
                        action = self.sample_constrained_action(
                            block_id, target_block_id, mesh_info
                        )
                        if check_collision(state, action, client=self.env.client):
                            collision = True
                            break
                        plan_skeletons.append([action])
                if not collision:
                    plan_skeletons.append(skeleton)
        else:
            raise NotImplementedError
        return plan_skeletons

    def get_plan(self, initial_state):
        # Cache all of the bandu objects, their meshes, and sort the faces sizes
        print("In get_plan")
        mesh_info = {}
        for block_id in initial_state.block_ids:
            block_dict = {}
            mesh = mesh_from_obj(block_id, client=self.env.client)
            tmesh = trimesh.Trimesh(mesh.vertices, mesh.faces)
            tmesh.fix_normals()
            face_sizes = sorted(
                [(i, area) for (i, area) in enumerate(tmesh.area_faces)],
                key=lambda x: x[1],
                reverse=True,
            )
            # Compute various mesh properties that will be useful for planning
            # (e.g., in the get_plan_skeletons function)
            # Quantities to compute could include:
            # - center of mass
            # - bounding box
            # - volume
            # - eccentricity
            # - mesh face areas
            # - areas of bottom faces
            # - etc.
            block_dict["face_sizes"] = face_sizes
            block_dict["mesh"] = tmesh
            block_dict["vertices"] = tmesh.vertices
            block_dict["faces"] = tmesh.faces
            block_dict["aabb"] = pbu.get_aabb(block_id, client=self.env.client)
            block_dict["center_of_mass"] = tmesh.center_mass
            block_dict["volume"] = tmesh.volume
            # Find geometric center of the block (i.e., center of the AABB)
            block_dict["geometric_center"] = np.mean(
                np.array(block_dict["aabb"].lower) + np.array(block_dict["aabb"].upper)
            )
            # Compute 'eccentricity' of the block (i.e., how far the center of mass is from the geometric center)
            # This could be used to sort blocks in a way that makes it easier to place them
            # (e.g., place the blocks with the lowest eccentricity first)
            block_dict["eccentricity"] = np.linalg.norm(
                block_dict["center_of_mass"] - block_dict["geometric_center"]
            )
            # Compute areas of bottom faces (i.e., faces that are closest to negative z-axis)
            # This could be used to sort blocks in a way that makes it easier to place them
            # (e.g., place the blocks with the largest bottom face areas first)
            # Remember to add face ids so that we know which face corresponds to which area
            block_dict["bottom_face_areas"] = [
                (i, area)
                for (i, area) in enumerate(tmesh.area_faces)
                if tmesh.face_normals[i][2] < 0
            ]
            # Sort the above list in descending order of areas
            block_dict["bottom_face_areas"] = sorted(
                block_dict["bottom_face_areas"], key=lambda x: x[1], reverse=True
            )

            mesh_info[block_id] = block_dict

        # Find k plan skeletons with correct collision free bounding boxes
        plan_skeletons = self.get_plan_skeletons(initial_state, mesh_info)

        # Sort by final state reward
        plan_skeletons = sorted(
            plan_skeletons,
            key=lambda plan: self.planning_heuristic(initial_state, plan),
            reverse=True,
        )

        # If using SBI, fit SBI inference models to each pair of source, target blocks
        if self.use_sbi:
            from bandu_stacking.sbi_utils import fit_sbi_model_pairwise

            # For each pair of blocks, fit an SBI model
            sbi_models = {}
            for source_block_id in initial_state.block_ids:
                for target_block_id in initial_state.block_ids:
                    if source_block_id == target_block_id:
                        continue
                    # Fit an SBI model for this pair of blocks
                    fit_sbi_model_pairwise(
                        source_block_id,
                        target_block_id,
                        mass=0.5,  # unused, but preserved for legacy purposes
                        friction=0.25,  # unused, but preserved for legacy purposes
                        restitution=0.1,  # unused, but preserved for legacy purposes
                        client=self.env.client,
                        prior_pose=None,  # unused, but preserved for legacy purposes
                        num_simulations=100,
                        proposal=None,
                    )
                    sbi_models[(source_block_id, target_block_id)] = sbi_model

        # For each skeleton, sample action parameters and evaluate in simulation
        for plan_skeleton in tqdm(plan_skeletons):
            self.env.set_sim_state(initial_state)
            concrete_plan = []
            diff_thresh = 0.005
            fail = False

            if self.use_sbi:
                # For each successive pair of blocks in the plan skeleton, use the SBI model
                # to estimate the likelihood of stability.
                for i in range(len(plan_skeleton) - 1):
                    source_block_id = plan_skeleton[i].grasp_block
                    target_block_id = plan_skeleton[i].target_block
                    sbi_model = sbi_models[(source_block_id, target_block_id)]

                    samples = sbi_model.sample((100,))
                    density = sbi_model.log_prob(samples)
                    best_sample = samples[np.argmax(density)]
                    yaw = best_sample[3]

                    pos, ori = self.env.client.getBasePositionAndOrientation(
                        source_block_id
                    )
                    euler = self.env.client.getEulerFromQuaternion(ori)
                    euler[2] = yaw
                    ori = self.env.client.getQuaternionFromEuler(euler)
                    action = Action(source_block_id, target_block_id, (pos, ori))

                    self.env.execute(action)
                    new_state = self.env.state_from_sim()
                    np.array(new_state.block_poses[action.grasp_block][0][:2])
                    adiff = np.linalg.norm(
                        np.array(new_state.block_poses[action.target_block][0][:2])
                        + np.array(action.pose[0][:2])
                        - np.array(new_state.block_poses[action.grasp_block][0][:2])
                    )
                    if adiff > diff_thresh:
                        fail = True
                        break
                    else:
                        concrete_plan.append(action)
                if not fail:
                    self.env.set_sim_state(initial_state)
                    return concrete_plan

            else:

                # Sample continuous parameters, test tower height, fail if height not expected
                print(plan_skeleton)
                for action in plan_skeleton:
                    self.env.execute(action)
                    new_state = self.env.state_from_sim()
                    np.array(new_state.block_poses[action.grasp_block][0][:2])
                    adiff = np.linalg.norm(
                        np.array(new_state.block_poses[action.target_block][0][:2])
                        + np.array(action.pose[0][:2])
                        - np.array(new_state.block_poses[action.grasp_block][0][:2])
                    )
                    if adiff > diff_thresh:
                        fail = True
                        break
                    else:
                        concrete_plan.append(action)

                if not fail:
                    pbu.wait_if_gui(
                        "Run CSP to construct the following tower?",
                        client=self.env.client,
                    )
                    self.env.set_sim_state(initial_state)
                    return concrete_plan

    def get_action(self, initial_state):
        if self.plan == None:
            abstract_actions = self.get_plan(initial_state)
            if abstract_actions is None:
                print("[Planner] There does not exist a physically plausible tower")
                sys.exit()
            current_state = initial_state
            self.plan = []
            for aai, aa in enumerate(abstract_actions):
                print(
                    "[Planner] Planning for abstract action {}: {}".format(
                        str(aai), str(aa)
                    )
                )
                sequence = get_pick_place_plan(aa, env=self.env)
                if sequence is None:
                    print("[Planner] Skeleton CSP failed")
                    sys.exit()
                else:
                    self.plan.append(sequence)

                current_state = self.env.step_abstract(current_state, aa)

        elif len(self.plan) == 0:
            # No more actions left, terminate the policy
            self.plan = None
            return None

        next_action = self.plan[0]
        self.plan = self.plan[1:]
        return next_action
