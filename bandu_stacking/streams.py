import math
import random
import time

import numpy as np

from bandu_stacking.env_utils import (
    Grasp,
    GroupConf,
    GroupTrajectory,
    ParentBody,
    Sequence,
    Switch,
)
from bandu_stacking.grasping import Z_AXIS, Plane, generate_mesh_grasps
from bandu_stacking.pb_utils import (
    BodySaver,
    Euler,
    Point,
    Pose,
    PoseSaver,
    Tuple,
    any_link_pair_collision,
    elapsed_time,
    get_aabb,
    get_aabb_center,
    get_extend_fn,
    get_moving_links,
    get_pose,
    get_top_and_bottom_grasps,
    invert,
    multiply,
    pairwise_collision,
    pairwise_collisions,
    plan_joint_motion,
    point_from_pose,
    quat_from_pose,
    randomize,
    recenter_oobb,
    scale_aabb,
    set_joint_positions,
    set_pose,
    stable_z_on_aabb,
)
from bandu_stacking.samplers import (
    COLLISION_DISTANCE,
    DISABLE_ALL_COLLISIONS,
    MOVABLE_DISTANCE,
    SELF_COLLISIONS,
    compute_gripper_path,
    plan_prehensile,
    plan_workspace_motion,
    sample_prehensive_base_confs,
    sample_visibility_base_confs,
    set_open_positions,
    workspace_collision,
)

TOOL_POSE = Pose(point=Point(x=0.00), euler=Euler(pitch=np.pi / 2))

SWITCH_BEFORE = "grasp"  # contact | grasp | pregrasp | arm | none # TODO: tractor
BASE_COST = 1
PROXIMITY_COST_TERM = False
REORIENT = False
RELAX_GRASP_COLLISIONS = False
GRASP_EXPERIMENT = False

GEOMETRIC_MODES = ["top", "side", "mesh"]
LEARNED_MODES = ["gpd", "graspnet"]
MODE_ORDERS = ["", "_random", "_best"]


def z_plane(z=0.0):
    normal = Z_AXIS
    origin = z * normal
    return Plane(normal, origin)


def close_until_collision(
    robot,
    gripper_joints,
    gripper_group,
    bodies=[],
    open_conf=None,
    closed_conf=None,
    num_steps=25,
    **kwargs,
):
    if not gripper_joints:
        return None

    closed_conf, open_conf = robot.get_group_limits(gripper_group)
    resolutions = np.abs(np.array(open_conf) - np.array(closed_conf)) / num_steps
    extend_fn = get_extend_fn(robot, gripper_joints, resolutions=resolutions, **kwargs)
    close_path = [open_conf] + list(extend_fn(open_conf, closed_conf))
    collision_links = frozenset(get_moving_links(robot, gripper_joints, **kwargs))
    for i, conf in enumerate(close_path):
        set_joint_positions(robot, gripper_joints, conf, **kwargs)
        if any(
            pairwise_collision((robot, collision_links), body, **kwargs)
            for body in bodies
        ):
            if i == 0:
                return None
            return close_path[i - 1][0]
    return close_path[-1][0]


def get_grasp_candidates(
    robot,
    obj,
    obj_aabb,
    obj_pose,
    grasp_mode="mesh",
    gripper_width=np.inf,
    tool_pose=TOOL_POSE,
    **kwargs,
):
    grasp_parts = grasp_mode.split("_")
    grasp_mode = grasp_parts[0]
    if grasp_mode == "mesh":
        return randomize(
            generate_mesh_grasps(
                obj, grasp_length=0.01, under=True, tool_pose=tool_pose, **kwargs
            )
        )  # get_top_cylinder_grasps
    elif grasp_mode == "top":
        return get_top_and_bottom_grasps(
            obj,
            obj_aabb,
            obj_pose,
            grasp_length=0.01,
            under=True,
            tool_pose=tool_pose,
            **kwargs,
        )


#######################################################


def get_grasp_gen_fn(
    robot,
    other_obstacles,
    grasp_mode="mesh",
    gripper_collisions=True,
    closed_fraction=5e-2,
    max_time=60,
    max_attempts=np.inf,
    **kwargs,
):
    grasp_mode = grasp_mode.split("_")[0]
    if grasp_mode in LEARNED_MODES:
        gripper_collisions = False

    def gen_fn(arm, obj, obj_aabb, obj_pose):
        # initial_poses = {obj: get_pose(world.bodies[obj]) for obj in world.items}
        side = robot.side_from_arm(arm)
        arm_group, gripper_group, tool_name = robot.manipulators[side]
        robot.link_from_name(tool_name)
        closed_conf, open_conf = robot.get_group_limits(gripper_group)

        set_open_positions(robot, side)
        max_width = robot.get_max_gripper_width(robot.get_group_joints(gripper_group))

        gripper = robot.get_component(gripper_group)
        parent_from_tool = robot.get_parent_from_tool(side)
        enable_collisions = gripper_collisions
        gripper_width = max_width - 1e-2  # TODO: gripper widthX_AXIS
        generator = iter(
            get_grasp_candidates(
                robot,
                obj,
                obj_aabb,
                obj_pose,
                grasp_mode=grasp_mode,
                gripper_width=gripper_width,
                **kwargs,
            )
        )
        new_tool_pose = TOOL_POSE
        last_time = time.time()
        last_attempts = 0
        while True:  # TODO: filter_grasps
            grasp_pose = next(generator)  # TODO: store past grasps

            if (
                (grasp_pose is None)
                or (elapsed_time(last_time) >= max_time)
                or (last_attempts >= max_attempts)
            ):
                if gripper_width == max_width:
                    print(
                        "Grasps for {} timed out after {} attempts and {:.3f} seconds".format(
                            obj, last_attempts, elapsed_time(last_time)
                        )
                    )
                    return
                gripper_width = max_width
                if RELAX_GRASP_COLLISIONS:
                    enable_collisions = (
                        False  # TODO: allow collisions with some parts of the gripper
                    )
                generator = iter(
                    get_grasp_candidates(
                        robot,
                        obj,
                        grasp_mode=grasp_mode,
                        gripper_width=max_width,
                        **kwargs,
                    )
                )
                last_time = time.time()
                last_attempts = 0
                continue

            last_attempts += 1

            set_pose(
                gripper,
                multiply(
                    get_pose(obj, **kwargs),
                    invert(multiply(parent_from_tool, grasp_pose)),
                ),
                **kwargs,
            )

            set_joint_positions(
                gripper, robot.get_component_joints(gripper_group), open_conf, **kwargs
            )

            obstacles = []
            if enable_collisions:
                obstacles.append(obj)

            obstacles.extend(other_obstacles)
            if pairwise_collisions(gripper, obstacles, **kwargs):
                continue

            set_pose(
                obj, multiply(robot.get_tool_link_pose(side), grasp_pose), **kwargs
            )
            set_open_positions(robot, side)

            if pairwise_collision(gripper, obj, **kwargs):
                continue

            set_pose(
                obj, multiply(robot.get_tool_link_pose(side), grasp_pose), **kwargs
            )

            closed_position = closed_conf[0]

            if SWITCH_BEFORE in ["contact", "grasp"]:
                closed_position = (1 + closed_fraction) * closed_position
            else:
                closed_position = closed_conf[0]

            grasp = Grasp(obj, grasp_pose, closed_position=closed_position, **kwargs)
            print("Generated grasp after {} attempts".format(last_attempts))

            yield Tuple(grasp)
            last_attempts = 0

    return gen_fn


#######################################################


def get_test_cfree_pose_pose(obj_obj_collisions=True, **kwargs):
    def test_cfree_pose_pose(obj1, pose1, obj2, pose2):
        if obj1 == obj2:  # or (pose2 is None): # TODO: skip if in the environment
            return True
        if obj2 in pose1.ancestors():
            return True
        pose1.assign()
        pose2.assign()
        return not pairwise_collision(obj1, obj2, max_distance=MOVABLE_DISTANCE)

    return test_cfree_pose_pose


def get_cfree_pregrasp_pose_test(robot, **kwargs):
    def test(arm, obj1, pose1, grasp1, obj2, pose2):
        side = robot.side_from_arm(arm)
        if obj1 == obj2:  # or (pose2 is None):
            return True
        if obj2 in pose1.ancestors():
            return True
        if (pose1.important and not obj1.is_fragile) and (
            pose2.important and not obj2.is_fragile
        ):
            return True
        pose2.assign()
        gripper_path = compute_gripper_path(pose1, grasp1)
        grasp = None if (pose1.important and pose2.important) else grasp1
        return not workspace_collision(
            robot,
            side,
            gripper_path,
            grasp,
            obstacles=[obj2],
            max_distance=MOVABLE_DISTANCE,
        )

    return test


def get_cfree_traj_pose_test(robot, **kwargs):
    def test(arm, sequence, obj2, pose2):
        if pose2 is None:  # (obj1 == obj2) or
            return True
        # if obj2 in pose1.ancestors():
        #     return True
        if sequence.name.startswith("pick") and (
            pose2.important and not obj2.is_fragile
        ):
            return True
        if obj2 in sequence.context_bodies:
            return True
        pose2.assign()
        set_open_positions(robot, arm)
        # state = State() # TODO: apply to the context

        for traj in sequence.commands:
            if not isinstance(traj, GroupTrajectory):
                continue
            if obj2 in traj.context_bodies:  # TODO: check the grasp
                continue
            moving_links = get_moving_links(traj.robot, traj.joints)
            # for _ in command.iterate(state=None):
            for _ in traj.traverse():
                # wait_if_gui()
                if any_link_pair_collision(
                    traj.robot, moving_links, obj2, max_distance=MOVABLE_DISTANCE
                ):  # \
                    # or any_link_pair_collision(traj.robot, moving_links, other_target_link, max_distance=MOVABLE_DISTANCE):
                    return False
        return True

    return test


#######################################################


def get_plan_pick_fn(robot, environment=[], **kwargs):
    robot_saver = BodySaver(robot, **kwargs)
    environment = environment

    def fn(arm, obj, pose, grasp, base_conf):
        # TODO: generator instead of a function
        # TODO: add the ancestors as collision obstacles
        robot_saver.restore()
        base_conf.assign()
        arm_path = plan_prehensile(robot, arm, obj, pose, grasp, **kwargs)

        if arm_path is None:
            return None

        arm_group, gripper_group, tool_name = robot.manipulators[
            robot.side_from_arm(arm)
        ]
        arm_traj = GroupTrajectory(
            robot,
            arm_group,
            arm_path[::-1],
            context=[pose],
            velocity_scale=0.25,
            client=robot.client,
        )
        arm_conf = arm_traj.first()

        closed_conf = grasp.closed_position * np.ones(
            len(robot.get_group_joints(gripper_group))
        )
        gripper_traj = GroupTrajectory(
            robot,
            gripper_group,
            path=[closed_conf],
            contexts=[pose],
            contact_links=robot.get_finger_links(robot.get_group_joints(gripper_group)),
            time_after_contact=1e-1,
            client=robot.client,
        )
        switch = Switch(
            obj,
            parent=ParentBody(
                body=robot, link=robot.link_from_name(tool_name), client=robot.client
            ),
        )

        # TODO: close the gripper a little bit before pregrasp
        if SWITCH_BEFORE == "contact":
            commands = [arm_traj, switch, arm_traj.reverse()]
        elif SWITCH_BEFORE == "grasp":
            commands = [arm_traj, switch, gripper_traj, arm_traj.reverse()]
        elif SWITCH_BEFORE == "pregrasp":
            commands = [arm_traj, gripper_traj, switch, arm_traj.reverse()]
        elif SWITCH_BEFORE == "arm":
            commands = [arm_traj, gripper_traj, arm_traj.reverse(), switch]
        elif SWITCH_BEFORE == "none":
            commands = [arm_traj, gripper_traj, arm_traj.reverse()]
        else:
            raise NotImplementedError(SWITCH_BEFORE)

        sequence = Sequence(
            commands=commands, name="pick-{}-{}".format(robot.side_from_arm(arm), obj)
        )
        return Tuple(arm_conf, sequence)

    return fn


#######################################################


def get_plan_place_fn(robot, **kwargs):
    robot_saver = BodySaver(robot, **kwargs)

    def fn(arm, obj, pose, grasp, base_conf):
        # TODO: generator instead of a function
        robot_saver.restore()
        base_conf.assign()
        arm_path = plan_prehensile(robot, arm, obj, pose, grasp, **kwargs)
        if arm_path is None:
            return None

        arm_group, gripper_group, tool_name = robot.manipulators[
            robot.side_from_arm(arm)
        ]
        arm_traj = GroupTrajectory(
            robot,
            arm_group,
            arm_path[::-1],
            context=[grasp],
            velocity_scale=0.25,
            client=robot.client,
        )
        arm_conf = arm_traj.first()

        closed_conf, open_conf = robot.get_group_limits(gripper_group)
        gripper_traj = GroupTrajectory(
            robot,
            gripper_group,
            path=[open_conf],
            contexts=[grasp],
            client=robot.client,
        )
        switch = Switch(obj, parent=None)

        # TODO: wait for a bit and remove colliding objects
        if SWITCH_BEFORE == "contact":
            commands = [arm_traj, switch, arm_traj.reverse()]
        elif SWITCH_BEFORE == "grasp":
            commands = [arm_traj, gripper_traj, switch, arm_traj.reverse()]
        elif SWITCH_BEFORE == "pregrasp":
            commands = [arm_traj, switch, gripper_traj, arm_traj.reverse()]
        elif SWITCH_BEFORE == "arm":
            commands = [switch, arm_traj, gripper_traj, arm_traj.reverse()]
        elif SWITCH_BEFORE == "none":
            commands = [arm_traj, gripper_traj, arm_traj.reverse()]
        else:
            raise NotImplementedError(SWITCH_BEFORE)
        sequence = Sequence(
            commands=commands, name="place-{}-{}".format(robot.side_from_arm(arm), obj)
        )
        return Tuple(arm_conf, sequence)

    return fn


def get_plan_mobile_place_fn(robot, **kwargs):
    robot_saver = BodySaver(robot, **kwargs)
    place_fn = get_plan_place_fn(robot, **kwargs)

    def fn(arm, obj, pose, grasp):
        robot_saver.restore()
        for base_conf in sample_prehensive_base_confs(
            robot, arm, obj, pose, grasp, **kwargs
        ):
            outputs = place_fn(arm, obj, pose, grasp, base_conf)
            if outputs is None:
                continue
            yield Tuple(base_conf) + outputs

    return fn


#######################################################


def get_plan_mobile_look_fn(
    robot, environment=[], max_head_attempts=10, max_base_attempts=100, **kwargs
):
    robot_saver = BodySaver(robot, **kwargs)

    def fn(obj, pose):
        while True:
            robot_saver.restore()
            # TODO: Return a head conf that will lead to visibility of obj at pose
            if robot.head_group is None:
                return None
            else:
                pose.assign()
                limits = list(robot.get_group_limits(robot.head_group))
                num_base_attempts = 0
                for base_conf in sample_visibility_base_confs(
                    robot, obj, pose, client=robot.client
                ):
                    visible = False
                    base_conf.assign()
                    num_head_attempts = 0
                    while not visible:
                        random_head_pos = [
                            random.uniform(*limit) for limit in zip(*limits)
                        ]
                        robot.set_group_positions(robot.head_group, random_head_pos)
                        visible = robot.cameras[0].object_visible(obj)
                        num_head_attempts += 1
                        if num_head_attempts >= max_head_attempts:
                            break
                    if num_head_attempts >= max_head_attempts:
                        continue
                    gp = random_head_pos
                    current_hq = GroupConf(
                        robot, robot.head_group, gp, client=robot.client
                    )
                    num_base_attempts += 1

                    yield (base_conf, current_hq)
                    if num_base_attempts > max_base_attempts:
                        return None
            num_attempts += 1

    return fn


def get_plan_look_fn(robot, environment=[], max_attempts=1000, **kwargs):
    robot_saver = BodySaver(robot, **kwargs)

    def fn(obj, pose, base_conf):
        while True:
            robot_saver.restore()
            base_conf.assign()
            # TODO: Return a head conf that will lead to visibility of obj at pose
            if robot.head_group is None:
                return None
            else:
                pose.assign()
                visible = False
                limits = list(robot.get_group_limits(robot.head_group))
                num_attempts = 0
                while not visible:
                    random_head_pos = [random.uniform(*limit) for limit in zip(*limits)]
                    robot.set_group_positions(robot.head_group, random_head_pos)
                    visible = robot.cameras[0].object_visible(obj)
                    num_attempts += 1
                    if num_attempts >= max_attempts:
                        num_attempts = 0
                        yield None

                gp = random_head_pos
                current_hq = GroupConf(robot, robot.head_group, gp, client=robot.client)
                yield (current_hq,)
            num_attempts += 1

    return fn


def get_plan_drop_fn(robot, environment=[], z_offset=2e-2, shrink=0.25, **kwargs):
    robot_saver = BodySaver(robot, **kwargs)

    def fn(arm, obj, grasp, bin, bin_pose, base_conf):
        # TODO: don't necessarily need the grasp
        robot_saver.restore()
        base_conf.assign()
        bin_pose.assign()
        obstacles = list(environment)

        side = robot.side_from_arm(arm)
        _, gripper_group, _ = robot.manipulators[side]
        gripper = robot.get_component(gripper_group)
        parent_from_tool = robot.get_parent_from_tool(side)

        bin_aabb = get_aabb(bin)
        # _, (_, _, z) = bin_aabb
        # x, y, _ = get_aabb_center(bin_aabb)
        # gripper_pose = Pose(point=Point(x, y, z + 0.1), euler=DOWNWARD_EULER)

        # reference_pose = unit_pose()
        reference_pose = multiply(
            Pose(euler=Euler(pitch=np.pi / 2, yaw=random.uniform(0, 2 * np.pi))),
            grasp.value,
        )
        # obj_pose = sample_placement_on_aabb(obj, bin_aabb, top_pose=reference_pose, percent=shrink, epsilon=1e-2)
        # _, extent = approximate_as_prism(obj, reference_pose=reference_pose)
        with PoseSaver(obj):
            set_pose(obj, reference_pose)
            obj_pose = (
                np.append(
                    get_aabb_center(bin_aabb)[:2],
                    [stable_z_on_aabb(obj, bin_aabb) + z_offset],
                ),
                quat_from_pose(reference_pose),
            )  # TODO: get_aabb_top, get_aabb_bottom

        if obj_pose is None:
            return None
        gripper_pose = multiply(obj_pose, invert(grasp.value))
        set_pose(gripper, multiply(gripper_pose, invert(parent_from_tool)))
        set_pose(obj, multiply(gripper_pose, grasp.value))
        if any(
            pairwise_collisions(body, environment, max_distance=0.0)
            for body in [obj, gripper]
        ):
            return None

        _, _, tool_name = robot.manipulators[robot.side_from_arm(arm)]
        attachment = grasp.create_attachment(
            robot, link=robot.link_from_name(tool_name)
        )

        arm_path = plan_workspace_motion(
            robot, side, [gripper_pose], attachment=attachment, obstacles=obstacles
        )
        if arm_path is None:
            return None
        arm_conf = GroupConf(robot, arm, positions=arm_path[0], **kwargs)
        switch = Switch(obj, parent=None)

        closed_conf, open_conf = robot.get_group_limits(gripper_group)
        # gripper_joints = robot.get_group_joints(gripper_group)
        # closed_conf = grasp.closed_position * np.ones(len(gripper_joints))
        gripper_traj = GroupTrajectory(
            robot,
            gripper_group,
            path=[closed_conf, open_conf],
            contexts=[],
            client=robot.client,
        )

        commands = [switch, gripper_traj]
        sequence = Sequence(
            commands=commands, name="drop-{}-{}".format(robot.side_from_arm(arm), obj)
        )
        return Tuple(arm_conf, sequence)

    return fn


#######################################################


def get_plan_motion_fn(
    robot, environment=[], **kwargs
):  # , collisions=True): #, teleport=False):
    robot_saver = BodySaver(robot, **kwargs)

    def fn(group, q1, q2, attachments=[]):
        robot_saver.restore()
        print("Plan motion fn {}->{}".format(q1, q2))

        obstacles = list(environment)
        attached = {attachment.child for attachment in attachments}
        obstacles = set(obstacles) - attached
        q1.assign()

        resolutions = math.radians(10) * np.ones(len(q2.joints))

        path = plan_joint_motion(
            robot,
            q2.joints,
            q2.positions,
            resolutions=resolutions,
            # weights=weights, # TODO: joint weights
            obstacles=obstacles,
            attachments=attachments,
            self_collisions=SELF_COLLISIONS,
            disabled_collisions=robot.disabled_collisions,
            max_distance=COLLISION_DISTANCE,
            custom_limits=robot.custom_limits,
            restarts=1,
            iterations=5,
            smooth=100,
            disable_collisions=DISABLE_ALL_COLLISIONS,
            **kwargs,
        )

        if path is None:
            for conf in [q1, q2]:
                conf.assign()
                for attachment in attachments:
                    attachment.assign()
            return None

        sequence = Sequence(
            commands=[
                GroupTrajectory(robot, group, path, client=robot.client),
            ],
            name="move-{}".format(group),
        )
        return Tuple(sequence)

    return fn


#######################################################


def get_nominal_test(robot, side="left", axis=1, **kwargs):
    def gen_fn(obj, pose, region, region_pose):
        value = point_from_pose(pose.relative_pose)[axis]
        success = (value > 0) if side == "left" else (value < 0)
        return success

    return gen_fn
