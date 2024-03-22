import math
import random
import time

import numpy as np

from bandu_stacking.env_utils import (
    ARM_GROUP,
    GRIPPER_GROUP,
    PANDA_TOOL_TIP,
    Grasp,
    GroupConf,
    GroupTrajectory,
    ParentBody,
    Sequence,
    Switch,
)
from bandu_stacking.grasping import Z_AXIS, Plane, generate_mesh_grasps, sorted_grasps
from bandu_stacking.pb_utils import (
    BodySaver,
    Euler,
    Point,
    Pose,
    PoseSaver,
    Tuple,
    elapsed_time,
    get_aabb,
    get_aabb_center,
    get_extend_fn,
    get_moving_links,
    get_pose,
    get_top_and_bottom_grasps,
    invert,
    link_from_name,
    multiply,
    pairwise_collision,
    pairwise_collisions,
    plan_joint_motion,
    point_from_pose,
    quat_from_pose,
    randomize,
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

    closed_conf, open_conf = robot.get_group_limits(gripper_group, **kwargs)
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


def get_grasp_candidates(obj, gripper_width=np.inf, grasp_mode="mesh", **kwargs):

    if grasp_mode == "mesh":
        pitches = [-np.pi, np.pi]
        target_tolerance = np.pi / 4
        z_threshold = 1e-2
        antipodal_tolerance = np.pi / 16

        generated_grasps = generate_mesh_grasps(
            obj,
            pitches=pitches,
            discrete_pitch=False,
            max_width=gripper_width,
            max_time=2,
            target_tolerance=target_tolerance,
            antipodal_tolerance=antipodal_tolerance,
            z_threshold=z_threshold,
            **kwargs,
        )

        if generated_grasps is not None:
            return (
                grasp
                for grasp, contact1, contact2, score in sorted_grasps(
                    generated_grasps, max_candidates=10, p_random=0.0, **kwargs
                )
            )
        else:
            return tuple([])
    elif grasp_mode == "top":
        return [multiply(Pose(euler=Euler(pitch=-np.pi / 2.0)), Pose(Point(z=-0.01)))]


#######################################################


def get_grasp_gen_fn(
    robot,
    environment=[],
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

    def gen_fn(obj, obj_aabb, obj_pose):
        link_from_name(robot, PANDA_TOOL_TIP, **kwargs)
        closed_conf, open_conf = robot.get_group_limits(GRIPPER_GROUP, **kwargs)

        set_open_positions(robot, **kwargs)
        max_width = robot.get_max_gripper_width(
            robot.get_group_joints(GRIPPER_GROUP, **kwargs), **kwargs
        )

        gripper = robot.get_component(GRIPPER_GROUP, **kwargs)
        parent_from_tool = robot.get_parent_from_tool(**kwargs)
        enable_collisions = gripper_collisions
        gripper_width = max_width - 1e-2  # TODO: gripper widthX_AXIS
        generator = iter(
            get_grasp_candidates(
                obj,
                grasp_mode=grasp_mode,
                gripper_width=gripper_width,
                **kwargs,
            )
        )
        last_time = time.time()
        last_attempts = 0
        while True:  # TODO: filter_grasps
            grasp_pose = next(generator)  # TODO: store past grasps
            print(grasp_pose)
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
                gripper,
                robot.get_component_joints(GRIPPER_GROUP, **kwargs),
                open_conf,
                **kwargs,
            )

            obstacles = []
            if enable_collisions:
                obstacles.append(obj)

            obstacles.extend(environment)
            if pairwise_collisions(gripper, obstacles, **kwargs):
                continue

            set_pose(
                obj, multiply(robot.get_tool_link_pose(**kwargs), grasp_pose), **kwargs
            )
            set_open_positions(robot, **kwargs)

            if pairwise_collision(gripper, obj, **kwargs):
                continue

            set_pose(
                obj, multiply(robot.get_tool_link_pose(**kwargs), grasp_pose), **kwargs
            )

            closed_position = closed_conf[0]

            if SWITCH_BEFORE in ["contact", "grasp"]:
                closed_position = (1 + closed_fraction) * closed_position
            else:
                closed_position = closed_conf[0]

            grasp = Grasp(obj, grasp_pose, closed_position=closed_position, **kwargs)
            print("Generated grasp after {} attempts".format(last_attempts))

            yield (grasp,)
            last_attempts = 0

    return gen_fn


def get_plan_pick_fn(robot, environment=[], **kwargs):
    robot_saver = BodySaver(robot, **kwargs)
    environment = environment

    def fn(obj, pose, grasp, base_conf):
        # TODO: generator instead of a function
        # TODO: add the ancestors as collision obstacles
        robot_saver.restore()
        base_conf.assign(**kwargs)
        arm_path = plan_prehensile(robot, obj, pose, grasp, **kwargs)

        if arm_path is None:
            return None

        arm_traj = GroupTrajectory(
            robot,
            ARM_GROUP,
            arm_path[::-1],
            context=[pose],
            velocity_scale=0.25,
            **kwargs,
        )
        arm_conf = arm_traj.first()

        closed_conf = grasp.closed_position * np.ones(
            len(robot.get_group_joints(GRIPPER_GROUP, **kwargs))
        )
        gripper_traj = GroupTrajectory(
            robot,
            GRIPPER_GROUP,
            path=[closed_conf],
            contexts=[pose],
            contact_links=robot.get_finger_links(
                robot.get_group_joints(GRIPPER_GROUP, **kwargs), **kwargs
            ),
            time_after_contact=1e-1,
            **kwargs,
        )
        switch = Switch(
            obj,
            parent=ParentBody(
                body=robot,
                link=link_from_name(robot, PANDA_TOOL_TIP, **kwargs),
                **kwargs,
            ),
        )

        # TODO: close the gripper a little bit before pregrasp
        if SWITCH_BEFORE == "contact":
            commands = [arm_traj, switch, arm_traj.reverse(**kwargs)]
        elif SWITCH_BEFORE == "grasp":
            commands = [arm_traj, switch, gripper_traj, arm_traj.reverse(**kwargs)]
        elif SWITCH_BEFORE == "pregrasp":
            commands = [arm_traj, gripper_traj, switch, arm_traj.reverse(**kwargs)]
        elif SWITCH_BEFORE == "arm":
            commands = [arm_traj, gripper_traj, arm_traj.reverse(**kwargs), switch]
        elif SWITCH_BEFORE == "none":
            commands = [arm_traj, gripper_traj, arm_traj.reverse(**kwargs)]
        else:
            raise NotImplementedError(SWITCH_BEFORE)

        sequence = Sequence(commands=commands, name="pick-{}".format(obj))
        return (arm_conf, sequence)

    return fn


#######################################################


def get_plan_place_fn(robot, environment=[], **kwargs):
    robot_saver = BodySaver(robot, **kwargs)
    environment = environment

    def fn(obj, pose, grasp, base_conf):
        # TODO: generator instead of a function
        robot_saver.restore()
        base_conf.assign(**kwargs)
        arm_path = plan_prehensile(
            robot, obj, pose, grasp, environment=environment, **kwargs
        )
        if arm_path is None:
            print("[plan_place_fn] arm_path is None")
            return None

        arm_traj = GroupTrajectory(
            robot,
            ARM_GROUP,
            arm_path[::-1],
            context=[grasp],
            velocity_scale=0.25,
            **kwargs,
        )
        arm_conf = arm_traj.first()

        closed_conf, open_conf = robot.get_group_limits(GRIPPER_GROUP, **kwargs)
        gripper_traj = GroupTrajectory(
            robot,
            GRIPPER_GROUP,
            path=[open_conf],
            contexts=[grasp],
            **kwargs,
        )
        switch = Switch(obj, parent=None)

        # TODO: wait for a bit and remove colliding objects
        if SWITCH_BEFORE == "contact":
            commands = [arm_traj, switch, arm_traj.reverse(**kwargs)]
        elif SWITCH_BEFORE == "grasp":
            commands = [arm_traj, gripper_traj, switch, arm_traj.reverse(**kwargs)]
        elif SWITCH_BEFORE == "pregrasp":
            commands = [arm_traj, switch, gripper_traj, arm_traj.reverse(**kwargs)]
        elif SWITCH_BEFORE == "arm":
            commands = [switch, arm_traj, gripper_traj, arm_traj.reverse(**kwargs)]
        elif SWITCH_BEFORE == "none":
            commands = [arm_traj, gripper_traj, arm_traj.reverse(**kwargs)]
        else:
            raise NotImplementedError(SWITCH_BEFORE)
        sequence = Sequence(commands=commands, name="place-{}".format(obj))
        return (arm_conf, sequence)

    return fn


def get_plan_mobile_place_fn(robot, **kwargs):
    robot_saver = BodySaver(robot, **kwargs)
    place_fn = get_plan_place_fn(robot, **kwargs)

    def fn(obj, pose, grasp):
        robot_saver.restore()
        for base_conf in sample_prehensive_base_confs(
            robot, obj, pose, grasp, **kwargs
        ):
            outputs = place_fn(obj, pose, grasp, base_conf)
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
                pose.assign(**kwargs)
                limits = list(robot.get_group_limits(robot.head_group))
                num_base_attempts = 0
                for base_conf in sample_visibility_base_confs(
                    robot,
                    obj,
                    pose,
                    **kwargs,
                ):
                    visible = False
                    base_conf.assign(**kwargs)
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
                        robot,
                        robot.head_group,
                        gp,
                        **kwargs,
                    )
                    num_base_attempts += 1

                    yield (base_conf, current_hq)
                    if num_base_attempts > max_base_attempts:
                        return None
            num_attempts += 1

    return fn


def get_plan_drop_fn(robot, environment=[], z_offset=2e-2, shrink=0.25, **kwargs):
    robot_saver = BodySaver(robot, **kwargs)

    def fn(obj, grasp, bin, bin_pose, base_conf):
        robot_saver.restore()
        base_conf.assign(**kwargs)
        bin_pose.assign(**kwargs)
        obstacles = list(environment)

        gripper = robot.get_component(GRIPPER_GROUP, **kwargs)
        parent_from_tool = robot.get_parent_from_tool(**kwargs)

        bin_aabb = get_aabb(bin)

        reference_pose = multiply(
            Pose(euler=Euler(pitch=np.pi / 2, yaw=random.uniform(0, 2 * np.pi))),
            grasp.value,
        )
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

        attachment = grasp.create_attachment(
            robot, link=link_from_name(robot, PANDA_TOOL_TIP, **kwargs)
        )

        arm_path = plan_workspace_motion(
            robot, [gripper_pose], attachment=attachment, obstacles=obstacles
        )
        if arm_path is None:
            return None
        arm_conf = GroupConf(robot, ARM_GROUP, positions=arm_path[0], **kwargs)
        switch = Switch(obj, parent=None)

        closed_conf, open_conf = robot.get_group_limits(GRIPPER_GROUP, **kwargs)
        # gripper_joints = robot.get_group_joints(gripper_group)
        # closed_conf = grasp.closed_position * np.ones(len(gripper_joints))
        gripper_traj = GroupTrajectory(
            robot,
            GRIPPER_GROUP,
            path=[closed_conf, open_conf],
            contexts=[],
            **kwargs,
        )

        commands = [switch, gripper_traj]
        sequence = Sequence(commands=commands, name="drop-{}".format(obj))
        return (arm_conf, sequence)

    return fn


#######################################################


def get_plan_motion_fn(
    robot, environment=[], **kwargs
):  # , collisions=True): #, teleport=False):
    robot_saver = BodySaver(robot, **kwargs)

    def fn(q1, q2, attachments=[]):
        robot_saver.restore()
        print("Plan motion fn {}->{}".format(q1, q2))

        obstacles = list(environment)
        attached = {attachment.child for attachment in attachments}
        obstacles = set(obstacles) - attached
        q1.assign(**kwargs)

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
            max_distance=COLLISION_DISTANCE,
            restarts=1,
            iterations=5,
            smooth=100,
            disable_collisions=DISABLE_ALL_COLLISIONS,
            **kwargs,
        )

        if path is None:
            for conf in [q1, q2]:
                conf.assign(**kwargs)
                for attachment in attachments:
                    attachment.assign(**kwargs)
            return None

        sequence = Sequence(
            commands=[
                GroupTrajectory(robot, ARM_GROUP, path, **kwargs),
            ],
            name="move-{}".format(ARM_GROUP),
        )
        return sequence

    return fn


#######################################################


def get_nominal_test(robot, side="left", axis=1, **kwargs):
    def gen_fn(obj, pose, region, region_pose):
        value = point_from_pose(pose.relative_pose)[axis]
        success = (value > 0) if side == "left" else (value < 0)
        return success

    return gen_fn
