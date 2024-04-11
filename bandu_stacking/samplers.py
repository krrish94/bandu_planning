from __future__ import print_function

import numpy as np

import bandu_stacking.pb_utils as pbu
from bandu_stacking.env_utils import (
    ARM_GROUP,
    GRIPPER_GROUP,
    PANDA_INFO,
    PANDA_TOOL_TIP,
)
from bandu_stacking.inverse_kinematics.ikfast import (
    closest_inverse_kinematics,
    get_ik_joints,
    ikfast_inverse_kinematics,
)

USING_ROS = False

# TODO: could update MAX_DISTANCE globally
COLLISION_DISTANCE = 5e-3  # Distance from fixed obstacles
# MOVABLE_DISTANCE = 1e-2 # Distance from movable objects
MOVABLE_DISTANCE = COLLISION_DISTANCE
EPSILON = 1e-3
SELF_COLLISIONS = True  # TODO: check self collisions
# URDF_USE_SELF_COLLISION: by default, Bullet disables self-collision. This flag let's you enable it.

MAX_IK_TIME = 0.05 if USING_ROS else 0.01
MAX_IK_DISTANCE = np.inf if USING_ROS else np.inf  # math.radians(30)
MAX_TOOL_DISTANCE = np.inf
DISABLE_ALL_COLLISIONS = True


def get_closest_distance(robot, arm_joints, parent_link, tool_link, gripper_pose, obj):
    # TODO: operate on the object
    # gripper_pose = get_pose(obj)
    reach_pose = (pbu.point_from_pose(gripper_pose), None)
    sample_fn = pbu.get_sample_fn(robot, arm_joints)
    pbu.set_joint_positions(robot, arm_joints, sample_fn())
    # inverse_kinematics(robot, tool_link, reach_pose)
    # sub_inverse_kinematics(robot, arm_joints[0], tool_link, reach_pose)
    pbu.inverse_kinematics(robot, arm_joints[0], tool_link, reach_pose)
    # collision_infos = get_closest_points(robot, obj, max_distance=np.inf)
    collision_infos = pbu.get_closest_points(
        robot, obj, link1=parent_link, max_distance=np.inf
    )  # tool_link
    print("Collisions: {}".format(len(collision_infos)))
    for collision_info in collision_infos:
        pbu.draw_collision_info(collision_info)
    pbu.wait_if_gui()
    return min(
        [np.inf]
        + [collision_info.contactDistance for collision_info in collision_infos]
    )


#######################################################


def compute_gripper_path(pose, grasp, pos_step_size=0.02):
    # TODO: move linearly out of contact and then interpolate (ensure no collisions with the table)
    # grasp -> pregrasp
    grasp_pose = pbu.multiply(pose.get_pose(), pbu.invert(grasp.grasp))
    pregrasp_pose = pbu.multiply(pose.get_pose(), pbu.invert(grasp.pregrasp))
    gripper_path = list(
        pbu.interpolate_poses(grasp_pose, pregrasp_pose, pos_step_size=pos_step_size)
    )
    # handles = list(flatten(draw_pose(waypoint_pose, length=0.02) for waypoint_pose in gripper_path))
    return gripper_path


def create_grasp_attachment(robot, grasp, **kwargs):
    # TODO: robot.get_tool_link(side)
    return grasp.create_attachment(
        robot, link=pbu.link_from_name(robot, PANDA_TOOL_TIP, **kwargs)
    )


def plan_workspace_motion(
    robot,
    tool_waypoints,
    attachment=None,
    obstacles=[],
    max_attempts=2,
    debug=False,
    **kwargs,
):
    
    assert tool_waypoints

    tool_link = pbu.link_from_name(robot, PANDA_TOOL_TIP, **kwargs)
    ik_joints = get_ik_joints(robot, PANDA_INFO, tool_link, **kwargs)  # Arm + torso
    fixed_joints = set(ik_joints) - set(
        robot.get_group_joints(ARM_GROUP, **kwargs)
    )  # Torso only
    arm_joints = [j for j in ik_joints if j not in fixed_joints]  # Arm only
    extract_arm_conf = lambda q: [
        p for j, p in pbu.safe_zip(ik_joints, q) if j not in fixed_joints
    ]
    # tool_path = interpolate_poses(tool_waypoints[0], tool_waypoints[-1])

    parts = [robot] + ([] if attachment is None else [attachment.child])
    collision_fn = pbu.get_collision_fn(
        robot,
        arm_joints,
        obstacles=obstacles,
        attachments=[],
        self_collisions=SELF_COLLISIONS,
        disable_collisions=DISABLE_ALL_COLLISIONS,
        **kwargs,
    )

    for attempts in range(max_attempts):
        for arm_conf in ikfast_inverse_kinematics(
            robot,
            PANDA_INFO,
            tool_link,
            tool_waypoints[0],
            fixed_joints=fixed_joints,
            max_attempts=5,
            max_time=np.inf,
            max_distance=None,
            use_halton=False,
            **kwargs,
        ):
            if debug:
                print("[plan_workspace_motion] Found IK solution")
            # TODO: can also explore multiple ways to proceed
            arm_conf = extract_arm_conf(arm_conf)

            if collision_fn(arm_conf):
                if debug:
                    print("[plan_workspace_motion] in collision (0)")
                    pbu.wait_if_gui(**kwargs)
                continue
            arm_waypoints = [arm_conf]
            for tool_pose in tool_waypoints[1:]:
                # TODO: joint weights
                arm_conf = next(
                    closest_inverse_kinematics(
                        robot,
                        PANDA_INFO,
                        tool_link,
                        tool_pose,
                        fixed_joints=fixed_joints,
                        max_candidates=np.inf,
                        max_time=MAX_IK_TIME,
                        max_distance=MAX_IK_DISTANCE,
                        verbose=False,
                        **kwargs,
                    ),
                    None,
                )
                if arm_conf is None:
                    print("[plan_workspace_motion] arm_conf is None")
                    break

                arm_conf = extract_arm_conf(arm_conf)
                if collision_fn(arm_conf):
                    if debug:
                        print("[plan_workspace_motion] in collision (1)")
                        pbu.wait_if_gui(**kwargs)
                    continue
                arm_waypoints.append(arm_conf)
            else:
                pbu.set_joint_positions(robot, arm_joints, arm_waypoints[-1], **kwargs)
                if attachment is not None:
                    attachment.assign(**kwargs)
                if (
                    any(
                        pbu.pairwise_collisions(
                            part,
                            obstacles,
                            max_distance=(COLLISION_DISTANCE + EPSILON),
                            **kwargs,
                        )
                        for part in parts
                    )
                    and not DISABLE_ALL_COLLISIONS
                ):
                    if debug:
                        print("[plan_workspace_motion] in collision (2)")
                        pbu.wait_if_gui(**kwargs)
                    continue
                arm_path = pbu.interpolate_joint_waypoints(
                    robot, arm_joints, arm_waypoints, **kwargs
                )

                if any(collision_fn(q) for q in arm_path):
                    if debug:
                        print("[plan_workspace_motion] in collision (3)")
                        pbu.wait_if_gui(**kwargs)
                    continue

                print(
                    "Found path with {} waypoints and {} configurations after {} attempts".format(
                        len(arm_waypoints), len(arm_path), attempts + 1
                    )
                )

                return arm_path
    print("[plan_workspace_motion] max_attempts reached")
    return None


#######################################################


def workspace_collision(
    robot,
    gripper_path,
    grasp=None,
    open_gripper=True,
    obstacles=[],
    max_distance=0.0,
    **kwargs,
):
    if DISABLE_ALL_COLLISIONS:
        return False
    gripper = robot.get_component(GRIPPER_GROUP, **kwargs)

    if open_gripper:
        _, open_conf = robot.get_group_limits(GRIPPER_GROUP, **kwargs)
        gripper_joints = robot.get_component_joints(GRIPPER_GROUP, **kwargs)
        pbu.set_joint_positions(gripper, gripper_joints, open_conf, **kwargs)

    parent_from_tool = robot.get_parent_from_tool(**kwargs)
    parts = [gripper]
    if grasp is not None:
        parts.append(grasp.body)
    for i, gripper_pose in enumerate(gripper_path):
        pbu.set_pose(
            gripper, pbu.multiply(gripper_pose, pbu.invert(parent_from_tool)), **kwargs
        )
        if grasp is not None:
            pbu.set_pose(grasp.body, pbu.multiply(gripper_pose, grasp.value), **kwargs)

        distance = (
            (COLLISION_DISTANCE + EPSILON)
            if (i == len(gripper_path) - 1)
            else max_distance
        )
        if any(
            pbu.pairwise_collisions(part, obstacles, max_distance=distance, **kwargs)
            for part in parts
        ):
            return True
    return False


def plan_prehensile(robot, obj, pose, grasp, environment=[], debug=False, **kwargs):

    pose.assign(**kwargs)

    gripper_path = compute_gripper_path(pose, grasp)
    gripper_waypoints = gripper_path[:1] + gripper_path[-1:]
    if workspace_collision(
        robot, gripper_path, grasp=None, obstacles=environment, **kwargs
    ):
        if debug:
            print("[plan_prehensile] workspace collision")
        pbu.wait_if_gui(**kwargs)
        return None

    create_grasp_attachment(robot, grasp, **kwargs)
    arm_path = plan_workspace_motion(
        robot, gripper_waypoints, attachment=None, obstacles=environment, debug=debug, **kwargs
    )
    return arm_path


def set_closed_positions(robot, **kwargs):
    closed_conf, _ = robot.get_group_limits(GRIPPER_GROUP, **kwargs)
    robot.controller.set_group_positions(GRIPPER_GROUP, closed_conf, **kwargs)
    return closed_conf


def set_open_positions(robot, **kwargs):
    _, open_conf = robot.get_group_limits(GRIPPER_GROUP, **kwargs)
    robot.controller.set_group_positions(GRIPPER_GROUP, open_conf, **kwargs)
    return open_conf
