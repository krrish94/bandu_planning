import math
import random

import pybullet as p
import pybullet_utils.bullet_client as bc
from policies.planning.entities import WORLD_BODY
from policies.planning.primitives import CaptureImage, GroupConf, RelativePose, Sequence
from policies.planning.streams import (
    get_grasp_gen_fn,
    get_plan_motion_fn,
    get_plan_pick_fn,
    get_plan_place_fn,
)
from policies.simulation.entities import create_hollow
from robots.panda.panda_utils import PANDA_PATH, PandaRobot

from pybullet_planning.pybullet_tools.utils import (
    RGBA,
    Point,
    create_box,
    get_aabb_extent,
    get_joint_positions,
    get_movable_joints,
    load_pybullet,
    set_joint_positions,
    set_pose,
    stable_z_on_aabb,
)

INSPECT_CONFS = [
    [
        -0.24346381572605202,
        -0.4785839982367398,
        0.17864301396526722,
        -2.3873172224652452,
        0.08927869251235522,
        2.0302744557062784,
        0.6175902593665653,
    ],
    [
        -0.1459076988429237,
        0.6949633843689634,
        0.24148362022533745,
        -0.910208981589267,
        -0.12965228416698804,
        1.1791536833312775,
        0.7894618372652266,
    ],
    [
        -1.0643247591221312,
        0.02121897188321648,
        0.2506260129560907,
        -1.9973080197690338,
        0.5795392203064469,
        1.7147175591654247,
        -1.2888502919144096,
    ],
    [
        -1.3590669053006905,
        -1.4573949022627717,
        1.3828655291774816,
        -1.7886990782950538,
        0.6468850058317185,
        1.4015975437031851,
        2.00850522324774,
    ],
    [
        -1.0518002134987732,
        -1.7437583506594088,
        1.28843475313354,
        -3.0477574553447715,
        1.7115016128619511,
        1.675945115881585,
        -0.19696571404396399,
    ],
    [
        -0.7880282290806941,
        -1.3914924154638462,
        2.2887294194075567,
        -2.458989610914196,
        2.863521501278592,
        3.6707750855812162,
        0.18524528525028877,
    ],
    [
        0.11735846221344906,
        -1.6897707799610335,
        -1.8303618250897054,
        -2.6862082099245304,
        -0.7715732036144834,
        2.117022200551626,
        -0.5426696152951982,
    ],
]
CLASS_THRESH = 0.98

GROUP = "main_arm"


class Planner:
    def __init__(self):
        pass

    def plan(self, robot, belief, goal):
        raise NotImplementedError

    def get_pick_place_home_plan(
        self,
        robot,
        obj,
        obj_aabb,
        obj_pose,
        environment=[],
        surface_aabb=None,
        placement_pose=None,
        client=None,
    ):
        # Create the stream sampler functions
        motion_planner = get_plan_motion_fn(
            robot, environment=environment, client=client
        )
        pick_planner = get_plan_pick_fn(robot, client=client)
        place_planner = get_plan_place_fn(robot, client=client)
        grasp_finder = get_grasp_gen_fn(
            robot, environment, grasp_mode="top", client=client
        )

        # Set up sampler inputs
        init_confs = {
            group: GroupConf(robot, group, important=True, client=client)
            for group in robot.groups
        }
        pose = RelativePose(obj, client=client)
        base_conf = init_confs["base"]
        q1 = init_confs[GROUP]

        # Get a grasp
        (grasp,) = next(grasp_finder(GROUP, obj, obj_aabb, obj_pose))

        # Use the grasp to get a pick motion
        pick = None
        while pick is None:
            pick = pick_planner(GROUP, obj, pose, grasp, base_conf)
            print(f"[get_pick_place_home_plan] Pick 1: {pick}")

        q2, at1 = pick

        motion_plan3 = None
        while motion_plan3 is None:
            motion_plan2 = None
            while motion_plan2 is None:
                motion_plan = None
                while motion_plan is None:
                    place = None
                    while place is None:
                        if placement_pose is None:
                            place_pose = (
                                Point(
                                    x=random.uniform(
                                        surface_aabb.lower[0], surface_aabb.upper[0]
                                    ),
                                    y=random.uniform(
                                        surface_aabb.lower[1], surface_aabb.upper[1]
                                    ),
                                    z=stable_z_on_aabb(
                                        obj, surface_aabb, client=client
                                    ),
                                ),
                                obj_pose[1],
                            )
                        else:
                            place_pose = placement_pose

                        # Use the grasp to get a pick motion
                        place_rp = RelativePose(
                            obj,
                            parent=WORLD_BODY,
                            relative_pose=place_pose,
                            client=client,
                        )

                        place = place_planner(GROUP, obj, place_rp, grasp, base_conf)
                        print(f"[get_pick_place_home_plan] Place 1: {place}")

                    q3, at2 = place

                    # Use the resulting conf to motion plan
                    motion_plan = motion_planner(GROUP, q1, q2)
                    print(f"[get_pick_place_home_plan] Motion plan 1: {motion_plan}")

                (motion_plan,) = motion_plan

                # Use the resulting conf to motion plan
                _, _, tool_name = robot.manipulators[robot.side_from_arm(GROUP)]
                attachment = grasp.create_attachment(
                    robot, link=robot.link_from_name(tool_name)
                )

                motion_plan2 = motion_planner(GROUP, q2, q3, attachments=[attachment])
                print(f"[get_pick_place_home_plan] Motion plan 2: {motion_plan2}")

            (motion_plan2,) = motion_plan2

            # Use the resulting conf to motion plan
            motion_plan3 = motion_planner(GROUP, q3, q1)
            print(f"[get_pick_place_home_plan] Motion plan 3: {motion_plan3}")

        (motion_plan3,) = motion_plan3

        # Execute the plan
        return place_pose, Sequence([motion_plan, at1, motion_plan2, at2, motion_plan3])

    def get_pick_home_plan(
        self, robot, obj, obj_aabb, obj_pose, environment=[], client=None
    ):
        # Create the stream sampler functions
        motion_planner = get_plan_motion_fn(
            robot, environment=environment, client=client
        )
        pick_planner = get_plan_pick_fn(robot, client=client)
        grasp_finder = get_grasp_gen_fn(
            robot, environment, grasp_mode="top", client=client
        )

        # Set up sampler inputs
        init_confs = {
            group: GroupConf(robot, group, important=True, client=client)
            for group in robot.groups
        }
        pose = RelativePose(obj, client=client)
        base_conf = init_confs["base"]
        q1 = init_confs[GROUP]

        # Get a grasp
        (grasp,) = next(grasp_finder(GROUP, obj, obj_aabb, obj_pose))

        motion_plan2 = None
        while motion_plan2 is None:
            motion_plan = None
            while motion_plan is None:
                # Use the grasp to get a pick motion
                pick = None
                while pick is None:
                    pick = pick_planner(GROUP, obj, pose, grasp, base_conf)
                    print(f"[get_pick_home_plan] Pick 1: {pick}")

                q2, at = pick

                # Use the resulting conf to motion plan
                motion_plan = motion_planner(GROUP, q1, q2)
                print(f"[get_pick_home_plan] Motion 1: {motion_plan}")

            (motion_plan,) = motion_plan

            # Use the resulting conf to motion plan
            motion_plan2 = motion_planner(GROUP, q2, q1)
            print(f"[get_pick_home_plan] Motion 2: {motion_plan}")

        (motion_plan2,) = motion_plan2

        # Execute the plan
        return Sequence([motion_plan, at, motion_plan2])

    def world_twin(self, robot, belief):
        # Set up a surrogate simulated world with known state
        print(f"Cloning world at belief time {belief.time}")
        if belief.time == None:
            client = bc.BulletClient(connection_mode=p.GUI)
        else:
            client = bc.BulletClient(connection_mode=p.DIRECT)

        local_robot_body = load_pybullet(PANDA_PATH, fixed_base=True, client=client)
        local_robot = PandaRobot(local_robot_body, client=client)
        print("Robot: " + str(local_robot))
        # Mimic actual robot joints
        robot_movable_joints = get_movable_joints(robot, client=robot.client)
        local_robot_movable_joints = get_movable_joints(local_robot, client=client)
        positions = get_joint_positions(
            robot, robot_movable_joints, client=robot.client
        )
        set_joint_positions(
            local_robot, local_robot_movable_joints, positions, client=client
        )

        # Create the floor
        width, length, thickness = get_aabb_extent(belief.surface.aabb)
        height = 1
        fence = create_hollow(
            "fence",
            width=width,
            length=length,
            height=height,
            thickness=thickness,
            color=RGBA(0.75, 0.75, 0.75, 0.2),
            client=client,
        )
        table = create_box(
            width, length, thickness, color=RGBA(0.75, 0.75, 0.75, 1), client=client
        )

        set_pose(fence, belief.surface.pose, client=client)
        set_pose(table, belief.surface.pose, client=client)
        # wait_if_gui(client=client)
        return local_robot, table, fence, client

    def get_inspect_plan(self, robot, client=None):
        # Get the three camera poses based on the object pose
        motion_planner = get_plan_motion_fn(robot, environment=[], client=client)
        qs = [GroupConf(robot, GROUP, important=True, client=client)]
        for q in INSPECT_CONFS:
            qs.append(
                GroupConf(robot, GROUP, important=True, client=client, positions=q)
            )

        commands = []
        for qi in range(len(qs) - 1):
            (motion_plan,) = motion_planner(GROUP, qs[qi], qs[qi + 1])
            commands += [motion_plan, CaptureImage()]

        (motion_plan,) = motion_planner(GROUP, qs[len(qs) - 1], qs[0])
        commands += [motion_plan]

        return Sequence(commands=commands)
