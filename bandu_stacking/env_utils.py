from __future__ import annotations

import itertools
import math
import os
import time
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import trimesh

import bandu_stacking.pb_utils as pbu
from bandu_stacking.inverse_kinematics.utils import IKFastInfo
from bandu_stacking.robot.robot_utils import FrankaFr3

GRIPPER_GROUP = "main_gripper"
CAMERA_FRAME = "camera_frame"
CAMERA_OPTICAL_FRAME = "camera_frame"
PANDA_TOOL_TIP = "panda_tool_tip"
ROBOT_IP = "192.168.1.11"
MAX_PANDA_FINGER = 0.045

TRANSPARENT = pbu.RGBA(0, 0, 0, 0)

ROOT_PATH = os.path.abspath(os.path.join(__file__, *[os.pardir] * 1))
SRL_PATH = os.path.join(ROOT_PATH, "models/srl")
PANDA_PATH = os.path.join(ROOT_PATH, "models/srl/franka_research_panda/panda.urdf")

WIDTH, HEIGHT = 640, 480
FX, FY = 525.0 / 2, 525.0 / 2

# OPEN_GRIPPER_POS = [0.045, 0.045]
OPEN_GRIPPER_POS = [10, 10]
CAMERA_MATRIX = pbu.get_camera_matrix(WIDTH, HEIGHT, FX, FY)
PANDA_GROUPS = {
    "base": [],
    "main_arm": ["panda_joint{}".format(i) for i in range(1, 8)],
    "main_gripper": ["panda_finger_joint1", "panda_finger_joint2"],
}

PANDA_INFO = IKFastInfo(
    module_name="franka_panda.ikfast_panda_arm",
    base_link="panda_link0",
    ee_link="panda_link8",
    free_joints=["panda_joint7"],
)

PANDA_TOOL_TIP = "panda_tool_tip"
ARM_GROUP = "main_arm"
GRIPPER_GROUP = "main_gripper"


YCB_PATH = os.path.join(SRL_PATH, "ycb")
USE_CONSTRAINTS = True

DEFAULT_ARM_POS = [
    -0.0806406098426434,
    -1.6722951504174777,
    0.07069076842695393,
    -2.7449419709102822,
    0.08184716251979611,
    1.7516337599063168,
    0.7849295270972781,
]


def ycb_type_from_name(name):
    return "_".join(name.split("_")[1:])


def ycb_type_from_file(path):
    # TODO: rename to be from_dir
    return ycb_type_from_name(os.path.basename(path))


def all_ycb_names():
    return [ycb_type_from_file(path) for path in pbu.list_paths(YCB_PATH)]


def all_ycb_paths():
    return pbu.list_paths(YCB_PATH)


def get_ycb_obj_path(ycb_type, use_concave=False):
    path_from_type = {
        ycb_type_from_file(path): path
        for path in pbu.list_paths(YCB_PATH)
        if os.path.isdir(path)
    }

    if ycb_type not in path_from_type:
        return None

    if use_concave:
        filename = "google_16k/textured_vhacd.obj"
    else:
        filename = "google_16k/textured.obj"

    return os.path.join(path_from_type[ycb_type], filename)


def create_ycb(
    name,
    use_concave=True,
    client=None,
    scale=1.0,
    **kwargs,
):
    client = client or pbu.DEFAULT_CLIENT
    concave_ycb_path = get_ycb_obj_path(name, use_concave=use_concave)
    ycb_path = get_ycb_obj_path(name)
    mass = -1

    # TODO: separate visual and collision boddies
    color = pbu.WHITE

    mesh = trimesh.load(ycb_path)

    # TODO: separate visual and collision geometries
    # TODO: compute OOBB to select the orientation
    visual_geometry = pbu.get_mesh_geometry(
        ycb_path, scale=scale
    )  # TODO: randomly transform
    collision_geometry = pbu.get_mesh_geometry(concave_ycb_path, scale=scale)
    geometry_pose = pbu.Pose(point=-mesh.center_mass)
    collision_id = pbu.create_collision_shape(
        collision_geometry, pose=geometry_pose, client=client
    )
    visual_id = pbu.create_visual_shape(
        visual_geometry, color=color, pose=geometry_pose, client=client
    )
    # collision_id, visual_id = create_shape(geometry, collision=True, color=WHITE)
    body = client.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        # basePosition=[0., 0., 0.1]
    )

    pbu.set_all_color(body, pbu.apply_alpha(color, alpha=1.0), client=client)
    # dump_body(body)
    # wait_if_gui()
    return body


TABLE_AABB = pbu.AABB(
    lower=(-0.50 / 2.0, -1.0 / 2.0, -0.03 / 2.0),
    upper=(0.50 / 2.0, 1.0 / 2.0, 0.03 / 2.0),
)
TABLE_POSE = pbu.Pose((0.45, 0, -TABLE_AABB.upper[2]))


def get_data_path():
    import pybullet_data

    return pybullet_data.getDataPath()


def add_data_path(data_path=None):
    if data_path is None:
        data_path = get_data_path()
    p.setAdditionalSearchPath(data_path)
    return data_path


def create_default_env(
    client=None, table_aabb=TABLE_AABB, table_pose=TABLE_POSE, **kwargs
):
    # TODO: p.loadSoftBody

    client.resetDebugVisualizerCamera(
        cameraDistance=2,
        cameraYaw=90,
        cameraPitch=-15,
        cameraTargetPosition=[-0.5, 0, 0.3],
    )
    # draw_pose(Pose(), length=1)

    add_data_path()
    floor, _ = add_table(
        *pbu.get_aabb_extent(table_aabb), table_pose=table_pose, client=client
    )
    obstacles = [
        floor,  # collides with the robot when MAX_DISTANCE >= 5e-3
    ]

    for obst in obstacles:
        pbu.set_dynamics(
            obst,
            lateralFriction=1.0,  # linear (lateral) friction
            spinningFriction=1.0,  # torsional friction around the contact normal
            rollingFriction=0.01,  # torsional friction orthogonal to contact normal
            restitution=0.0,  # restitution: 0 => inelastic collision, 1 => elastic collision
            client=client,
        )

    return floor, obstacles


def add_table(
    table_width: float = 0.75,
    table_length: float = 1.22,
    table_thickness: float = 0.03,
    table_pose: pbu.Pose = TABLE_POSE,
    color: Tuple[float, float, float, float] = (0.75, 0.75, 0.75, 1.0),
    client=None,
) -> Tuple[int, List[int]]:
    # Panda table downstairs very roughly (few cm of error)
    table = pbu.create_box(
        table_width, table_length, table_thickness, color=color, client=client
    )
    pbu.set_pose(table, table_pose, client=client)
    workspace = []

    return table, workspace


def all_ycb_names():
    return [ycb_type_from_file(path) for path in pbu.list_paths(YCB_PATH)]


class SimulatedController:
    """Function calls on this object execute on the robot in simulation."""

    def __init__(self, robot, **kwargs):
        self.robot = robot

    def open_gripper(self, arm):  # These are mirrored on the pr2
        _, gripper_group, _ = self.robot.manipulator
        _, open_conf = self.robot.get_group_limits(gripper_group)
        self.command_group(gripper_group, open_conf)

    def close_gripper(self, arm):  # These are mirrored on the pr2
        _, gripper_group, _ = self.robot.manipulator
        closed_conf, _ = self.robot.get_group_limits(gripper_group)
        self.command_group(gripper_group, closed_conf)

    def get_group_joints(self, group, **kwargs):
        return pbu.joints_from_names(
            self.robot, self.robot.joint_groups[group], **kwargs
        )

    def set_group_conf(self, group, positions, **kwargs):
        pbu.set_joint_positions(
            self.robot, self.get_group_joints(group, **kwargs), positions, **kwargs
        )

    def set_group_positions(self, group, positions, **kwargs):
        self.set_group_conf(group, positions, **kwargs)

    def get_joint_positions(self, **kwargs):
        joints = pbu.get_joints(self.robot, **kwargs)
        joint_positions = pbu.get_joint_positions(self.robot, joints, **kwargs)
        joint_names = pbu.get_joint_names(self.robot, joints, **kwargs)
        return {k: v for k, v in zip(joint_names, joint_positions)}

    def command_group(self, group, positions, **kwargs):  # TODO: default timeout
        self.set_group_positions(group, positions, **kwargs)

    def command_group_dict(
        self, group, positions_dict, **kwargs
    ):  # TODO: default timeout
        positions = [positions_dict[nm] for nm in self.robot.joint_groups[group]]
        self.command_group(group, positions, **kwargs)

    def command_group_trajectory(self, group, positions, dt=0.01, **kwargs):
        for position in positions:
            self.command_group(group, position, **kwargs)
            time.sleep(dt)

    def wait(self, duration):
        time.sleep(duration)

    def any_arm_fully_closed(self):
        return False


def setup_robot_pybullet(gui=False):
    client = bc.BulletClient(connection_mode=p.GUI if gui else p.DIRECT)
    client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    client.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    robot_body = pbu.load_pybullet(PANDA_PATH, fixed_base=True, client=client)
    return robot_body, client


class Conf(object):  # TODO: parent class among Pose, Grasp, and Conf
    # TODO: counter
    def __init__(self, body, joints, positions=None, important=False, **kwargs):
        # TODO: named conf
        self.body = body
        self.joints = joints
        assert positions is not None
        self.positions = tuple(positions)
        self.important = important
        # TODO: parent state?

    @property
    def robot(self):
        return self.body

    @property
    def values(self):
        return self.positions

    def assign(self, **kwargs):
        pbu.set_joint_positions(self.body, self.joints, self.positions, **kwargs)

    def iterate(self):
        yield self

    def __repr__(self):
        return "q{}".format(id(self) % 1000)


class GroupConf(Conf):
    def __init__(self, body, group, *args, **kwargs):
        joints = body.get_group_joints(group, **kwargs)
        super(GroupConf, self).__init__(body, joints, *args, **kwargs)
        self.group = group

    def __repr__(self):
        return "{}q{}".format(self.group[0], id(self) % 1000)


class Command(object):

    @property
    def context_bodies(self):
        return set()

    def iterate(self, state, **kwargs):
        raise NotImplementedError()

    def controller(self, *args, **kwargs):
        raise NotImplementedError()

    def execute(self, controller, *args, **kwargs):
        # raise NotImplementedError()
        return True

    def to_lisdf(self):
        raise NotImplementedError


class Switch(Command):
    def __init__(self, body, parent=None):
        # TODO: add client
        self.body = body
        self.parent = parent

    def iterate(self, state, real_controller: RealController = None, **kwargs):

        if self.parent is None and self.body in state.attachments.keys():
            if real_controller is not None:
                real_controller.ungrasp()
            del state.attachments[self.body]

        elif self.parent is not None:

            if real_controller is not None:
                real_controller.grasp()

            robot, tool_link = self.parent

            gripper_joints = robot.get_group_joints(GRIPPER_GROUP, **kwargs)
            finger_links = robot.get_finger_links(gripper_joints, **kwargs)

            movable_bodies = [
                body for body in pbu.get_bodies(**kwargs) if (body != robot)
            ]

            # collision_bodies = [body for body in movable_bodies if any_link_pair_collision(
            #    robot, finger_links, body, max_distance=1e-2)]

            # gripper_width = robot.get_gripper_width(gripper_joints)
            # max_width = robot.get_max_gripper_width(
            #     robot.get_group_joints(GRIPPER_GROUP)
            # )

            max_distance = 5e-2
            collision_bodies = [
                body
                for body in movable_bodies
                if (
                    all(
                        pbu.get_closest_points(
                            robot, body, link1=link, max_distance=max_distance, **kwargs
                        )
                        for link in finger_links
                    )
                    and pbu.get_mass(body, **kwargs) != pbu.STATIC_MASS
                )
            ]

            if len(collision_bodies) > 0:
                relative_pose = RelativePose(
                    collision_bodies[0], parent=self.parent, **kwargs
                )
                state.attachments[self.body] = relative_pose

        return pbu.empty_sequence()

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.body)

    def to_lisdf(self):
        return []


class Trajectory(Command):
    def __init__(
        self,
        body,
        joints,
        path,
        velocity_scale=1.0,
        contact_links=[],
        time_after_contact=np.inf,
        contexts=[],
        **kwargs,
    ):
        self.body = body
        self.joints = joints
        self.path = tuple(path)  # waypoints_from_path
        self.velocity_scale = velocity_scale
        self.contact_links = tuple(contact_links)
        self.time_after_contact = time_after_contact
        self.contexts = tuple(contexts)
        # self.kwargs = dict(kwargs) # TODO: doesn't save unpacked values

    @property
    def robot(self):
        return self.body

    @property
    def context_bodies(self):
        return {self.body} | {
            context.body for context in self.contexts
        }  # TODO: ancestors

    def conf(self, positions):
        return Conf(self.body, self.joints, positions=positions)

    def first(self):
        return self.conf(self.path[0])

    def last(self):
        return self.conf(self.path[-1])

    def reverse(self):
        return self.__class__(
            self.body,
            self.joints,
            self.path[::-1],
            velocity_scale=self.velocity_scale,
            contact_links=self.contact_links,
            time_after_contact=self.time_after_contact,
            contexts=self.contexts,
        )

    def adjust_path(self, **kwargs):
        current_positions = pbu.get_joint_positions(
            self.body, self.joints, **kwargs
        )  # Important for adjust_path
        return pbu.adjust_path(
            self.body, self.joints, [current_positions] + list(self.path), **kwargs
        )  # Accounts for the wrap around

    def compute_waypoints(self, **kwargs):
        return pbu.waypoints_from_path(
            pbu.adjust_path(self.body, self.joints, self.path, **kwargs)
        )

    def compute_curve(self, **kwargs):
        path = self.adjust_path(**kwargs)
        positions_curve = pbu.interpolate_path(self.body, self.joints, path, **kwargs)
        return positions_curve

    def iterate(self, state, teleport=False, **kwargs):
        if teleport:
            pbu.set_joint_positions(self.body, self.joints, self.path[-1], **kwargs)
            return self.path[-1]
        else:
            return pbu.step_curve(
                self.body, self.joints, self.compute_curve(**kwargs), **kwargs
            )

    def __repr__(self):
        return "t{}".format(id(self) % 1000)


class GroupTrajectory(Trajectory):
    def __init__(self, body, group, path, *args, **kwargs):
        # TODO: rename body to robot
        joints = body.get_group_joints(group, **kwargs)
        super(GroupTrajectory, self).__init__(body, joints, path, *args, **kwargs)
        self.group = group

    def iterate(self, state, real_controller: RealController = None, **kwargs):

        if real_controller is not None:
            # current_joint_positions = real_controller.get_joint_positions(**kwargs)
            path = list(self.path)

            # if(np.sum(np.abs(np.array(path[0])-np.array(current_joint_positions)))>0.05):
            #     path = [current_joint_positions]+path

            real_controller.command_group_trajectory(self.group, path, **kwargs)
        return super().iterate(state, **kwargs)

    def reverse(self, **kwargs):
        return self.__class__(
            self.body,
            self.group,
            self.path[::-1],
            velocity_scale=self.velocity_scale,
            contact_links=self.contact_links,
            time_after_contact=self.time_after_contact,
            contexts=self.contexts,
            **kwargs,
        )

    def __repr__(self):
        return "{}t{}".format(self.group[0], id(self) % 1000)


class RelativePose(object):
    def __init__(
        self,
        body,
        parent=None,
        parent_state=None,
        relative_pose=None,
        important=False,
        **kwargs,
    ):
        self.body = body
        self.parent = parent
        self.parent_state = parent_state
        if not isinstance(self.body, int):
            self.body = int(str(self.body).split("#")[1])
        if relative_pose is None:
            relative_pose = pbu.multiply(
                pbu.invert(self.get_parent_pose(**kwargs)),
                pbu.get_pose(self.body, **kwargs),
            )
        self.relative_pose = tuple(relative_pose)
        self.important = important

    @property
    def value(self):
        return self.relative_pose

    def ancestors(self):
        if self.parent_state is None:
            return [self.body]
        return self.parent_state.ancestors() + [self.body]

    def get_parent_pose(self, **kwargs):
        if self.parent is None:
            return pbu.unit_pose()
        if self.parent_state is not None:
            self.parent_state.assign(**kwargs)
        return self.parent.get_pose(**kwargs)

    def get_pose(self, **kwargs):
        return pbu.multiply(self.get_parent_pose(**kwargs), self.relative_pose)

    def assign(self, **kwargs):
        world_pose = self.get_pose(**kwargs)
        pbu.set_pose(self.body, world_pose, **kwargs)
        return world_pose

    def get_attachment(self, **kwargs):
        assert self.parent is not None
        parent_body, parent_link = self.parent
        return pbu.Attachment(
            parent_body, parent_link, self.relative_pose, self.body, **kwargs
        )

    def __repr__(self):
        name = "wp" if self.parent is None else "rp"
        return "{}{}".format(name, id(self) % 1000)


class RealController:

    def __init__(self, body):
        self.body = body
        self.robot = FrankaFr3(ROBOT_IP, np.eye(4))

    def command_group_trajectory(self, group, positions, dt=0.01, **kwargs):
        for position in positions:
            self.command_group(group, position, **kwargs)
            time.sleep(dt)

    def grasp(self):
        print("[Real Robot Controller] Grasp from switch")
        self.robot.grasp()

    def ungrasp(self):
        print("[Real Robot Controller] Ungrasp from switch")
        self.robot.release_gripper()

    def command_group(self, group, positions, **kwargs):  # TODO: default timeout

        if group == ARM_GROUP:
            positions_degrees = [math.degrees(p) for p in positions]
            self.robot.move_joints(positions_degrees)
        else:
            if list(positions)[0] > (MAX_PANDA_FINGER - 0.01):
                print("[Real Robot Controller] Grasp from command")
                self.robot.release_gripper()
            else:
                print("[Real Robot Controller] Ungrasp from command")
                self.robot.grasp()

    def get_joint_positions(self, **kwargs):
        return [math.radians(angle) for angle in self.robot.joint_angles]


class PandaRobot:
    def __init__(
        self,
        robot_body,
        link_names={},
        **kwargs,
    ):
        self.link_names = link_names
        self.body = robot_body
        self.joint_groups = PANDA_GROUPS
        self.components = {}
        self.controller = SimulatedController(self)

        self.max_depth = 3.0
        self.min_z = 0.0
        self.BASE_LINK = "panda_link0"

        self.reset(**kwargs)

    def __int__(self):
        return self.body

    def get_default_conf(self):
        conf = {
            "main_arm": DEFAULT_ARM_POS,
            "main_gripper": [MAX_PANDA_FINGER, MAX_PANDA_FINGER],
        }
        return conf

    def get_closed_positions(self):
        return {"panda_finger_joint1": 0, "panda_finger_joint2": 0}

    def get_open_positions(self):
        return {
            "panda_finger_joint1": MAX_PANDA_FINGER,
            "panda_finger_joint2": MAX_PANDA_FINGER,
        }

    def get_group_joints(self, group, **kwargs):
        return pbu.joints_from_names(self.body, PANDA_GROUPS[group], **kwargs)

    def update_conf(self, conf, **kwargs):
        for name, position in conf.items():
            joint = pbu.joint_from_name(self, name, **kwargs)
            pbu.set_joint_position(self, joint, position, **kwargs)
        return conf

    def update_sim_conf(self, **kwargs):
        return self.update_conf(
            dict(self.controller.get_joint_positions(**kwargs)), **kwargs
        )

    def reset(self, **kwargs):
        conf = self.get_default_conf()
        for group, positions in conf.items():
            self.controller.set_group_positions(group, positions, **kwargs)

    def get_group_limits(self, group, **kwargs):
        return pbu.get_custom_limits(
            self.body, self.get_group_joints(group, **kwargs), **kwargs
        )

    def get_gripper_width(self, gripper_joints, **kwargs):
        [link1, link2] = self.get_finger_links(gripper_joints, **kwargs)
        collision_info = pbu.get_closest_points(
            self.body, self.body, link1, link2, max_distance=np.inf, **kwargs
        )[0]
        point1 = collision_info.positionOnA
        point2 = collision_info.positionOnB
        max_width = pbu.get_distance(point1, point2)
        return max_width

    def get_max_gripper_width(self, gripper_joints, **kwargs):
        with pbu.ConfSaver(self, **kwargs):
            pbu.set_joint_positions(
                self.body,
                gripper_joints,
                pbu.get_max_limits(self.body, gripper_joints, **kwargs),
                **kwargs,
            )
            return self.get_gripper_width(gripper_joints, **kwargs)

    def get_finger_links(self, gripper_joints, **kwargs):
        moving_links = pbu.get_moving_links(self.body, gripper_joints, **kwargs)
        shape_links = [
            link
            for link in moving_links
            if pbu.get_collision_data(self.body, link, **kwargs)
        ]
        finger_links = [
            link
            for link in shape_links
            if not any(
                pbu.get_collision_data(self.body, child, **kwargs)
                for child in pbu.get_link_children(self.body, link, **kwargs)
            )
        ]
        if len(finger_links) != 2:
            raise RuntimeError(finger_links)
        return finger_links

    def get_group_parent(self, group, **kwargs):
        # TODO: handle unordered joints
        return pbu.get_link_parent(
            self.body, self.get_group_joints(group, **kwargs)[0], **kwargs
        )

    def get_tool_link_pose(self, **kwargs):
        tool_link = pbu.link_from_name(self.body, PANDA_TOOL_TIP, **kwargs)
        return pbu.get_link_pose(self.body, tool_link, **kwargs)

    def get_parent_from_tool(self, **kwargs):
        tool_tip_link = pbu.link_from_name(self.body, PANDA_TOOL_TIP, **kwargs)
        parent_link = self.get_group_parent(GRIPPER_GROUP, **kwargs)
        return pbu.get_relative_pose(self.body, tool_tip_link, parent_link, **kwargs)

    def get_component_mapping(self, group, **kwargs):
        assert group in self.components
        component_joints = pbu.get_movable_joints(
            self.components[group], draw=False, **kwargs
        )
        body_joints = pbu.get_movable_joint_descendants(
            self.body, self.get_group_parent(group, **kwargs), **kwargs
        )
        return OrderedDict(pbu.safe_zip(body_joints, component_joints))

    def get_component_joints(self, group, **kwargs):
        mapping = self.get_component_mapping(group, **kwargs)
        return list(map(mapping.get, self.get_group_joints(group, **kwargs)))

    def get_component_info(self, fn, group, **kwargs):
        return fn(self.body, self.get_group_joints(group, **kwargs))

    def get_group_subtree(self, group, **kwargs):
        return pbu.get_link_subtree(
            self.body, self.get_group_parent(group, **kwargs), **kwargs
        )

    def get_component(self, group, visual=True, **kwargs):
        if group not in self.components:
            component = pbu.clone_body(
                self.body,
                links=self.get_group_subtree(group, **kwargs),
                visual=False,
                collision=True,
                **kwargs,
            )
            if not visual:
                pbu.set_all_color(component, TRANSPARENT)
            self.components[group] = component
        return self.components[group]

    def remove_components(self, **kwargs):
        for component in self.components.values():
            pbu.remove_body(component, **kwargs)
        self.components = {}


class WorldState(pbu.State):
    def __init__(self, savers=[], attachments={}, client=None):
        # a part of the state separate from PyBullet
        # TODO: other fluent things
        super(WorldState, self).__init__(attachments)
        self.attachments = attachments
        self.world_saver = pbu.WorldSaver(client=client)
        self.savers = tuple(savers)
        self.client = client

    def assign(self):
        self.world_saver.restore()
        for saver in self.savers:
            saver.restore()
        self.propagate()

    def copy(self):  # update
        return self.__class__(savers=self.savers, attachments=self.attachments)

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, list(self.savers), sorted(self.attachments)
        )


class Sequence(Command):  # Commands, CommandSequence
    def __init__(self, commands=[], name=None):
        self.context = None  # TODO: make a State?
        self.commands = tuple(commands)
        self.name = self.__class__.__name__.lower()[:3] if (name is None) else name

    @property
    def context_bodies(self):
        return set(
            itertools.chain(*[command.context_bodies for command in self.commands])
        )

    def __len__(self):
        return len(self.commands)

    def iterate(self, *args, **kwargs):
        for command in self.commands:
            print("Executing {} command: {}".format(type(command), str(command)))
            for output in command.iterate(*args, **kwargs):
                yield output

    def controller(self, *args, **kwargs):
        return itertools.chain.from_iterable(
            command.controller(*args, **kwargs) for command in self.commands
        )

    def execute(self, *args, return_executed=False, **kwargs):
        executed = []
        for command in self.commands:
            if not command.execute(*args, **kwargs):
                return False, executed if return_executed else False
            executed.append(command)
        return True, executed if return_executed else True

    def reverse(self):
        return Sequence(
            [command.reverse() for command in reversed(self.commands)], name=self.name
        )

    def dump(self):
        print("[{}]".format(" -> ".join(map(repr, self.commands))))

    def __repr__(self):
        return "{}({})".format(self.name, len(self.commands))

    def to_lisdf(self):
        return sum([command.to_lisdf() for command in self.commands], [])


PREGRASP_DISTANCE = 0.05


def get_pregrasp(
    grasp_tool,
    gripper_from_tool=pbu.unit_pose(),
    tool_distance=PREGRASP_DISTANCE,
    object_distance=PREGRASP_DISTANCE,
):
    # TODO: rename to approach, standoff, guarded, ...
    return pbu.multiply(
        gripper_from_tool,
        pbu.Pose(pbu.Point(x=tool_distance)),
        grasp_tool,
        pbu.Pose(pbu.Point(z=-object_distance)),
    )


class Grasp(object):  # RelativePose
    def __init__(self, body, grasp, pregrasp=None, closed_position=0.0, **kwargs):
        # TODO: condition on a gripper (or list valid pairs)
        self.body = body
        self.grasp = grasp
        if pregrasp is None:
            pregrasp = get_pregrasp(grasp)
        self.pregrasp = pregrasp
        self.closed_position = closed_position  # closed_positions

    @property
    def value(self):
        return self.grasp

    @property
    def approach(self):
        return self.pregrasp

    def create_relative_pose(
        self, robot, link=pbu.BASE_LINK, **kwargs
    ):  # create_attachment
        parent = ParentBody(body=robot, link=link, **kwargs)
        return RelativePose(
            self.body, parent=parent, relative_pose=self.grasp, **kwargs
        )

    def create_attachment(self, *args, **kwargs):
        # TODO: create_attachment for a gripper
        relative_pose = self.create_relative_pose(*args, **kwargs)
        return relative_pose.get_attachment()

    def __repr__(self):
        return "g{}".format(id(self) % 1000)


class ParentBody(object):  # TODO: inherit from Shape?
    def __init__(self, body=None, link=pbu.BASE_LINK, **kwargs):
        self.body = body
        self.link = link

    def __iter__(self):
        return iter([self.body, self.link])

    def get_pose(self, **kwargs):
        if self.body is None:
            return pbu.unit_pose()
        return pbu.get_link_pose(self.body, self.link, **kwargs)

    # TODO: hash & equals by extending tuple
    def __repr__(self):
        return "Parent({})".format(self.body)


def extract_normal(mesh, index):
    return np.array(mesh.face_normals[index, :])


def mesh_from_obj(obj, use_concave=True, client=None, **kwargs) -> pbu.Mesh:
    # PyBullet creates multiple collision elements (with unknown_file) when nonconvex
    [data] = pbu.get_visual_data(obj, -1, client=client)
    filename = pbu.get_data_filename(data)
    if use_concave:
        filename = filename.replace("vhacd_", "")
    scale = pbu.get_data_scale(data)
    if filename == pbu.UNKNOWN_FILE:
        raise RuntimeError(filename)
    elif filename == "":
        # Unknown mesh, approximate with bounding box
        aabb = pbu.get_aabb(obj, client=client)
        aabb_center = pbu.get_aabb_center(aabb)
        centered_aabb = pbu.AABB(
            lower=aabb.lower - aabb_center, upper=aabb.upper - aabb_center
        )
        mesh = pbu.mesh_from_points(pbu.get_aabb_vertices(centered_aabb))
    else:
        mesh = pbu.read_obj(filename, decompose=False)

    vertices = [scale * np.array(vertex) for vertex in mesh.vertices]
    vertices = pbu.tform_points(pbu.get_data_pose(data), vertices)
    return pbu.Mesh(vertices, mesh.faces)


ASSET_PATH = "./models"


def get_absolute_pose(base_pose, action):
    rx = base_pose[0][0] + action.pose[0][0]
    ry = base_pose[0][1] + action.pose[0][1]
    rz = base_pose[0][2] + action.pose[0][2]
    pose = ([rx, ry, rz], action.pose[1])
    return pose


def check_collision(state, action, client=None):
    assert client is not None
    target_pose = state.block_poses[action.target_block]
    src_pose = get_absolute_pose(target_pose, action)
    for block, block_pose in state.block_poses.items():
        pbu.set_pose(block, block_pose, client=client)

    pbu.set_pose(action.grasp_block, src_pose, client=client)
    for block, block_pose in state.block_poses.items():
        if block != action.grasp_block:
            if pbu.pairwise_collision(action.grasp_block, block, client=client):
                return True
    return False


def create_pybullet_block(
    color, half_extents, mass, friction, restitution, orientation, client=None
):
    """A generic utility for creating a new block.

    Returns the PyBullet ID of the newly created block.
    """
    # The poses here are not important because they are overwritten by
    # the state values when a task is reset.
    pose = (0, 0, 0)

    # Create the collision shape.
    collision_id = client.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)

    # Create the visual_shape.
    visual_id = client.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color
    )

    # Create the body.
    block_id = client.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=pose,
        baseOrientation=orientation,
    )
    client.changeDynamics(
        block_id, linkIndex=-1, lateralFriction=friction  # -1 for the base
    )
    client.changeDynamics(
        block_id, linkIndex=-1, restitution=restitution  # -1 for the base
    )

    return block_id


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
