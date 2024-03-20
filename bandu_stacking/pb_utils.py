from __future__ import print_function

import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, List, Optional, Tuple

import numpy as np
import pybullet as p

DEFAULT_CLIENT = None
CLIENT = 0
BASE_LINK = -1
STATIC_MASS = 0
MAX_DISTANCE = 0
NULL_ID = -1
INFO_FROM_BODY = {}
UNKNOWN_FILE = "unknown_file"
DEFAULT_RADIUS = 0.5
DEFAULT_EXTENTS = [1, 1, 1]
DEFAULT_SCALE = [1, 1, 1]
DEFAULT_NORMAL = [0, 0, 1]
DEFAULT_HEIGHT = 1


@dataclass
class RGB:
    red: int
    green: int
    blue: int

    def __list__(self):
        return [self.red, self.green, self.blue]


@dataclass
class RGBA:
    red: int
    green: int
    blue: int
    alpha: float

    def __iter__(self):
        return iter(asdict(self).values())


@dataclass
class Mesh:
    vertices: List[Tuple[float, float, float]]
    faces: List[Tuple[int, int, int]]


RED = RGBA(1, 0, 0, 1)
GREEN = RGBA(0, 1, 0, 1)
BLUE = RGBA(0, 0, 1, 0.1)
BLACK = RGBA(0, 0, 0, 1)
WHITE = RGBA(1, 1, 1, 1)
BROWN = RGBA(0.396, 0.263, 0.129, 1)
TAN = RGBA(0.824, 0.706, 0.549, 1)
GREY = RGBA(0.5, 0.5, 0.5, 1)
YELLOW = RGBA(1, 1, 0, 1)
TRANSPARENT = RGBA(0, 0, 0, 0)


@dataclass
class Interval:
    lower: float
    upper: float


UNIT_LIMITS = Interval(0.0, 1.0)
CIRCULAR_LIMITS = Interval(-np.pi, np.pi)
UNBOUNDED_LIMITS = Interval(-np.inf, np.inf)


def quat_from_euler(euler):
    return p.getQuaternionFromEuler(euler)


def Euler(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0):
    return np.array([roll, pitch, yaw])


def Point(x=0.0, y=0.0, z=0.0):
    return np.array([x, y, z])


def Pose(point: Point = None, euler: Euler = None):
    point = Point() if point is None else point
    euler = Euler() if euler is None else euler
    return point, quat_from_euler(euler)


@dataclass
class JointState:
    jointPosition: float
    jointVelocity: float
    jointReactionForces: Tuple[float, float, float, float, float, float]
    appliedJointMotorTorque: float


@dataclass
class Pixel:
    row: int
    column: int


@dataclass
class CollisionInfo:
    contactFlag: int
    bodyUniqueIdA: int
    bodyUniqueIdB: int
    linkIndexA: int
    linkIndexB: int
    positionOnA: Tuple[float, float, float]
    positionOnB: Tuple[float, float, float]
    contactNormalOnB: Tuple[float, float, float]
    contactDistance: float
    normalForce: float
    lateralFriction1: float
    lateralFrictionDir1: Tuple[float, float, float]
    lateralFriction2: float
    lateralFrictionDir2: Tuple[float, float, float]


@dataclass
class CollisionPair:
    body: int
    links: List[int]


@dataclass
class AABB:
    lower: list
    upper: list


@dataclass
class CollisionShapeData:
    object_unique_id: int
    linkIndex: int
    geometry_type: int
    dimensions: list
    filename: str
    local_frame_pos: List[float]
    local_frame_orn: List[float]


@dataclass
class BodyInfo:
    base_name: str
    body_name: str


@dataclass
class JointInfo:
    jointIndex: int
    jointName: str
    jointType: int
    qIndex: int
    uIndex: int
    flags: List[str]
    jointDamping: float
    jointFriction: float
    jointLowerLimit: float
    jointUpperLimit: float
    jointMaxForce: float
    jointMaxVelocity: float
    linkName: float
    jointAxis: float
    parentFramePos: float
    parentFrameOrn: float
    parentIndex: float


@dataclass
class LinkState:
    linkWorldPosition: Tuple[float, float, float]
    linkWorldOrientation: Tuple[float, float, float, float]
    localInertialFramePosition: Tuple[float, float, float]
    localInertialFrameOrientation: Tuple[float, float, float, float]
    worldLinkFramePosition: Tuple[float, float, float]
    worldLinkFrameOrientation: Tuple[float, float, float, float]


@dataclass
class CameraImage:
    rgbPixels: Any
    depthPixels: Any
    segmentationMaskBuffer: Any
    camera_pose: Pose
    camera_matrix: Any


@dataclass
class DynamicsInfo:
    mass: float
    lateral_friction: float
    local_inertia_diagonal: Tuple[float, float, float]
    local_inertial_pos: Tuple[float, float, float]
    local_inertial_orn: Tuple[float, float, float, float]
    restitution: float
    rolling_friction: float
    spinning_friction: float
    contact_damping: float
    contact_stiffness: float


@dataclass
class VisualShapeData:
    objectUniqueId: int
    linkIndex: int
    visualGeometryType: int
    dimensions: Optional[Tuple[float, ...]]
    meshAssetFileName: str
    localVisualFrame_position: Tuple[float, float, float]
    localVisualFrame_orientation: Tuple[float, float, float, float]
    rgbaColor: Tuple[float, float, float, float]
    textureUniqueId: int


@dataclass
class ModelInfo:
    name: str
    path: str
    fixed_base: bool
    scale: float


@dataclass
class MouseEvent:
    eventType: str
    mousePosX: int
    mousePosY: int
    buttonIndex: int
    buttonState: str


def get_urdf_flags(cache=False, cylinder=False, merge=False, sat=False, **kwargs):
    flags = 0
    if cache:
        flags |= p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    if cylinder:
        flags |= p.URDF_USE_IMPLICIT_CYLINDER
    if merge:
        flags |= p.URDF_MERGE_FIXED_LINKS
    if sat:
        flags |= p.URDF_INITIALIZE_SAT_FEATURES
    # flags |= p.URDF_USE_INERTIA_FROM_FILE
    return flags


def load_pybullet(filename, fixed_base=False, scale=1.0, client=None, **kwargs):
    # fixed_base=False implies infinite base mass
    client = client or DEFAULT_CLIENT
    with LockRenderer(client=client):
        flags = get_urdf_flags(**kwargs)
        if filename.endswith(".urdf"):
            body = client.loadURDF(
                filename, useFixedBase=fixed_base, flags=flags, globalScaling=scale
            )
        elif filename.endswith(".sdf"):
            body = client.loadSDF(filename)
        elif filename.endswith(".xml"):
            body = client.loadMJCF(filename, flags=flags)
        elif filename.endswith(".bullet"):
            body = client.loadBullet(filename)
        elif filename.endswith(".obj"):
            # TODO: fixed_base => mass = 0?
            body = create_obj(filename, scale=scale, client=client, **kwargs)
        else:
            raise ValueError(filename)
    INFO_FROM_BODY[CLIENT, body] = ModelInfo(None, filename, fixed_base, scale)
    return body


def get_connection(client=None):
    client = client or DEFAULT_CLIENT
    return client.getConnectionInfo()["connectionMethod"]


def has_gui(client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    return get_connection(client=client) == p.GUI


class Saver(object):
    # TODO: contextlib
    def save(self):
        pass

    def restore(self):
        raise NotImplementedError()

    def __enter__(self):
        # TODO: move the saving to enter?
        self.save()
        # return self

    def __exit__(self, type, value, traceback):
        self.restore()


def set_renderer(enable, client=None):
    client = client or DEFAULT_CLIENT
    if not has_gui(client=client):
        return

    client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, int(enable))


class LockRenderer(Saver):
    # disabling rendering temporary makes adding objects faster
    def __init__(self, client=None, lock=True, **kwargs):
        self.client = client or DEFAULT_CLIENT
        # skip if the visualizer isn't active
        if has_gui(client=self.client) and lock:
            set_renderer(enable=False, client=self.client)

    def restore(self):
        if not has_gui(client=self.client):
            return

        set_renderer(enable=True, client=self.client)


def create_obj(path, scale=1.0, mass=STATIC_MASS, color=GREY, **kwargs):
    collision_id, visual_id = create_shape(
        get_mesh_geometry(path, scale=scale), color=color, **kwargs
    )
    body = create_body(collision_id, visual_id, mass=mass, **kwargs)
    fixed_base = mass == STATIC_MASS
    INFO_FROM_BODY[CLIENT, body] = ModelInfo(
        None, path, fixed_base, scale
    )  # TODO: store geometry info instead?
    return body


def load_pybullet(filename, fixed_base=False, scale=1.0, client=None, **kwargs):
    # fixed_base=False implies infinite base mass
    client = client or DEFAULT_CLIENT
    with LockRenderer(client=client):
        flags = get_urdf_flags(**kwargs)
        if filename.endswith(".urdf"):
            body = client.loadURDF(
                filename, useFixedBase=fixed_base, flags=flags, globalScaling=scale
            )
        elif filename.endswith(".sdf"):
            body = client.loadSDF(filename)
        elif filename.endswith(".xml"):
            body = client.loadMJCF(filename, flags=flags)
        elif filename.endswith(".bullet"):
            body = client.loadBullet(filename)
        elif filename.endswith(".obj"):
            # TODO: fixed_base => mass = 0?
            body = create_obj(filename, scale=scale, client=client, **kwargs)
        else:
            raise ValueError(filename)
    INFO_FROM_BODY[CLIENT, body] = ModelInfo(None, filename, fixed_base, scale)
    return body


def unit_point():
    return (0.0, 0.0, 0.0)


def unit_quat():
    return quat_from_euler([0, 0, 0])  # [X,Y,Z,W]


def unit_pose():
    return (unit_point(), unit_quat())


def create_shape(geometry, pose=unit_pose(), collision=True, **kwargs):
    collision_id = (
        create_collision_shape(geometry, pose=pose, **kwargs) if collision else NULL_ID
    )
    visual_id = create_visual_shape(
        geometry, pose=pose, **kwargs
    )  # if collision else NULL_ID
    return collision_id, visual_id


def create_body(
    collision_id=NULL_ID, visual_id=NULL_ID, mass=STATIC_MASS, client=None, **kwargs
):
    client = client or DEFAULT_CLIENT
    return client.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
    )


def get_box_geometry(width, length, height):
    return {
        "shapeType": p.GEOM_BOX,
        "halfExtents": [width / 2.0, length / 2.0, height / 2.0],
    }


def create_box(w, l, h, mass=STATIC_MASS, color=RED, **kwargs):
    collision_id, visual_id = create_shape(
        get_box_geometry(w, l, h), color=color, **kwargs
    )
    return create_body(collision_id, visual_id, mass=mass, **kwargs)


def join_paths(*paths):
    return os.path.abspath(os.path.join(*paths))


def list_paths(directory):
    return sorted(join_paths(directory, filename) for filename in os.listdir(directory))


def get_max_limit(body, joint, **kwargs):
    return get_joint_limits(body, joint, **kwargs)[1]


def get_max_limits(body, joints, **kwargs):
    return [get_max_limit(body, joint, **kwargs) for joint in joints]


def get_joint_limits(body, joint, **kwargs):
    # TODO: make a version for several joints?
    if is_circular(body, joint, **kwargs):
        # TODO: return UNBOUNDED_LIMITS
        return CIRCULAR_LIMITS
    joint_info = get_joint_info(body, joint, **kwargs)
    return joint_info.jointLowerLimit, joint_info.jointUpperLimit


def get_joint_state(body, joint, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    return JointState(*client.getJointState(int(body), joint))


def get_joint_position(body, joint, **kwargs):
    return get_joint_state(body, joint, **kwargs).jointPosition


def get_joint_velocity(body, joint, **kwargs):
    return get_joint_state(body, joint, **kwargs).jointVelocity


def get_joint_velocities(body, joints, **kwargs):
    return tuple(get_joint_velocity(body, joint, **kwargs) for joint in joints)


def get_joint_positions(body, joints, **kwargs):
    return tuple(get_joint_position(body, joint, **kwargs) for joint in joints)


def get_camera_matrix(width, height, fx, fy=None):
    if fy is None:
        fy = fx
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def set_pose(body, pose, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    (point, quat) = pose
    client.resetBasePositionAndOrientation(int(body), point, quat)


def pixel_from_ray(camera_matrix, ray):
    return camera_matrix.dot(np.array(ray) / ray[2])[:2]


def pixel_from_point(camera_matrix, point_camera):
    px, py = pixel_from_ray(camera_matrix, point_camera)
    width, height = dimensions_from_camera_matrix(camera_matrix)
    if (0 <= px < width) and (0 <= py < height):
        r, c = np.floor([py, px]).astype(int)
        return Pixel(r, c)
    return None


def aabb_from_extent_center(extent, center=None):
    if center is None:
        center = np.zeros(len(extent))
    else:
        center = np.array(center)
    half_extent = np.array(extent) / 2.0
    lower = center - half_extent
    upper = center + half_extent
    return AABB(lower, upper)


def get_aabb_center(aabb):
    return (np.array(aabb.lower) + np.array(aabb.upper)) / 2.0


def get_aabb_extent(aabb):
    return np.array(aabb.upper) - np.array(aabb.lower)


def buffer_aabb(aabb, buffer):
    if (aabb is None) or (np.isscalar(buffer) and (buffer == 0.0)):
        return aabb
    extent = get_aabb_extent(aabb)
    if np.isscalar(buffer):
        # buffer = buffer - DEFAULT_AABB_BUFFER # TODO: account for the default
        buffer = buffer * np.ones(len(extent))
    new_extent = np.add(2 * buffer, extent)
    center = get_aabb_center(aabb)
    return aabb_from_extent_center(new_extent, center)


def get_link_state(body, link, kinematics=True, velocity=True, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    return LinkState(*client.getLinkState(int(body), link))


def get_link_pose(body, link, **kwargs):
    if link == BASE_LINK:
        return get_pose(body, **kwargs)
    # if set to 1 (or True), the Cartesian world position/orientation will be recomputed using forward kinematics.
    link_state = get_link_state(
        body, link, **kwargs
    )  # , kinematics=True, velocity=False)
    return link_state.worldLinkFramePosition, link_state.worldLinkFrameOrientation


def aabb_overlap(aabb1, aabb2):
    if (aabb1 is None) or (aabb2 is None):
        return False
    lower1, upper1 = aabb1
    lower2, upper2 = aabb2
    return all(l1 <= u2 for l1, u2 in zip(lower1, upper2)) and all(
        l2 <= u1 for l2, u1 in zip(lower2, upper1)
    )


def get_buffered_aabb(body, link=None, max_distance=MAX_DISTANCE, **kwargs):
    body, links = parse_body(body, link=link)
    return buffer_aabb(
        aabb_union(get_aabbs(body, links=links, **kwargs)), buffer=max_distance
    )


def set_dynamics(body, link=BASE_LINK, client=None, **kwargs):
    # TODO: iterate over all links
    client = client or DEFAULT_CLIENT
    client.changeDynamics(int(body), link)


def get_mesh_geometry(path, scale=1.0):
    return {
        "shapeType": p.GEOM_MESH,
        "fileName": path,
        "meshScale": scale * np.ones(3),
    }


def apply_alpha(color, alpha=1.0):
    if color is None:
        return None
    return RGBA(color.red, color.green, color.blue, alpha)


def create_visual_shape(
    geometry, pose=unit_pose(), color=RED, specular=None, client=None, **kwargs
):
    client = client or DEFAULT_CLIENT
    if color is None:  # or not has_gui():
        return NULL_ID
    point, quat = pose
    visual_args = {
        "rgbaColor": color,
        "visualFramePosition": point,
        "visualFrameOrientation": quat,
    }
    visual_args.update(geometry)
    # if specular is not None:
    visual_args["specularColor"] = [0, 0, 0]
    return client.createVisualShape(**visual_args)


def create_collision_shape(geometry, pose=unit_pose(), client=None, **kwargs):
    # TODO: removeCollisionShape
    # https://github.com/bulletphysics/bullet3/blob/5ae9a15ecac7bc7e71f1ec1b544a55135d7d7e32/examples/pybullet/examples/getClosestPoints.py
    client = client or DEFAULT_CLIENT
    point, quat = pose
    collision_args = {
        "collisionFramePosition": point,
        "collisionFrameOrientation": quat,
        #'flags': p.GEOM_FORCE_CONCAVE_TRIMESH,
    }
    collision_args.update(geometry)
    if "length" in collision_args:
        # TODO: pybullet bug visual => length, collision => height
        collision_args["height"] = collision_args["length"]
        del collision_args["length"]
    return client.createCollisionShape(**collision_args)


def get_closest_points(
    body1,
    body2,
    link1=None,
    link2=None,
    max_distance=MAX_DISTANCE,
    use_aabb=False,
    client=None,
    **kwargs,
):
    client = client or DEFAULT_CLIENT

    if use_aabb and not aabb_overlap(
        get_buffered_aabb(body1, link1, max_distance=max_distance / 2.0),
        get_buffered_aabb(body2, link2, max_distance=max_distance / 2.0),
    ):
        return []
    if (link1 is None) and (link2 is None):
        results = client.getClosestPoints(
            bodyA=int(body1), bodyB=int(body2), distance=max_distance
        )
    elif link2 is None:
        results = client.getClosestPoints(
            bodyA=int(body1), bodyB=int(body2), linkIndexA=link1, distance=max_distance
        )
    elif link1 is None:
        results = client.getClosestPoints(
            bodyA=int(body1), bodyB=int(body2), linkIndexB=link2, distance=max_distance
        )
    else:
        results = client.getClosestPoints(
            bodyA=int(body1),
            bodyB=int(body2),
            linkIndexA=link1,
            linkIndexB=link2,
            distance=max_distance,
        )

    if results is None:
        results = []  # Strange pybullet failure case

    return [CollisionInfo(*info) for info in results]


def body_collision(body1, body2, **kwargs):
    return len(get_closest_points(body1, body2, **kwargs)) != 0


def get_joint_name(body, joint, **kwargs):
    return get_joint_info(body, joint, **kwargs).jointName.decode("UTF-8")


def joint_from_name(body, name, **kwargs):
    for joint in get_joints(body, **kwargs):
        if get_joint_name(body, joint, **kwargs) == name:
            return joint
    raise ValueError(body, name)


def joints_from_names(body, names, **kwargs):
    return tuple(joint_from_name(body, name, **kwargs) for name in names)


def invert(pose):
    point, quat = pose
    return p.invertTransform(point, quat)


def get_joint_info(body, joint, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    return JointInfo(*client.getJointInfo(int(body), joint))


def get_link_name(body, link, **kwargs):
    if link == BASE_LINK:
        return get_base_name(body, **kwargs)
    return get_joint_info(body, link, **kwargs).linkName.decode("UTF-8")


def get_num_joints(body, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    return client.getNumJoints(int(body))


def get_joints(body, **kwargs):
    return list(range(get_num_joints(body, **kwargs)))


get_links = get_joints  # Does not include BASE_LINK


def dimensions_from_camera_matrix(camera_matrix: list):
    cx, cy = np.array(camera_matrix)[:2, 2]
    width, height = (2 * cx + 1), (2 * cy + 1)
    return width, height


def get_collision_data(body, link=BASE_LINK, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    while True:
        try:
            tups = client.getCollisionShapeData(int(body), link)
            break
        except:
            print("Pybullet error getting collision shape. Trying again.")

    return [CollisionShapeData(*tup) for tup in tups]


def can_collide(body, link=BASE_LINK, **kwargs):
    return len(get_collision_data(body, link=link, **kwargs)) != 0


def get_all_links(body, **kwargs):
    # TODO: deprecate get_links
    return [BASE_LINK] + list(get_links(body, **kwargs))


def get_aabbs(body, links=None, only_collision=True, **kwargs):
    if links is None:
        links = get_all_links(body, **kwargs)
    if only_collision:
        # TODO: return the null bounding box
        links = [link for link in links if can_collide(body, link, **kwargs)]
    return [get_aabb(body, link=link, **kwargs) for link in links]


def aabb_union(aabbs: list):
    if not aabbs:
        return None
    if len(aabbs) == 1:
        return aabbs[0]
    d = len(aabbs[0][0])
    lower = [min(aabb[0][k] for aabb in aabbs) for k in range(d)]
    upper = [max(aabb[1][k] for aabb in aabbs) for k in range(d)]
    return AABB(lower, upper)


def get_aabb(body, link=None, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    if link is None:
        return aabb_union(get_aabbs(body, client=client, **kwargs))
    return AABB(*client.getAABB(int(body), linkIndex=link))


def get_body_info(body, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    return BodyInfo(*client.getBodyInfo(int(body)))


def any_link_pair_collision(body1, links1, body2, links2=None, **kwargs):
    if links1 is None:
        links1 = get_all_links(body1, **kwargs)
    if links2 is None:
        links2 = get_all_links(body2, **kwargs)
    for link1, link2 in product(links1, links2):
        if (body1 == body2) and (link1 == link2):
            continue
        if pairwise_link_collision(body1, link1, body2, link2, **kwargs):
            return True
    return False


def expand_links(body, **kwargs):
    body, links = parse_body(body)
    if links is None:
        links = get_all_links(body, **kwargs)
    return CollisionPair(body, links)


def pairwise_collision(body1, body2, **kwargs):
    if isinstance(body1, tuple) or isinstance(body2, tuple):
        body1, links1 = expand_links(body1, **kwargs)
        body2, links2 = expand_links(body2, **kwargs)
        return any_link_pair_collision(body1, links1, body2, links2, **kwargs)
    return body_collision(body1, body2, **kwargs)


def get_base_name(body, **kwargs):
    return get_body_info(body, **kwargs).base_name.decode(encoding="UTF-8")


def link_from_name(body, name, **kwargs):
    if name == get_base_name(body, **kwargs):
        return BASE_LINK
    for link in get_joints(body, **kwargs):
        if get_link_name(body, link, **kwargs) == name:
            return link
    raise ValueError(body, name)


def parse_body(body, link=None):
    return body if isinstance(body, tuple) else CollisionPair(body, link)


def pairwise_link_collision(body1, link1, body2, link2=BASE_LINK, **kwargs):
    return (
        len(get_closest_points(body1, body2, link1=link1, link2=link2, **kwargs)) != 0
    )


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def spaced_colors(n, s=1, v=1):
    import colorsys

    return [colorsys.hsv_to_rgb(h, s, v) for h in np.linspace(0, 1, n, endpoint=False)]


def get_bodies(client=None):
    client = client or DEFAULT_CLIENT
    # Note that all APIs already return body unique ids, so you typically never need to use getBodyUniqueId if you keep track of them
    return [client.getBodyUniqueId(i) for i in range(client.getNumBodies())]


def save_image(filename, rgba):
    import imageio

    imageio.imwrite(filename, rgba)


def image_from_segmented(segmented, color_from_body=None, **kwargs):
    if color_from_body is None:
        bodies = get_bodies(**kwargs)
        color_from_body = dict(zip(bodies, spaced_colors(len(bodies))))
    image = np.zeros(segmented.shape[:2] + (3,))
    for r in range(segmented.shape[0]):
        for c in range(segmented.shape[1]):
            body, link = segmented[r, c, :]
            image[r, c, :] = list(color_from_body.get(body, BLACK))[:3]  # TODO: alpha
    return image


def save_camera_images(camera_image, directory="", prefix="", client=None, **kwargs):
    # safe_remove(directory)
    ensure_dir(directory)
    depth_image = camera_image.depthPixels
    seg_image = camera_image.segmentationMaskBuffer
    save_image(
        os.path.join(directory, "{}rgb.png".format(prefix)), camera_image.rgbPixels
    )  # [0, 255]
    depth_image = (
        (depth_image - np.min(depth_image))
        / (np.max(depth_image) - np.min(depth_image))
        * 255
    ).astype(np.uint8)
    save_image(
        os.path.join(directory, "{}depth.png".format(prefix)), depth_image
    )  # [0, 1]
    if seg_image is None:
        return None

    segmented_image = image_from_segmented(seg_image, client=client)
    save_image(
        os.path.join(directory, "{}segmented.png".format(prefix)), segmented_image
    )  # [0, 255]
    return segmented_image


def safe_zip(sequence1, sequence2):  # TODO: *args
    sequence1, sequence2 = list(sequence1), list(sequence2)
    assert len(sequence1) == len(sequence2)
    return list(zip(sequence1, sequence2))


def set_joint_positions(body, joints, values, **kwargs):
    for joint, value in safe_zip(joints, values):
        set_joint_position(body, joint, value, **kwargs)


def set_joint_position(body, joint, value, client=None, **kwargs):
    # TODO: remove targetVelocity=0
    client = client or DEFAULT_CLIENT
    client.resetJointState(int(body), joint, targetValue=value, targetVelocity=0)


def stable_z_on_aabb(body, aabb, **kwargs):
    center, extent = get_center_extent(body, **kwargs)
    return (aabb.upper + extent / 2 + (get_point(body, **kwargs) - center))[2]


def get_center_extent(body, **kwargs):
    aabb = get_aabb(body, **kwargs)
    return get_aabb_center(aabb), get_aabb_extent(aabb)


def get_point(body, **kwargs):
    return get_pose(body, **kwargs)[0]


def get_pose(body, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    return client.getBasePositionAndOrientation(int(body))


def multiply(*poses):
    pose = poses[0]
    for next_pose in poses[1:]:
        pose = p.multiplyTransforms(pose[0], pose[1], *next_pose)
    return pose


def point_from_pose(pose):
    return pose[0]


def quat_from_pose(pose):
    return pose[1]


def tform_point(affine, point):
    return point_from_pose(multiply(affine, Pose(point=point)))


def get_joint_name(body, joint, **kwargs):
    return get_joint_info(body, joint, **kwargs).jointName.decode("UTF-8")


def get_joint_names(body, joints, **kwargs):
    return [
        get_joint_name(body, joint, **kwargs) for joint in joints
    ]  # .encode('ascii')


def flatten_links(body, links=None, **kwargs):
    if links is None:
        links = get_all_links(body, **kwargs)
    return {CollisionPair(body, frozenset([link])) for link in links}


def child_link_from_joint(joint):
    link = joint
    return link


def get_link_parent(body, link, **kwargs):
    if link == BASE_LINK:
        return None
    return get_joint_info(body, link, **kwargs).parentIndex


def get_all_link_parents(body, **kwargs):
    return {
        link: get_link_parent(body, link, **kwargs)
        for link in get_links(body, **kwargs)
    }


def get_all_link_children(body, **kwargs):
    children = {}
    for child, parent in get_all_link_parents(body, **kwargs).items():
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
    return children


def get_link_children(body, link, **kwargs):
    children = get_all_link_children(body, **kwargs)
    return children.get(link, [])


def get_link_descendants(body, link, test=lambda l: True, **kwargs):
    descendants = []
    for child in get_link_children(body, link, **kwargs):
        if test(child):
            descendants.append(child)
            descendants.extend(get_link_descendants(body, child, test=test, **kwargs))
    return descendants


def get_link_subtree(body, link, **kwargs):
    return [link] + get_link_descendants(body, link, **kwargs)


def get_moving_links(body, joints, **kwargs):
    moving_links = set()
    for joint in joints:
        link = child_link_from_joint(joint)
        if link not in moving_links:
            moving_links.update(get_link_subtree(body, link, **kwargs))
    return list(moving_links)


def parent_joint_from_link(link):
    # note that link index == joint index
    joint = link
    return joint


def get_field_of_view(camera_matrix):
    dimensions = np.array(dimensions_from_camera_matrix(camera_matrix))
    focal_lengths = np.array([camera_matrix[i, i] for i in range(2)])
    return 2 * np.arctan(np.divide(dimensions, 2 * focal_lengths))


def get_joint_descendants(body, link, **kwargs):
    return list(map(parent_joint_from_link, get_link_descendants(body, link, **kwargs)))


def get_movable_joint_descendants(body, link, **kwargs):
    return prune_fixed_joints(
        body, get_joint_descendants(body, link, **kwargs), **kwargs
    )


def get_projection_matrix(
    width, height, vertical_fov, near, far, client=None, **kwargs
):
    client = client or DEFAULT_CLIENT
    aspect = float(width) / height
    fov_degrees = math.degrees(vertical_fov)
    projection_matrix = client.computeProjectionMatrixFOV(
        fov=fov_degrees, aspect=aspect, nearVal=near, farVal=far
    )
    return projection_matrix


def compiled_with_numpy():
    return bool(p.isNumpyEnabled())


def get_image_flags(segment=False, segment_links=False):
    if segment:
        if segment_links:
            return p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        return 0  # TODO: adjust output dimension when not segmenting links
    return p.ER_NO_SEGMENTATION_MASK


def point_from_tform(tform):
    return np.array(tform)[:3, 3]


def demask_pixel(pixel):
    body = pixel & ((1 << 24) - 1)
    link = (pixel >> 24) - 1
    return body, link


def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True
    """
    q = np.empty((4,), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


def matrix_from_tform(tform):
    return np.array(tform)[:3, :3]


def quat_from_matrix(rot):
    matrix = np.eye(4)
    matrix[:3, :3] = rot[:3, :3]
    return quaternion_from_matrix(matrix)


def pose_from_tform(tform):
    return point_from_tform(tform), quat_from_matrix(matrix_from_tform(tform))


def extract_segmented(seg_image):
    segmented = np.zeros(seg_image.shape + (2,))
    for r in range(segmented.shape[0]):
        for c in range(segmented.shape[1]):
            pixel = seg_image[r, c]
            segmented[r, c, :] = demask_pixel(pixel)
    return segmented


def get_focal_lengths(dims, fovs):
    return np.divide(dims, np.tan(fovs / 2)) / 2


def get_image(
    camera_pos=None,
    target_pos=None,
    width=640,
    height=480,
    vertical_fov=60.0,
    near=0.02,
    far=5.0,
    tiny=False,
    segment=False,
    client=None,
    **kwargs,
):
    client = client or DEFAULT_CLIENT
    up_vector = [0, 0, 1]  # up vector of the camera, in Cartesian world coordinates
    camera_flags = {}
    view_matrix = None
    if (camera_pos is None) or (target_pos is None):
        pass
    else:
        view_matrix = client.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=up_vector,
        )
        camera_flags["viewMatrix"] = view_matrix
    projection_matrix = get_projection_matrix(
        width, height, vertical_fov, near, far, client=client
    )

    flags = get_image_flags(segment=segment, **kwargs)
    renderer = p.ER_TINY_RENDERER if tiny else p.ER_BULLET_HARDWARE_OPENGL
    width, height, rgb, d, seg = client.getCameraImage(
        width,
        height,
        projectionMatrix=projection_matrix,
        shadow=False,
        flags=flags,
        renderer=renderer,
        **camera_flags,
    )
    if not compiled_with_numpy():
        rgb = np.reshape(rgb, [height, width, -1])  # 4
        d = np.reshape(d, [height, width])
        seg = np.reshape(seg, [height, width])

    depth = far * near / (far - (far - near) * d)
    segmented = None
    if segment:
        segmented = extract_segmented(seg)

    if view_matrix is None:
        view_matrix = np.identity(4)  # TODO: hack
    camera_tform = np.reshape(view_matrix, [4, 4])  # TODO: transpose?
    camera_tform[:3, 3] = camera_pos
    camera_tform[3, :3] = 0

    view_pose = multiply(pose_from_tform(camera_tform), Pose(euler=Euler(roll=np.pi)))

    focal_length = get_focal_lengths(height, vertical_fov)  # TODO: horizontal_fov
    camera_matrix = get_camera_matrix(width, height, focal_length)

    return CameraImage(rgb, depth, segmented, view_pose, camera_matrix)


def get_image_at_pose(camera_pose, camera_matrix, far=5.0, **kwargs):
    width, height = map(int, dimensions_from_camera_matrix(camera_matrix))
    _, vertical_fov = get_field_of_view(camera_matrix)
    camera_point = point_from_pose(camera_pose)
    target_point = tform_point(camera_pose, np.array([0, 0, far]))
    return get_image(
        camera_point,
        target_point,
        width=width,
        height=height,
        vertical_fov=vertical_fov,
        far=far,
        **kwargs,
    )


def get_data_type(data):
    return (
        data.geometry_type
        if isinstance(data, CollisionShapeData)
        else data.visualGeometryType
    )


def get_data_radius(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_SPHERE:
        return dimensions[0]
    if geometry_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
        return dimensions[1]
    return DEFAULT_RADIUS


def get_joint_inertial_pose(body, joint, **kwargs):
    dynamics_info = get_dynamics_info(body, joint, **kwargs)
    return dynamics_info.local_inertial_pos, dynamics_info.local_inertial_orn


def get_data_pose(data):
    if isinstance(data, CollisionShapeData):
        return (data.local_frame_pos, data.local_frame_orn)
    return (data.localVisualFrame_position, data.localVisualFrame_orientation)


def get_data_extents(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_BOX:
        return dimensions
    return DEFAULT_EXTENTS


def get_data_height(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
        return dimensions[0]
    return DEFAULT_HEIGHT


def get_data_scale(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_MESH:
        return dimensions
    return DEFAULT_SCALE


def get_data_normal(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_PLANE:
        return dimensions
    return DEFAULT_NORMAL


def collision_shape_from_data(data, body, link, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    filename = data.filename.decode(encoding="UTF-8")
    if (data.geometry_type == p.GEOM_MESH) and (filename == UNKNOWN_FILE):
        return NULL_ID
    pose = multiply(
        get_joint_inertial_pose(body, link, client=client), get_data_pose(data)
    )
    point, quat = pose

    return client.createCollisionShape(
        shapeType=data.geometry_type,
        radius=get_data_radius(data),
        halfExtents=np.array(get_data_extents(data)) / 2,
        height=get_data_height(data),
        fileName=filename,
        meshScale=get_data_scale(data),
        planeNormal=get_data_normal(data),
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
        collisionFramePosition=point,
        collisionFrameOrientation=quat,
    )


def get_length(vec, norm=2):
    return np.linalg.norm(vec, ord=norm)


def get_difference(p1, p2):
    assert len(p1) == len(p2)
    return np.array(p2) - np.array(p1)


def get_distance(p1, p2, **kwargs):
    return get_length(get_difference(p1, p2), **kwargs)


def get_relative_pose(body, link1, link2=BASE_LINK, **kwargs):
    world_from_link1 = get_link_pose(body, link1, **kwargs)
    world_from_link2 = get_link_pose(body, link2, **kwargs)
    link2_from_link1 = multiply(invert(world_from_link2), world_from_link1)
    return link2_from_link1


def is_circular(body, joint, **kwargs):
    joint_info = get_joint_info(body, joint, **kwargs)
    if joint_info.jointType == p.JOINT_FIXED:
        return False
    return joint_info.jointUpperLimit < joint_info.jointLowerLimit


def get_custom_limits(
    body, joints, custom_limits={}, circular_limits=UNBOUNDED_LIMITS, **kwargs
):
    joint_limits = []
    for joint in joints:
        if joint in custom_limits:
            joint_limits.append(custom_limits[joint])
        elif is_circular(body, joint, **kwargs):
            joint_limits.append(circular_limits)
        else:
            joint_limits.append(get_joint_limits(body, joint, **kwargs))
    return zip(*joint_limits)


def get_movable_joints(body, **kwargs):
    return prune_fixed_joints(body, get_joints(body, **kwargs), **kwargs)


def get_joint_type(body, joint, **kwargs):
    return get_joint_info(body, joint, **kwargs).jointType


def is_fixed(body, joint, **kwargs):
    return get_joint_type(body, joint, **kwargs) == p.JOINT_FIXED


def is_movable(body, joint, **kwargs):
    return not is_fixed(body, joint, **kwargs)


def prune_fixed_joints(body, joints, **kwargs):
    return [joint for joint in joints if is_movable(body, joint, **kwargs)]


def set_joint_state(body, joint, position, velocity, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    client.resetJointState(
        int(body), joint, targetValue=position, targetVelocity=velocity
    )


def set_joint_states(body, joints, positions, velocities, **kwargs):
    assert len(joints) == len(positions) == len(velocities)
    for joint, position, velocity in zip(joints, positions, velocities):
        set_joint_state(body, joint, position, velocity, **kwargs)


def get_dynamics_info(body, link=BASE_LINK, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    return DynamicsInfo(
        *client.getDynamicsInfo(int(body), link)[: len(DynamicsInfo._fields)]
    )


def get_client(client=None):
    if client is None:
        return CLIENT
    return client


def clone_collision_shape(body, link, client=None):
    client = get_client(client)
    collision_data = get_collision_data(body, link, client=client)
    if not collision_data:
        return NULL_ID
    assert len(collision_data) == 1
    # TODO: can do CollisionArray
    try:
        return collision_shape_from_data(collision_data[0], body, link, client=client)
    except p.error:
        return NULL_ID


def get_visual_data(body, link=BASE_LINK, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    flags = p.VISUAL_SHAPE_DATA_TEXTURE_UNIQUE_IDS
    visual_data = [
        VisualShapeData(*tup) for tup in client.getVisualShapeData(int(body), flags)
    ]
    # return visual_data
    return list(filter(lambda d: d.linkIndex == link, visual_data))


def visual_shape_from_data(data, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    if (data.visualGeometryType == p.GEOM_MESH) and (
        data.meshAssetFileName == UNKNOWN_FILE
    ):
        return NULL_ID
    point, quat = get_data_pose(data)
    return client.createVisualShape(
        shapeType=data.visualGeometryType,
        radius=get_data_radius(data),
        halfExtents=np.array(get_data_extents(data)) / 2,
        length=get_data_height(data),
        fileName=data.meshAssetFileName,
        meshScale=get_data_scale(data),
        planeNormal=get_data_normal(data),
        rgbaColor=data.rgbaColor,
        visualFramePosition=point,
        visualFrameOrientation=quat,
    )


def clone_visual_shape(body, link, client=None):
    client = client or DEFAULT_CLIENT
    visual_data = get_visual_data(body, link)
    if not visual_data:
        return NULL_ID
    assert len(visual_data) == 1
    return visual_shape_from_data(visual_data[0], client=client)


def get_joint_parent_frame(body, joint, **kwargs):
    joint_info = get_joint_info(body, joint, **kwargs)
    return joint_info.parentFramePos, joint_info.parentFrameOrn


def get_local_link_pose(body, joint, **kwargs):
    parent_joint = get_link_parent(body, joint, **kwargs)
    parent_com = get_joint_parent_frame(body, joint, **kwargs)
    tmp_pose = invert(
        multiply(get_joint_inertial_pose(body, joint, **kwargs), parent_com)
    )
    parent_inertia = get_joint_inertial_pose(body, parent_joint, **kwargs)
    # return multiply(parent_inertia, tmp_pose) # TODO: why is this wrong...
    _, orn = multiply(parent_inertia, tmp_pose)
    pos, _ = multiply(parent_inertia, Pose(parent_com[0]))
    return (pos, orn)


def clone_body(body, links=None, collision=True, visual=True, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    if links is None:
        links = get_links(body)
    # movable_joints = [joint for joint in links if is_movable(body, joint)]
    new_from_original = {}
    base_link = (
        get_link_parent(body, links[0], client=client, **kwargs) if links else BASE_LINK
    )
    new_from_original[base_link] = NULL_ID

    masses = []
    collision_shapes = []
    visual_shapes = []
    positions = []  # list of local link positions, with respect to parent
    orientations = []  # list of local link orientations, w.r.t. parent
    inertial_positions = []  # list of local inertial frame pos. in link frame
    inertial_orientations = []  # list of local inertial frame orn. in link frame
    parent_indices = []
    joint_types = []
    joint_axes = []
    for i, link in enumerate(links):
        new_from_original[link] = i
        joint_info = get_joint_info(body, link, client=client)
        dynamics_info = get_dynamics_info(body, link, client=client)
        masses.append(dynamics_info.mass)
        collision_shapes.append(
            clone_collision_shape(body, link, client=client) if collision else NULL_ID
        )
        visual_shapes.append(
            clone_visual_shape(body, link, client=client) if visual else NULL_ID
        )
        point, quat = get_local_link_pose(body, link, client=client)
        positions.append(point)
        orientations.append(quat)
        inertial_positions.append(dynamics_info.local_inertial_pos)
        inertial_orientations.append(dynamics_info.local_inertial_orn)
        parent_indices.append(new_from_original[joint_info.parentIndex] + 1)
        joint_types.append(joint_info.jointType)
        joint_axes.append(joint_info.jointAxis)

    base_dynamics_info = get_dynamics_info(body, base_link, client=client)
    base_point, base_quat = get_link_pose(body, base_link, client=client)
    new_body = client.createMultiBody(
        baseMass=base_dynamics_info.mass,
        baseCollisionShapeIndex=(
            clone_collision_shape(body, base_link, client=client)
            if collision
            else NULL_ID
        ),
        baseVisualShapeIndex=(
            clone_visual_shape(body, base_link, client=client) if visual else NULL_ID
        ),
        basePosition=base_point,
        baseOrientation=base_quat,
        baseInertialFramePosition=base_dynamics_info.local_inertial_pos,
        baseInertialFrameOrientation=base_dynamics_info.local_inertial_orn,
        linkMasses=masses,
        linkCollisionShapeIndices=collision_shapes,
        linkVisualShapeIndices=visual_shapes,
        linkPositions=positions,
        linkOrientations=orientations,
        linkInertialFramePositions=inertial_positions,
        linkInertialFrameOrientations=inertial_orientations,
        linkParentIndices=parent_indices,
        linkJointTypes=joint_types,
        linkJointAxis=joint_axes,
    )
    # set_configuration(new_body, get_joint_positions(body, movable_joints)) # Need to use correct client
    for joint, value in zip(
        range(len(links)), get_joint_positions(body, links, client=client)
    ):
        # TODO: check if movable?
        client.resetJointState(int(new_body), joint, value, targetVelocity=0)
    return new_body


def remove_body(body, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    if (CLIENT, body) in INFO_FROM_BODY:
        del INFO_FROM_BODY[CLIENT, body]
    return client.removeBody(int(body))


def set_color(body, color, link=BASE_LINK, shape_index=NULL_ID, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    if link is None:
        return set_all_color(body, color, **kwargs)
    return client.changeVisualShape(
        int(body), link, shapeIndex=shape_index, rgbaColor=color
    )


def set_all_color(body, color, **kwargs):
    for link in get_all_links(body, **kwargs):
        set_color(body, color, link, **kwargs)


def get_aabb_vertices(aabb):
    d = len(aabb[0])
    return [
        tuple(aabb[i[k]][k] for k in range(d))
        for i in product(range(len(aabb)), repeat=d)
    ]


def wait_if_gui(*args, **kwargs):
    if has_gui(**kwargs):
        wait_for_user(*args, **kwargs)


def get_mouse_events():
    return list(MouseEvent(*event) for event in p.getMouseEvents())


def update_viewer():
    get_mouse_events()


def elapsed_time(start_time):
    return time.time() - start_time


def is_darwin():
    return platform.system() == "Darwin"


def wait_for_duration(duration):
    t0 = time.time()
    while elapsed_time(t0) <= duration:
        update_viewer()


def wait_for_user(message="Press enter to continue", **kwargs):
    if has_gui(**kwargs) and is_darwin():
        return threaded_input(message)
    return input(message)


def threaded_input(*args, **kwargs):
    import threading

    data = []
    thread = threading.Thread(
        target=lambda: data.append(input(*args, **kwargs)), args=[]
    )
    thread.start()
    try:
        while thread.is_alive():
            update_viewer()
    finally:
        thread.join()
    return data[-1]


def get_data_path():
    import pybullet_data

    return pybullet_data.getDataPath()


def add_data_path(data_path=None):
    if data_path is None:
        data_path = get_data_path()
    p.setAdditionalSearchPath(data_path)
    return data_path


def get_pitch(point):
    dx, dy, dz = point
    return np.math.atan2(dz, np.sqrt(dx**2 + dy**2))


def get_yaw(point):
    dx, dy = point[:2]
    return np.math.atan2(dy, dx)


def set_camera_pose(camera_point, target_point=np.zeros(3), client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    delta_point = np.array(target_point) - np.array(camera_point)
    distance = np.linalg.norm(delta_point)
    yaw = get_yaw(delta_point) - np.pi / 2
    pitch = get_pitch(delta_point)
    client.resetDebugVisualizerCamera(
        distance, math.degrees(yaw), math.degrees(pitch), target_point
    )


def get_data_filename(data):
    return (
        data.filename
        if isinstance(data, CollisionShapeData)
        else data.meshAssetFileName
    ).decode(encoding="UTF-8")


def convex_hull(points):
    from scipy.spatial import ConvexHull

    # TODO: cKDTree is faster, but KDTree can do all pairs closest
    hull = ConvexHull(list(points), incremental=False)
    new_indices = {i: ni for ni, i in enumerate(hull.vertices)}
    vertices = hull.points[hull.vertices, :]
    faces = np.vectorize(lambda i: new_indices[i])(hull.simplices)
    return Mesh(vertices.tolist(), faces.tolist())


def get_unit_vector(vec):
    norm = get_length(vec)
    if norm == 0:
        return vec
    return np.array(vec) / norm


def get_normal(v1, v2, v3):
    return get_unit_vector(np.cross(np.array(v3) - v1, np.array(v2) - v1))


def orient_face(vertices, face, point=None):
    if point is None:
        point = np.average(vertices, axis=0)
    v1, v2, v3 = vertices[face]
    normal = get_normal(v1, v2, v3)
    if normal.dot(point - v1) < 0:
        face = face[::-1]
    return tuple(face)


def read(filename):
    with open(filename, "r") as f:
        return f.read()


def write(filename, string):
    with open(filename, "w") as f:
        f.write(string)


def mesh_from_points(points, under=True):
    vertices, faces = map(np.array, convex_hull(points))
    centroid = np.average(vertices, axis=0)
    new_faces = [orient_face(vertices, face, point=centroid) for face in faces]
    if under:
        new_faces.extend(map(tuple, map(reversed, list(new_faces))))
    return Mesh(vertices.tolist(), new_faces)


def read_obj(path, decompose=True):
    mesh = Mesh([], [])
    meshes = {}
    vertices = []
    faces = []
    for line in read(path).split("\n"):
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "o":
            name = tokens[1]
            mesh = Mesh([], [])
            meshes[name] = mesh
        elif tokens[0] == "v":
            vertex = tuple(map(float, tokens[1:4]))
            vertices.append(vertex)
        elif tokens[0] in ("vn", "s"):
            pass
        elif tokens[0] == "f":
            face = tuple(int(token.split("/")[0]) - 1 for token in tokens[1:])
            faces.append(face)
            mesh.faces.append(face)
    if not decompose:
        return Mesh(vertices, faces)

    for name, mesh in meshes.items():
        indices = sorted({i for face in mesh.faces for i in face})
        mesh.vertices[:] = [vertices[i] for i in indices]
        new_index_from_old = {i2: i1 for i1, i2 in enumerate(indices)}
        mesh.faces[:] = [
            tuple(new_index_from_old[i1] for i1 in face) for face in mesh.faces
        ]
    return meshes


def tform_points(affine, points):
    return [tform_point(affine, p) for p in points]


def sample_curve(positions_curve, time_step=1e-2):
    start_time = positions_curve.x[0]
    end_time = positions_curve.x[-1]
    times = np.append(np.arange(start_time, end_time, step=time_step), [end_time])
    for t in times:
        q = positions_curve(t)
        yield t, q


def get_velocity(body, client=None):
    client = client or DEFAULT_CLIENT
    linear, angular = client.getBaseVelocity(int(body))
    return linear, angular  # [x,y,z], [wx,wy,wz]


def set_velocity(body, linear=None, angular=None, client=None, **kwargs):
    client = client or DEFAULT_CLIENT
    if linear is not None:
        client.resetBaseVelocity(int(body), linearVelocity=linear)
    if angular is not None:
        client.resetBaseVelocity(int(body), angularVelocity=angular)


class PoseSaver(Saver):
    def __init__(self, body, pose=None, client=None):
        self.client = client
        self.body = body
        if pose is None:
            pose = get_pose(self.body, client=client)
        self.pose = pose
        self.velocity = get_velocity(self.body, client=client)

    def apply_mapping(self, mapping):
        self.body = mapping.get(self.body, self.body)

    def restore(self):
        set_pose(self.body, self.pose, client=self.client)
        set_velocity(self.body, *self.velocity, client=self.client)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.body)


class ConfSaver(Saver):
    def __init__(self, body, joints=None, positions=None, client=None, **kwargs):
        self.body = body
        self.client = client
        if joints is None:
            joints = get_movable_joints(self.body, client=self.client)
        self.joints = joints
        if positions is None:
            positions = get_joint_positions(self.body, self.joints, client=self.client)
        self.positions = positions
        self.velocities = get_joint_velocities(
            self.body, self.joints, client=self.client
        )

    @property
    def conf(self):
        return self.positions

    def apply_mapping(self, mapping):
        self.body = mapping.get(self.body, self.body)

    def restore(self):
        set_joint_states(
            self.body, self.joints, self.positions, self.velocities, client=self.client
        )

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.body)


class BodySaver(Saver):
    def __init__(self, body, client=None, **kwargs):
        self.body = body
        self.client = client
        self.pose_saver = PoseSaver(body, client=client)
        self.conf_saver = ConfSaver(body, client=client, **kwargs)
        self.savers = [self.pose_saver, self.conf_saver]

    def apply_mapping(self, mapping):
        for saver in self.savers:
            saver.apply_mapping(mapping)

    def restore(self):
        for saver in self.savers:
            saver.restore()

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.body)


class WorldSaver(Saver):
    def __init__(self, bodies=None, client=None, **kwargs):
        if bodies is None:
            bodies = get_bodies(client=client, **kwargs)
        self.bodies = bodies
        self.client = client
        self.body_savers = [BodySaver(body, client=client) for body in self.bodies]

    def restore(self, **kwargs):
        for body_saver in self.body_savers:
            body_saver.restore(**kwargs)


def body_from_end_effector(end_effector_pose, grasp_pose):
    """world_from_parent * parent_from_child = world_from_child."""
    return multiply(end_effector_pose, grasp_pose)


class Attachment(object):
    def __init__(self, parent, parent_link, grasp_pose, child, **kwargs):
        self.parent = parent
        self.parent_link = parent_link
        self.grasp_pose = grasp_pose
        self.child = child

    @property
    def bodies(self):
        return flatten_links(self.child) | flatten_links(
            self.parent, get_link_subtree(self.parent, self.parent_link)
        )

    def assign(self, **kwargs):
        parent_link_pose = get_link_pose(self.parent, self.parent_link, **kwargs)
        child_pose = body_from_end_effector(parent_link_pose, self.grasp_pose)
        set_pose(self.child, child_pose, **kwargs)
        return child_pose

    def apply_mapping(self, mapping):
        self.parent = mapping.get(self.parent, self.parent)
        self.child = mapping.get(self.child, self.child)

    def __repr__(self):
        return "{}({},{})".format(self.__class__.__name__, self.parent, self.child)


def pairwise_collisions(body, obstacles, link=None, **kwargs):
    return any(
        pairwise_collision(body1=body, body2=other, link1=link, **kwargs)
        for other in obstacles
        if body != other
    )


DEFAULT_RESOLUTION = math.radians(3)  # 0.05


def get_default_resolution(body, joint, **kwargs):
    joint_type = get_joint_type(body, joint, **kwargs)
    if joint_type == p.JOINT_REVOLUTE:
        return math.radians(3)  # 0.05
    elif joint_type == p.JOINT_PRISMATIC:
        return 0.02
    return DEFAULT_RESOLUTION


def wrap_interval(value, interval=UNIT_LIMITS, **kwargs):
    lower, upper = interval
    if (lower == -np.inf) and (np.inf == upper):
        return value
    assert -np.inf < lower <= upper < np.inf
    return (value - lower) % (upper - lower) + lower


def interval_difference(value2, value1, interval=UNIT_LIMITS):
    value2 = wrap_interval(value2, interval)
    value1 = wrap_interval(value1, interval)
    lower, upper = interval
    straight_distance = value2 - value1
    if value2 >= value1:
        wrap_difference = (lower - value1) + (value2 - upper)
    else:
        wrap_difference = (upper - value1) + (value2 - lower)
    # return [straight_distance, wrap_difference]
    if abs(wrap_difference) < abs(straight_distance):
        return wrap_difference
    return straight_distance


def interval_distance(value1, value2, **kwargs):
    return abs(interval_difference(value2, value1, **kwargs))


def circular_interval(lower=-np.pi):  # [-np.pi, np.pi)
    return Interval(lower, lower + 2 * np.pi)


def wrap_angle(theta, **kwargs):
    return wrap_interval(theta, interval=circular_interval())


def circular_difference(theta2, theta1, **kwargs):
    interval = circular_interval()
    extent = get_aabb_extent(interval)
    diff_interval = Interval(-extent / 2, +extent / 2)
    difference = wrap_interval(theta2 - theta1, interval=diff_interval)
    return difference


def get_difference_fn(body, joints, **kwargs):
    circular_joints = [is_circular(body, joint, **kwargs) for joint in joints]

    def fn(q2, q1):
        return tuple(
            circular_difference(value2, value1) if circular else (value2 - value1)
            for circular, value2, value1 in zip(circular_joints, q2, q1)
        )

    return fn


def wrap_position(body, joint, position, **kwargs):
    if is_circular(body, joint, **kwargs):
        return wrap_angle(position, **kwargs)
    return position


def wrap_positions(body, joints, positions, **kwargs):
    assert len(joints) == len(positions)
    return [
        wrap_position(body, joint, position, **kwargs)
        for joint, position in zip(joints, positions)
    ]


def get_refine_fn(body, joints, num_steps=0, **kwargs):
    difference_fn = get_difference_fn(body, joints, **kwargs)
    num_steps = num_steps + 1

    def fn(q1, q2):
        q = q1
        for i in range(num_steps):
            positions = (1.0 / (num_steps - i)) * np.array(difference_fn(q2, q)) + q
            q = tuple(wrap_positions(body, joints, positions, **kwargs))
            yield q

    return fn


def get_default_resolutions(body, joints, resolutions=None, **kwargs):
    if resolutions is not None:
        return resolutions
    return np.array([get_default_resolution(body, joint, **kwargs) for joint in joints])


def get_extend_fn(body, joints, resolutions=None, norm=2, **kwargs):
    # norm = 1, 2, INF
    resolutions = get_default_resolutions(body, joints, resolutions, **kwargs)
    difference_fn = get_difference_fn(body, joints, **kwargs)

    def fn(q1, q2):
        # steps = int(np.max(np.abs(np.divide(difference_fn(q2, q1), resolutions))))
        steps = int(
            np.linalg.norm(np.divide(difference_fn(q2, q1), resolutions), ord=norm)
        )
        refine_fn = get_refine_fn(body, joints, num_steps=steps, **kwargs)
        return refine_fn(q1, q2)

    return fn


def interpolate_joint_waypoints(
    body,
    joints,
    waypoints,
    resolutions=None,
    collision_fn=lambda *args, **kwargs: False,
    **kwargs,
):
    # TODO: unify with refine_path
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions, **kwargs)
    path = waypoints[:1]
    for waypoint in waypoints[1:]:
        assert len(joints) == len(waypoint)
        for q in list(extend_fn(path[-1], waypoint)):
            if collision_fn(q):
                return None
            path.append(q)  # TODO: could instead yield
    return path
