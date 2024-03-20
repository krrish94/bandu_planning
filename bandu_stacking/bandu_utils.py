from collections import namedtuple

import numpy as np
import pybullet as p

from pybullet_planning.pybullet_tools.utils import (
    AABB,
    UNKNOWN_FILE,
    Mesh,
    apply_affine,
    get_aabb,
    get_aabb_center,
    get_aabb_vertices,
    get_data_filename,
    get_data_pose,
    get_data_scale,
    get_visual_data,
    mesh_from_points,
    read_obj,
    set_pose,
    wait_if_gui,
)


def extract_normal(mesh, index):
    return np.array(mesh.face_normals[index, :])


def mesh_from_obj(obj, use_concave=True, client=None, **kwargs):
    # PyBullet creates multiple collision elements (with unknown_file) when nonconvex
    [data] = get_visual_data(obj, -1, client=client)
    filename = get_data_filename(data)
    if use_concave:
        filename = filename.replace("vhacd_", "")
    scale = get_data_scale(data)
    if filename == UNKNOWN_FILE:
        raise RuntimeError(filename)
    elif filename == "":
        # Unknown mesh, approximate with bounding box
        aabb = get_aabb(obj, client=client)
        aabb_center = get_aabb_center(aabb)
        centered_aabb = AABB(
            lower=aabb.lower - aabb_center, upper=aabb.upper - aabb_center
        )
        mesh = mesh_from_points(get_aabb_vertices(centered_aabb))
    else:
        mesh = read_obj(filename, decompose=False)

    vertices = [scale * np.array(vertex) for vertex in mesh.vertices]
    vertices = apply_affine(get_data_pose(data), vertices)
    return Mesh(vertices, mesh.faces)


ASSET_PATH = "./models"


def get_absolute_pose(base_pose, action):
    rx = base_pose[0][0] + action.pose[0][0]
    ry = base_pose[0][1] + action.pose[0][1]
    rz = base_pose[0][2] + action.pose[0][2]
    pose = ([rx, ry, rz], action.pose[1])
    return pose


def check_collision(state, action, client=None):
    assert client is not None
    target_pose = state.block_poses[state.block_ids.index(action.target_block)]
    src_pose = get_absolute_pose(target_pose, action)

    set_pose(action.grasp_block, src_pose, client=client)
    for block, block_pose in zip(state.block_ids, state.block_poses):
        if block != action.grasp_block:
            set_pose(block, block_pose, client=client)
            if pairwise_collision(action.grasp_block, block, client=client):
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


BASE_LINK = -1
MAX_DISTANCE = 1e-3
CollisionInfo = namedtuple(
    "CollisionInfo",
    """
                           contactFlag
                           bodyUniqueIdA
                           bodyUniqueIdB
                           linkIndexA
                           linkIndexB
                           positionOnA
                           positionOnB
                           contactNormalOnB
                           contactDistance
                           normalForce
                           lateralFriction1
                           lateralFrictionDir1
                           lateralFriction2
                           lateralFrictionDir2
                           """.split(),
)


def get_closest_points(body1, body2, max_distance=MAX_DISTANCE, client=None):
    results = client.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance)

    return [CollisionInfo(*info) for info in results]


def body_collision(body1, body2, **kwargs):
    return len(get_closest_points(body1, body2, **kwargs)) != 0


def pairwise_collision(body1, body2, **kwargs):
    return body_collision(body1, body2, **kwargs)
