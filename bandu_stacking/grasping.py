from __future__ import print_function

import random
from collections import namedtuple
from heapq import heapify, heappop, heappush
from itertools import islice

import numpy as np
import trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector

from bandu_stacking.pb_utils import (
    AABB,
    RED,
    UNKNOWN_FILE,
    Euler,
    Mesh,
    Point,
    Pose,
    PoseSaver,
    add_line,
    angle_between,
    convex_combination,
    euler_from_quat,
    get_aabb,
    get_aabb_center,
    get_aabb_vertices,
    get_data_filename,
    get_data_pose,
    get_data_scale,
    get_distance,
    get_length,
    get_pose,
    get_time_step,
    get_unit_vector,
    get_visual_data,
    get_yaw,
    invert,
    mesh_from_points,
    multiply,
    multiply_quats,
    pairwise_collision,
    quat_from_euler,
    quat_from_matrix,
    read_obj,
    remove_handles,
    set_pose,
    single_collision,
    tform_point,
    tform_points,
    unit_pose,
)

X_AXIS = np.array([1, 0, 0])  # TODO: make immutable
Z_AXIS = np.array([0, 0, 1])

Plane = namedtuple("Plane", ["normal", "origin"])


def point_plane_distance(plane, point, signed=True):
    # TODO: reorder name
    plane_normal, plane_point = plane
    # from trimesh.points import point_plane_distance # NOTE: the output is signed
    signed_distance = np.dot(plane_normal, np.array(point) - np.array(plane_point))
    if signed:
        return signed_distance
    return abs(signed_distance)


def project_plane(plane, point):
    # from trimesh.points import project_to_plane
    normal, _ = plane
    return np.array(point) - point_plane_distance(plane, point) * normal


def get_plane_quat(normal):
    from trimesh.points import plane_transform

    # from trimesh.geometry import align_vectors
    # transform that will move that plane to be coplanar with the XY plane
    plane = Plane(normal, np.zeros(3))
    normal, origin = plane
    tform = np.linalg.inv(plane_transform(origin, -normal))  # origin=None
    quat1 = quat_from_matrix(tform)
    pose1 = Pose(origin, euler=euler_from_quat(quat1))

    projection_world = project_plane(plane, Z_AXIS)
    projection = tform_point(invert(pose1), projection_world)
    yaw = get_yaw(projection[:2])
    quat2 = multiply_quats(quat1, quat_from_euler(Euler(yaw=yaw)))

    return quat2


PREGRASP_DISTANCE = 0.05  # 0.05 | 0.07
POSTGRASP_DISTANCE = 0.005  # 0.05 | 0.07

# FINGER_LENGTH = PR2_FINGER_DIMENSIONS[1] / 2.
FINGER_LENGTH = 0.0  # 0. | 0.01 | 0.015 | 0.02

ScoredGrasp = namedtuple("ScoredGrasp", ["pose", "contact1", "contact2", "score"])

##################################################


def sample_sphere_surface(d, uniform=True):
    # TODO: hyperspherical coordinates
    # https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    while True:
        v = np.random.randn(d)
        r = np.sqrt(v.dot(v))
        if not uniform or (r <= 1.0):
            return v / r


def control_until_contact(
    controller, body, contact_links=[], time_after_contact=np.inf, all_contacts=True
):
    # TODO: unify with close_gripper
    # TODO: list different grasping control strategies
    agg = all if all_contacts else any
    if (time_after_contact is np.inf) or not contact_links:
        for output in controller:
            yield output
        return
    dt = get_time_step()
    countdown = np.inf
    for output in controller:  # TODO: force control for grasping
        if countdown <= 0.0:
            break

        if (countdown == np.inf) and agg(
            single_collision(body, link=link) for link in contact_links
        ):  # any | all
            # (time_after_contact != np.inf) and contact_links
            countdown = time_after_contact
            print(
                "Contact! Simulating for an additional {:.3f} sec".format(
                    time_after_contact
                )
            )
            # break
        yield output
        countdown -= dt


##################################################


def get_grasp(grasp_tool, gripper_from_tool=unit_pose()):
    return multiply(gripper_from_tool, grasp_tool)


def get_pregrasp(
    grasp_tool,
    gripper_from_tool=unit_pose(),
    tool_distance=PREGRASP_DISTANCE,
    object_distance=PREGRASP_DISTANCE,
):
    # TODO: rename to approach, standoff, guarded, ...
    return multiply(
        gripper_from_tool,
        Pose(Point(x=tool_distance)),
        grasp_tool,
        Pose(Point(z=-object_distance)),
    )


def get_postgrasp(
    grasp_tool,
    gripper_from_tool=unit_pose(),
    tool_distance=-POSTGRASP_DISTANCE,
    object_distance=-POSTGRASP_DISTANCE,
):
    # TODO: rename to approach, standoff, guarded, ...
    return multiply(
        gripper_from_tool,
        Pose(Point(x=tool_distance)),
        grasp_tool,
        Pose(Point(z=-object_distance)),
    )


##################################################


def filter_grasps(
    gripper,
    obj,
    grasp_generator,
    gripper_from_tool=unit_pose(),
    obstacles=[],
    draw=False,
    **kwargs,
):
    obj_pose = get_pose(obj)
    for grasp_tool in grasp_generator:
        if grasp_tool is None:
            continue
        grasp_pose = multiply(
            obj_pose, invert(get_grasp(grasp_tool, gripper_from_tool))
        )
        with PoseSaver(gripper):
            set_pose(gripper, grasp_pose)  # grasp_pose | pregrasp_pose
            if any(pairwise_collision(gripper, obst) for obst in [obj] + obstacles):
                continue
            yield grasp_tool


##################################################


def mesh_from_obj(obj, use_concave=True, client=None, **kwargs):
    # PyBullet creates multiple collision elements (with unknown_file) when nonconvex
    [data] = get_visual_data(obj, -1, client=client)
    filename = get_data_filename(data)
    if use_concave:
        filename = filename.replace("textured", "textured_vhacd")

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
    vertices = tform_points(get_data_pose(data), vertices)
    return Mesh(vertices, mesh.faces)


def extract_normal(mesh, index):
    return np.array(mesh.face_normals[index, :])


def sample_grasp(
    obj,
    point1,
    point2,
    normal1,
    normal2,
    pitches=[-np.pi, np.pi],
    discrete_pitch=False,
    finger_length=FINGER_LENGTH,
    draw=False,
    **kwargs,
):
    grasp_point = convex_combination(point1, point2)
    direction2 = point2 - point1
    quat = get_plane_quat(direction2)  # Biases toward the smallest rotation to align
    pitches = sorted(pitches)

    while True:
        if discrete_pitch:
            pitch = random.choice(pitches)
        else:
            pitch_range = [pitches[0], pitches[-1]]
            pitch = random.uniform(*pitch_range)
        roll = random.choice([0, np.pi])

        grasp_quat = multiply_quats(
            quat,
            quat_from_euler(Euler(roll=np.pi / 2)),
            quat_from_euler(
                Euler(pitch=np.pi + pitch)
            ),  # TODO: local pitch or world pitch?
            quat_from_euler(Euler(roll=roll)),  # Switches fingers
        )
        grasp_pose = Pose(grasp_point, euler_from_quat(grasp_quat))
        grasp_pose = multiply(grasp_pose, Pose(Point(x=finger_length)))  # FINGER_LENGTH

        yield invert(
            grasp_pose
        ), []  # TODO: tool_from_grasp or grasp_from_tool convention?


##################################################


def tuplify_score(s):
    if isinstance(s, tuple):
        return s
    return (s,)


def negate_score(s):
    if isinstance(s, tuple):
        return s.__class__(map(negate_score, s))
    return -s


def combine_scores(score, *scores):
    combined_score = tuplify_score(score)
    for other_score in scores:
        combined_score = combined_score + tuplify_score(other_score)
    return combined_score


def score_width(point1, point2):
    return -get_distance(point1, point2)  # Priorities small widths


def score_antipodal(error1, error2):
    return -(error1 + error2)


def score_torque(mesh, tool_from_grasp, **kwargs):
    center_mass = mesh.center_mass
    x, _, z = tform_point(tool_from_grasp, center_mass)  # Distance in xz plane
    return -get_length([x, z])


def score_overlap(
    intersector,
    point1,
    point2,
    num_samples=15,
    radius=1.5e-2,
    draw=False,
    verbose=False,
    **kwargs,
):
    handles = []
    if draw:
        handles.append(add_line(point1, point2, color=RED))
    midpoint = np.average([point1, point2], axis=0)
    direction1 = point1 - point2
    direction2 = point2 - point1

    origins = []
    for _ in range(num_samples):
        # TODO: could return the set of surface points within a certain distance of the center
        # sample_sphere | sample_sphere_surface
        # from trimesh.sample import sample_surface_sphere
        other_direction = radius * sample_sphere_surface(
            d=3
        )  # TODO: sample rectangle for the PR2's fingers
        orthogonal_direction = np.cross(
            get_unit_vector(direction1), other_direction
        )  # TODO: deterministic
        orthogonal_direction = radius * get_unit_vector(orthogonal_direction)
        origin = midpoint + orthogonal_direction
        origins.append(origin)
        # print(get_distance(midpoint, origin))
        if draw:
            handles.append(add_line(midpoint, origin, color=RED))
    rays = list(range(len(origins)))

    direction_differences = []
    for direction in [direction1, direction2]:
        point = midpoint + direction / 2.0
        contact_distance = get_distance(midpoint, point)

        # section, slice_plane
        results = intersector.intersects_id(
            origins,
            len(origins) * [direction],  # face_indices, ray_indices, location
            return_locations=True,
            multiple_hits=True,
        )
        intersections_from_ray = {}
        for face, ray, location in zip(*results):
            intersections_from_ray.setdefault(ray, []).append((face, location))

        differences = []
        for ray in rays:
            if ray in intersections_from_ray:
                face, location = min(
                    intersections_from_ray[ray],
                    key=lambda pair: get_distance(point, pair[-1]),
                )
                distance = get_distance(origins[ray], location)
                difference = abs(contact_distance - distance)
                # normal = extract_normal(mesh, face) # TODO: use the normal for lexiographic scoring
            else:
                difference = np.nan  # np.inf
            differences.append(difference)
            # TODO: extract_normal(mesh, index) for the rays
        direction_differences.append(differences)

    differences1, differences2 = direction_differences
    combined = differences1 + differences2
    percent = np.count_nonzero(~np.isnan(combined)) / (len(combined))
    np.nanmean(combined)

    score = percent

    if verbose:
        print(
            "Score: {} | Percent1: {} | Average1: {:.3f} | Percent2: {} | Average2: {:.3f}".format(
                score,
                np.mean(~np.isnan(differences1)),
                np.nanmean(differences1),  # nanmedian
                np.mean(~np.isnan(differences2)),
                np.nanmean(differences2),
            )
        )  # 0.032 sec

    return score


##################################################


def generate_mesh_grasps(
    obj,
    max_width=np.inf,
    target_tolerance=np.pi / 4,
    antipodal_tolerance=0,
    z_threshold=-np.inf,
    max_attempts=np.inf,
    score_type="combined",
    **kwargs,
):
    target_vector = get_unit_vector(Z_AXIS)

    vertices, faces = mesh_from_obj(obj, **kwargs)
    # handles = draw_mesh(Mesh(vertices, faces))

    mesh = trimesh.Trimesh(vertices, faces)
    mesh.fix_normals()

    lower, upper = AABB(*mesh.bounds)
    surface_z = lower[2]
    min_z = surface_z + z_threshold
    intersector = RayMeshIntersector(mesh)

    attempts = last_attempts = 0

    while attempts < max_attempts:
        if last_attempts >= max_attempts:
            # break
            last_attempts = 0
            yield None
            continue
        attempts += 1
        last_attempts += 1

        [point1, point2], [index1, index2] = trimesh.sample.sample_surface(
            mesh=mesh,
            count=2,
            face_weight=None,  # seed=random.randint(1, 1e8)
        )

        if any(point[2] < min_z for point in [point1, point2]):
            continue
        distance = get_distance(point1, point2)
        if (distance > max_width) or (distance < 1e-3):
            continue
        direction2 = point2 - point1
        if abs(angle_between(target_vector, direction2) - np.pi / 2) > target_tolerance:
            continue

        normal1 = extract_normal(mesh, index1)
        if normal1.dot(-direction2) < 0:
            normal1 *= -1
        error1 = angle_between(normal1, -direction2)

        normal2 = extract_normal(mesh, index2)
        if normal2.dot(direction2) < 0:
            normal2 *= -1
        error2 = angle_between(normal2, direction2)

        if (error1 > antipodal_tolerance) or (error2 > antipodal_tolerance):
            continue

        # TODO: average the normals to select a new pair of contact points

        tool_from_grasp, handles = next(
            sample_grasp(obj, point1, point2, normal1, normal2, **kwargs)
        )

        assert score_type == "combined"

        score = combine_scores(
            score_overlap(intersector, point1, point2, **kwargs),
            score_torque(mesh, tool_from_grasp, **kwargs),
        )
        yield ScoredGrasp(tool_from_grasp, point1, point2, score)

        last_attempts = 0
        remove_handles(handles, **kwargs)


##################################################


def sorted_grasps(generator, max_candidates=10, p_random=0.0, **kwargs):
    candidates = []
    selected = []
    while True:
        for grasp in islice(generator, max_candidates - len(candidates)):
            if grasp is None:
                return
            else:
                index = len(selected) + len(candidates)
                heappush(candidates, (negate_score(grasp.score), index, grasp))
        if not candidates:
            break
        if p_random < random.random():
            score, index, grasp = candidates.pop(random.randint(0, len(candidates) - 1))
            heapify(candidates)
        else:
            score, index, grasp = heappop(candidates)

        yield grasp
        selected.append(grasp)


def filter_grasps(generator):
    raise NotImplementedError()
