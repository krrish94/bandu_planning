from __future__ import print_function

import os
import random
import time
from collections import namedtuple
from heapq import heapify, heappop, heappush
from itertools import islice

import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc

from pybullet_planning.pybullet_tools.utils import (
    AABB,
    INF,
    PI,
    RED,
    UNKNOWN_FILE,
    Euler,
    Mesh,
    Point,
    Pose,
    PoseSaver,
    add_line,
    angle_between,
    apply_affine,
    convex_combination,
    draw_point,
    draw_pose,
    elapsed_time,
    euler_from_quat,
    flatten,
    get_aabb,
    get_aabb_center,
    get_aabb_vertices,
    get_data_filename,
    get_data_pose,
    get_data_scale,
    get_distance,
    get_length,
    get_model_info,
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
    unit_pose,
    wait_if_gui,
)

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


X_AXIS = np.array([1, 0, 0])  # TODO: make immutable
Z_AXIS = np.array([0, 0, 1])


def sample_sphere_surface(d, uniform=True):
    # TODO: hyperspherical coordinates
    # https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    while True:
        v = np.random.randn(d)
        r = np.sqrt(v.dot(v))
        if not uniform or (r <= 1.0):
            return v / r


PREGRASP_DISTANCE = 0.07  # 0.05 | 0.07
FINGER_LENGTH = 0.0

ScoredGrasp = namedtuple("ScoredGrasp", ["pose", "contact1", "contact2", "score"])

##################################################


def control_until_contact(
    controller, body, contact_links=[], time_after_contact=INF, all_contacts=True
):
    # TODO: unify with close_gripper
    # TODO: list different grasping control strategies
    agg = all if all_contacts else any
    if (time_after_contact is INF) or not contact_links:
        for output in controller:
            yield output
        return
    dt = get_time_step()
    countdown = INF
    for output in controller:  # TODO: force control for grasping
        if countdown <= 0.0:
            break

        if (countdown == INF) and agg(
            single_collision(body, link=link) for link in contact_links
        ):  # any | all
            # (time_after_contact != INF) and contact_links
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
    # TODO: move to experiment.py
    obj_pose = get_pose(obj)
    for grasp_tool in grasp_generator:
        if grasp_tool is None:
            continue
        grasp_pose = multiply(
            obj_pose, invert(get_grasp(grasp_tool, gripper_from_tool))
        )
        # pregrasp_pose = multiply(obj_pose, invert(get_pregrasp(grasp_tool, gripper_from_tool, **kwargs)))
        with PoseSaver(gripper):
            set_pose(gripper, grasp_pose)  # grasp_pose | pregrasp_pose
            if any(pairwise_collision(gripper, obst) for obst in [obj] + obstacles):
                continue
            if draw:
                # print([pairwise_collision(gripper, obst) for obst in [obj] + obstacles])
                handles = draw_pose(grasp_pose)
                wait_if_gui()
                remove_handles(handles)
                # continue
            # TODO: check the ground plane (bounding box of the gripper)
            # TODO: check the pregrasp pose
            # conf = close_until_collision(gripper, gripper_joints, bodies=[obj],
            #                              open_conf=open_conf, closed_conf=closed_conf)
            yield grasp_tool


##################################################


def get_mesh_path(obj):
    info = get_model_info(obj)
    if info is None:
        [data] = get_visual_data(obj)
        path = get_data_filename(data)
    else:
        path = info.path

    return path


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


def extract_normal(mesh, index):
    return np.array(mesh.face_normals[index, :])


def sample_grasp(
    obj,
    point1,
    point2,
    normal1,
    normal2,
    pitches=[-PI, PI],
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
        roll = random.choice([0, PI])

        grasp_quat = multiply_quats(
            quat,
            quat_from_euler(Euler(roll=PI / 2)),
            quat_from_euler(
                Euler(pitch=PI + pitch)
            ),  # TODO: local pitch or world pitch?
            quat_from_euler(Euler(roll=roll)),  # Switches fingers
        )
        grasp_pose = Pose(grasp_point, euler_from_quat(grasp_quat))
        grasp_pose = multiply(grasp_pose, Pose(Point(x=finger_length)))  # FINGER_LENGTH

        handles = []
        if draw:
            # set_pose(gripper, multiply(grasp_pose, tool_from_root))
            draw_length = 0.05
            handles.extend(
                flatten(
                    [
                        # draw_point(grasp_point),
                        draw_point(point1, parent=obj),
                        draw_point(point2, parent=obj),
                        draw_pose(grasp_pose, length=draw_length, parent=obj),
                        [
                            add_line(point1, point2, parent=obj),
                            add_line(
                                point1,
                                point1 + draw_length * normal1,
                                parent=obj,
                                color=RED,
                            ),
                            add_line(
                                point2,
                                point2 + draw_length * normal2,
                                parent=obj,
                                color=RED,
                            ),
                        ],
                    ]
                )
            )
            # wait_if_gui() # TODO: wait_if_unlocked
            # remove_handles(handles)
        yield invert(
            grasp_pose
        ), handles  # TODO: tool_from_grasp or grasp_from_tool convention?


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
    # TODO: could also do with PyBullet using batch_ray_collision
    # TODO: use trimesh to get the polygon intersection of the object and fingers
    # TODO: could test whether points on the circle are inside the object
    # import trimesh
    # intersector.contains_points
    # intersector.intersects_any
    # intersector.intersects_first
    # intersector.intersects_id
    # intersector.intersects_location
    # intersector.intersects_id
    # trimesh.ray.ray_triangle.ray_bounds
    # trimesh.ray.ray_triangle.ray_triangle_candidates
    # trimesh.ray.ray_triangle.ray_triangle_id

    start_time = time.time()
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
                difference = np.nan  # INF
            differences.append(difference)
            # TODO: extract_normal(mesh, index) for the rays
        direction_differences.append(differences)

    differences1, differences2 = direction_differences
    combined = differences1 + differences2
    percent = np.count_nonzero(~np.isnan(combined)) / (len(combined))
    np.nanmean(combined)

    # return np.array([percent, -average])
    score = percent
    # score = 1e3*percent + (-average) # TODO: true lexiographic sorting
    # score = (percent, -average)

    if verbose:
        print(
            "Score: {} | Percent1: {} | Average1: {:.3f} | Percent2: {} | Average2: {:.3f} | Time: {:.3f}".format(
                score,
                np.mean(~np.isnan(differences1)),
                np.nanmean(differences1),  # nanmedian
                np.mean(~np.isnan(differences2)),
                np.nanmean(differences2),
                elapsed_time(start_time),
            )
        )  # 0.032 sec
    if draw:
        wait_if_gui()
        remove_handles(handles, **kwargs)
    return score


##################################################


def generate_mesh_grasps(
    obj,
    max_width=INF,
    target_tolerance=PI / 4,
    antipodal_tolerance=PI / 6,
    z_threshold=-INF,
    max_time=INF,
    max_attempts=INF,
    score_type="combined",
    verbose=False,
    **kwargs,
):
    # TODO: sample xy lines
    # TODO: compute pairs of faces that could work
    # TODO: some amount of basic collision checking here
    target_vector = get_unit_vector(Z_AXIS)

    import trimesh

    vertices, faces = mesh_from_obj(obj, **kwargs)
    # handles = draw_mesh(Mesh(vertices, faces))
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.fix_normals()

    lower, upper = AABB(*mesh.bounds)
    surface_z = lower[2]
    min_z = surface_z + z_threshold

    from trimesh.ray.ray_triangle import RayMeshIntersector

    intersector = RayMeshIntersector(mesh)

    last_time = time.time()
    attempts = last_attempts = 0
    while attempts < max_attempts:
        if (elapsed_time(last_time) >= max_time) or (last_attempts >= max_attempts):
            # break
            last_time = time.time()
            last_attempts = 0
            yield None
            continue
        attempts += 1
        last_attempts += 1

        [point1, point2], [index1, index2] = mesh.sample(2, return_index=True)
        if any(point[2] < min_z for point in [point1, point2]):
            continue
        distance = get_distance(point1, point2)
        if (distance > max_width) or (distance < 1e-3):
            continue
        direction2 = point2 - point1
        if abs(angle_between(target_vector, direction2) - PI / 2) > target_tolerance:
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
        tool_axis = tform_point(invert(tool_from_grasp), X_AXIS)
        if tool_axis[2] > 0:  # TODO: angle from plane
            # handles.append(add_line(tform_point(invert(tool_from_grasp), unit_point()), tool_axis, parent=obj))
            # wait_if_gui()
            # remove_handles(handles)
            continue

        if score_type is None:
            score = 0.0
        elif score_type == "torque":
            score = score_torque(mesh, tool_from_grasp, **kwargs)
        elif score_type == "antipodal":
            score = score_antipodal(error1, error2, **kwargs)
        elif score_type == "width":
            score = score_width(point1, point2, **kwargs)
        elif score_type == "overlap":
            score = score_overlap(intersector, point1, point2, **kwargs)
        elif score_type == "combined":
            score = combine_scores(
                score_overlap(intersector, point1, point2, **kwargs),
                score_torque(mesh, tool_from_grasp, **kwargs),
                # score_antipodal(error1, error2),
            )
        else:
            raise NotImplementedError(score_type)

        if verbose:
            print(
                "Runtime: {:.3f} sec | Attempts: {} | Score: {}".format(
                    elapsed_time(last_time), last_attempts, score
                )
            )
        yield ScoredGrasp(
            tool_from_grasp, point1, point2, score
        )  # Could just return orientation and contact points

        last_time = time.time()
        last_attempts = 0
        # wait_if_gui()
        remove_handles(handles, **kwargs)


##################################################


def sorted_grasps(generator, max_candidates=10, p_random=0.0, **kwargs):
    # TODO: pose distance metric and diverse planning
    # TODO: prune grasps that are too close to each other
    # TODO: is there an existing python method that does this?
    # TODO: sort after checking for collisions
    candidates = []
    selected = []
    while True:
        start_time = time.time()
        for grasp in islice(generator, max_candidates - len(candidates)):
            if grasp is None:
                return
            else:
                index = len(selected) + len(candidates)
                heappush(candidates, (negate_score(grasp.score), index, grasp))
        if not candidates:
            break
        if p_random < random.random():
            # TODO: could also make a high score or immediately return
            # TODO: geometric distribution for how long an element will be in the queue
            # TODO: ordinal statistics for sampling
            score, index, grasp = candidates.pop(random.randint(0, len(candidates) - 1))
            heapify(candidates)
        else:
            score, index, grasp = heappop(candidates)
        print(
            "Grasp: {} | Index: {} | Candidates: {} | Score: {} | Time: {:.3f}".format(
                len(selected),
                index,
                len(candidates) + 1,
                negate_score(score),
                elapsed_time(start_time),
            )
        )
        yield grasp
        selected.append(grasp)


if __name__ == "__main__":
    client = bc.BulletClient(connection_mode=p.DIRECT)
    bounding_boxes = []
    block_ids = []
    bandu_model_path = "./models/bandu_models"
    bandu_filenames = [
        f
        for f in os.listdir(bandu_model_path)
        if os.path.isfile(os.path.join(bandu_model_path, f))
    ]
    bandu_urdfs = [f for f in bandu_filenames if f.endswith("urdf")]
    bandu_urdf = os.path.join(bandu_model_path, random.choice(bandu_urdfs))
    obj = client.loadURDF(bandu_urdf, globalScaling=0.002)
    mesh_grasps = generate_mesh_grasps(obj, max_attempts=10000, client=client)
    print(f"Grasps: {len(list(mesh_grasps))}")
