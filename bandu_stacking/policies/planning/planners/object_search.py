from __future__ import print_function

from policies.planning.planners.main import CLASS_THRESH, Planner

from pybullet_planning.pybullet_tools.utils import (
    OOBB,
    Pose,
    create_box,
    get_aabb,
    get_aabb_extent,
    get_pose,
    set_pose,
    stable_z_on_aabb,
)


def reset_robot(robot):
    conf = robot.get_default_conf()
    for group, positions in conf.items():
        robot.set_group_positions(group, positions)


class ObjectSearch(Planner):
    def __init__(self):
        pass

    def plan(self, robot, belief, goal):
        if "unseen" not in belief.memory:
            belief.memory["unseen"] = []

        local_robot, table, fence, client = self.world_twin(robot, belief)

        selected_candidate = None
        selected_obj = None

        most_obstructing = None
        obstructing_value = -float("inf")
        most_obstructing_obj = None

        environment = [table, fence]

        for unseen_oobb in belief.memory["unseen"]:
            extents = get_aabb_extent(unseen_oobb.aabb)
            box = create_box(extents[0], extents[1], extents[2], client=client)
            set_pose(box, unseen_oobb.pose, client=client)
            environment.append(box)

        for segment in belief.segments:
            aabb = segment.aabb
            pose = Pose(point=segment.origin)
            max_class_prob = 0
            is_model = False
            for candidate in segment.candidates:
                if candidate.probability > max_class_prob:
                    is_model = True
                    max_class_prob = candidate.probability
                    aabb = candidate.oobb.aabb
                    pose = candidate.oobb.pose

            extents = get_aabb_extent(aabb)
            box = create_box(extents[0], extents[1], extents[2], client=client)
            set_pose(box, pose=pose, client=client)

            # reset pose to be stable on surface if ycb
            if is_model:
                new_z = (
                    stable_z_on_aabb(box, get_aabb(table, client=client), client=client)
                    + 0.01
                )
                new_pose = ((pose[0][0], pose[0][1], new_z), pose[1])
                set_pose(box, pose=new_pose, client=client)

            environment.append(box)

            if segment.obstructed_volume > obstructing_value:
                most_obstructing = segment  # class and pose are not important
                obstructing_value = segment.obstructed_volume
                most_obstructing_obj = box

            for candidate in segment.candidates:
                if candidate.category == goal and candidate.probability > CLASS_THRESH:
                    selected_candidate = candidate
                    selected_obj = box
            if selected_candidate is not None:
                break

        if selected_candidate is None:
            # Remove obstructing objects
            extents = get_aabb_extent(most_obstructing.aabb)
            # z = stable_z_on_aabb(most_obstructing_obj, get_aabb(table, client=client), client=client)
            # placement_location = Pose(point=Point(x=0.3, y=0.4, z=z), euler=Euler(yaw=math.pi/2.0))
            print("Environment: " + str(environment))

            placement_pose, traj = self.get_pick_place_home_plan(
                local_robot,
                most_obstructing_obj,
                most_obstructing.aabb,
                get_pose(most_obstructing_obj, client=client),
                surface_aabb=belief.surface.aabb,
                environment=environment,
                client=client,
            )

            belief.memory["unseen"].append(OOBB(most_obstructing.aabb, placement_pose))

            return traj, belief
        else:
            # Pick the candidate
            return (
                self.get_pick_home_plan(
                    local_robot,
                    selected_obj,
                    selected_candidate.oobb.aabb,
                    selected_candidate.oobb.pose,
                    environment=environment,
                    client=client,
                ),
                belief,
            )
