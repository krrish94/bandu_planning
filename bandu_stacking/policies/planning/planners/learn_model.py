from __future__ import print_function

from policies.planning.planners.main import CLASS_THRESH, Planner

from pybullet_planning.pybullet_tools.utils import (
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


class LearnModel(Planner):
    def __init__(self):
        pass

    def plan(self, robot, belief, goal):
        local_robot, table, fence, client = self.world_twin(robot, belief)

        environment = [table, fence]

        segment = belief.segments[0]
        bin_id = None
        selected_candidate = None
        for segment in belief.segments:
            for candidate in segment.candidates:
                if candidate.probability > CLASS_THRESH:
                    selected_candidate = candidate
                    if (
                        "previously_recognized" not in belief.memory
                        or selected_candidate.category
                        not in belief.memory["previously_recognized"]
                    ):
                        if "previously_recognized" not in belief.memory:
                            belief.memory["previously_recognized"] = []
                        bin_id = len(belief.memory["previously_recognized"])
                        belief.memory["previously_recognized"] = list(
                            belief.memory["previously_recognized"]
                            + [selected_candidate.category]
                        )
                    else:
                        bin_id = belief.memory["previously_recognized"].index(
                            selected_candidate.category
                        )

            if selected_candidate is not None:
                break

        if selected_candidate is None:
            return self.get_inspect_plan(local_robot, client=client), belief
        else:
            extents = get_aabb_extent(selected_candidate.oobb.aabb)
            box = create_box(extents[0], extents[1], extents[2], client=client)
            set_pose(box, candidate.oobb.pose, client=client)
            # Temporary hack to get bin pose from simulation. Will need to be different for real world
            bin_pose = get_pose(
                belief.bins[bin_id], client=robot.client
            )  # TODO: Move to local client
            bin_aabb = get_aabb(
                belief.bins[bin_id], client=robot.client
            )  # TODO: Move to local client
            z = stable_z_on_aabb(box, bin_aabb, client=client)
            placement_pose = Pose([bin_pose[0][0], bin_pose[0][1], z])
            placement_pose, seq = self.get_pick_place_home_plan(
                local_robot,
                box,
                candidate.oobb.aabb,
                candidate.oobb.pose,
                placement_pose=placement_pose,
                client=client,
                environment=environment,
            )
            return seq, belief
