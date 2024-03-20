class State:
    def __init__(self, block_ids, block_poses):
        self.block_ids = block_ids
        self.block_poses = block_poses

    def __repr__(self):
        return f"State(block_ids={self.block_ids}, block_poses={self.block_poses})"


class Action:
    def __init__(self, grasp_block, target_block, pose):
        self.grasp_block = grasp_block
        self.target_block = target_block
        self.pose = pose

    def __repr__(self):
        return f"Action(grasp_block={self.grasp_block}, target_block={self.target_block}, pose={self.pose})"


class Policy:
    def __init__(self):
        pass

    def get_action(self, initial_state):
        raise NotImplementedError