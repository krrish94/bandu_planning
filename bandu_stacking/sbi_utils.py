from __future__ import print_function

"""Helper functions for simulation-based inference (SBI) using the sbi
toolbox."""

import numpy as np
import pybullet as p
import torch
from sbi import utils as sbi_utils
from sbi.inference.base import infer


def sample_physical_properties(num_samples):
    """Sample from a Gaussian prior distribution."""
    mass = np.random.normal(0.75, 0.1, num_samples)
    friction = np.random.normal(0.5, 0.1, num_samples)
    restitution = np.random.normal(0.3, 0.1, num_samples)
    return mass, friction, restitution


# Sample a prior pose (orientation) for the block.
def sample_prior_pose():
    """Sample a prior pose (orientation) for the block."""
    sampled_pose = p.getQuaternionFromEuler([0, 0, np.random.uniform(0, 2 * np.pi)])
    prior_pose = torch.tensor(sampled_pose)
    return prior_pose


# Simulate the forward model (use the pybullet client, adapt the physical properties and pose,
# run the simulation forward)
def simulate_forward_model(block_id, mass, friction, restitution, client, prior_pose):
    """Simulate the forward model."""
    pos, ori = client.getBasePositionAndOrientation(block_id)
    # Get the AABB of the block before simulation and read the height (Z-coordinate).
    aabb_before = p.getAABB(block_id)
    height_before = aabb_before[1][2]
    # Set the physical properties and pose for the block.
    client.changeDynamics(
        block_id,
        -1,
        mass=mass,
        lateralFriction=friction,
        restitution=restitution,
    )
    # Set the pose of block block_id to the prior pose. 0.1 is added to Z-coordinate to
    # simulate dropping the block from a height.
    pos_new = (pos[0], pos[1], pos[2] + 0.1)
    client.resetBasePositionAndOrientation(block_id, pos_new, prior_pose)
    # Run the simulation forward for a fixed number of steps.
    for _ in range(100):
        client.stepSimulation()
    # Get the AABB and height of the block after simulation.
    aabb_after = p.getAABB(block_id)
    height_after = aabb_after[1][2]
    height_change = height_after - height_before
    # If the height change is positive, the block is stable.
    is_stable = height_change > 0
    return height_change, is_stable


# Replicate the above function, but allow to use a source_block_id and a target_block_id
# (i.e., change states of two blocks) and run SBI like above.
def fit_sbi_model_pairwise(
    source_block_id,
    target_block_id,
    mass,
    friction,
    restitution,
    client,
    prior_pose,
    num_simulations=100,
):
    """Fit an SBI model (SNRE) for pairwise block stability."""
    # Define prior for mass, friction, restitution, and pose
    prior = sbi_utils.BoxUniform(
        low=torch.tensor([0.5, 0.1, 0.1, -np.pi]),
        high=torch.tensor([1.0, 0.8, 0.4, np.pi]),
    )

    def simulate_fn(params):
        pos, ori = client.getBasePositionAndOrientation(source_block_id)
        # _mass, _friction, _restitution, _yaw = params
        _mass = params[0].item()
        _friction = params[1].item()
        _restitution = params[2].item()
        _yaw = params[3].item()
        # Get the AABB of the block before simulation and read the height (Z-coordinate).
        aabb_before = p.getAABB(source_block_id)
        height_before = aabb_before[1][2]
        # Set the physical properties and pose for the block.
        client.changeDynamics(
            source_block_id,
            -1,
            mass=_mass,
            lateralFriction=_friction,
            restitution=_restitution,
        )
        # Set the pose of block block_id to the prior pose. 0.1 is added to Z-coordinate to
        # simulate dropping the block from a height.
        target_pos, target_ori = client.getBasePositionAndOrientation(target_block_id)
        target_aabb = client.getAABB(target_block_id)
        target_height = target_aabb[1][2]
        pos_new = (target_pos[0], target_pos[1], target_height + 0.1)
        ori_old = p.getEulerFromQuaternion(ori)
        ori_new = p.getQuaternionFromEuler([ori_old[0], ori_old[1], _yaw])
        client.resetBasePositionAndOrientation(source_block_id, pos_new, ori_new)
        # Run the simulation forward for a fixed number of steps.
        for _ in range(100):
            client.stepSimulation()
        # Get the AABB and height of the block after simulation.
        aabb_after = client.getAABB(source_block_id)
        height_after = aabb_after[1][2]
        height_change = height_after - height_before
        # If the height change is positive, the block is stable.
        is_stable = height_change > 0
        return height_change, is_stable

    posterior = infer(
        simulate_fn,
        prior,
        method="SNRE",
        num_simulations=num_simulations,
        # num_rounds=num_rounds,
    )

    # sampling_method = "mcmc"
    # mcmc_method = "slice_np"
    # posterior = inference.build_posterior(
    #     sample_with=sampling_method, mcmc_method=mcmc_method
    # )

    return posterior
