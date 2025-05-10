#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm


# Training parameters
M = 10  # number of steps
dataset_size = 200  # number of data points
d = 1 / M  # time step (sampling interval)
# Distribution parameters
VAR = 0.3  # variance
R = 1e8  # plotting range
device = torch.device('cpu')
# Lorentz force parameters
C_para = 1.5e8  # parallel force magnitude
C_perp = 1.5e8  # perpendicular force magnitude
c = 3e8  # speed of light
m = 1  # mass of particles


class DotDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'DotDict' object has no attribute '{name}'")
    def __setattr__(self, name, value):
        self[name] = value
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, state):
        self.__dict__.update(state)


def Create_Dataset(size):
    r_min, r_max = 1e1, 1e6  # Define the exclusion zone (r_min) and max radius (r_max)
    theta = torch.rand(size) * 2 * torch.pi  # Uniformly sample angles in [0, 2π]
    r = torch.rand(size) * (r_max - r_min) + r_min  # Uniform distribution between r_min and r_max
    # Clip radius to ensure all points are within [r_min, r_max]
    r = torch.clamp(r, r_min, r_max)
    # Convert polar coordinates to Cartesian (x, y)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    samples_0 = torch.stack([x, y], dim=1)  # Shape: (size, 2)
    # Initialize velocity as equal to position
    velocity = samples_0.clone()
    # Store trajectory data
    all_positions = [samples_0.clone()]
    all_velocities = [velocity.clone()]
    all_para_forces = []
    all_perp_forces = []
    all_accelerations = []
    # Simulation of motion with Lorentz force
    for _ in range(M - 1):
        # Compute Lorentz factor (gamma)
        speed_sq = (velocity ** 2).sum(dim=1, keepdim=True)
        gamma = 1 / torch.sqrt(1 - speed_sq / c**2)
        # Compute parallel force (same direction as velocity)
        velocity_norm = torch.norm(velocity, dim=1, keepdim=True)
        velocity_unit = velocity / (velocity_norm + 1e-8)  # Normalize to avoid division by zero
        F_para = C_para * velocity_unit  # Force parallel to velocity
        # Compute perpendicular force (random "up" or "down" direction)
        perp_directions = torch.ones((size, 1), dtype=torch.int)
        velocity_perp = torch.cat([-velocity[:, 1:2], velocity[:, 0:1]], dim=1)  # Rotate 90 degrees
        velocity_perp = velocity_perp / (velocity_norm + 1e-8)  # Normalize
        F_perp = C_perp * velocity_perp * perp_directions  # Random "up" or "down" force
        # Compute accelerations
        a_para = F_para / (m * gamma**3)
        a_perp = F_perp / (m * gamma)
        acceleration = a_para + a_perp
        # Update velocity and position
        velocity = velocity + acceleration * d
        position = all_positions[-1] + (velocity + all_velocities[-1]) * d / 2 
        # Store values
        all_positions.append(position.clone())
        all_velocities.append(velocity.clone())
        all_accelerations.append(acceleration.clone())
    # Compute final acceleration step
    speed_sq = (velocity ** 2).sum(dim=1, keepdim=True)
    gamma = 1 / torch.sqrt(1 - speed_sq / c**2)
    velocity_norm = torch.norm(velocity, dim=1, keepdim=True)
    velocity_unit = velocity / (velocity_norm + 1e-8)
    F_para = C_para * velocity_unit
    perp_directions = torch.ones((size, 1), dtype=torch.int)
    velocity_perp = torch.cat([-velocity[:, 1:2], velocity[:, 0:1]], dim=1)
    velocity_perp = velocity_perp / (velocity_norm + 1e-8)
    F_perp = C_perp * velocity_perp * perp_directions
    a_para = F_para / (m * gamma**3)
    a_perp = F_perp / (m * gamma)
    acceleration = a_para + a_perp
    all_accelerations.append(acceleration.clone())
    # Convert lists to tensors
    all_positions = torch.stack(all_positions, dim=1)  # Shape: (size, M+1, 2)
    all_velocities = torch.stack(all_velocities, dim=1)  # Shape: (size, M+1, 2)
    all_accelerations = torch.stack(all_accelerations, dim=1)  # Shape: (size, M, 2)
    samples_1 = all_positions[:, -1]  # Final positions after M steps
    all_para_forces = torch.full((size, M, 1), C_para)
    all_perp_forces = torch.full((size, M, 1), C_perp)
    # Create dataset with DotDict
    dataset = DotDict({
        "p0": samples_0,
        "pm": all_positions,
        "v": all_velocities,
        "a": all_accelerations,
        "pf": samples_1,
        "para": all_para_forces,
        "perp": all_perp_forces
    })
    return dataset

def Save_Dataset(dataset):
    # Plotting
    plt.figure(figsize=(4.3, 4))
    plt.xlim(-R / 1e8, R / 1e8)
    plt.ylim(-R / 1e8, R / 1e8)
    plt.title(r'Samples from $\pi_0$ and $\pi_1$', fontsize=19)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # Plot the initial and final samples
    plt.scatter(dataset.p0[:, 0].cpu().numpy() / 1e8, dataset.p0[:, 1].cpu().numpy() / 1e8, alpha=1.0, c='#EE7959', label=r'$\pi_0$')
    plt.scatter(dataset.pf[:, 0].cpu().numpy() / 1e8, dataset.pf[:, 1].cpu().numpy() / 1e8, alpha=1.0, c='#D9A0B3', label=r'$\pi_1$')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'size': 12})
    plt.tight_layout()

def Plot_Trajectories(dataset):
    # Determine the number of trajectories to plot (dataset_size / 10)
    total_samples = dataset.pm.shape[0]
    num_trajectories = max(1, int(total_samples / 10))
    # Randomly select indices
    indices = torch.randperm(total_samples)[:num_trajectories]
    # Create figure with similar style as before
    plt.figure(figsize=(4.3, 4))
    # Use the same axis limits (scaled as in Save_Dataset)
    R = 1e8  # plotting range constant as defined above
    plt.xlim(-R / 1e8, R / 1e8)
    plt.ylim(-R / 1e8, R / 1e8)
    plt.title("Sample Trajectories", fontsize=19)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # Prepare a color cycle to distinguish trajectories
    colors = plt.cm.jet(np.linspace(0, 1, num_trajectories))
    # Plot each selected trajectory
    for i, idx in enumerate(indices):
        # Each trajectory is of shape (M+1, 2)
        traj = dataset.pm[idx].cpu().numpy()
        # Plot the trajectory as a line with markers at each time step.
        plt.plot(traj[:, 0] / 1e8, traj[:, 1] / 1e8, marker='o',
                 markersize=3, color=colors[i], label=f'Traj {idx.item()}')
    # Optionally, add a legend (if not too crowded)
    if num_trajectories <= 10:
        plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('tmp_figures/trajectory_circle_constant_constant.pdf', format='pdf', bbox_inches='tight')


# In[2]:


dataset = Create_Dataset(dataset_size)
Plot_Trajectories(dataset)

