#!/usr/bin/env python
# coding: utf-8

# # Create Dataset Function

# In[1]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from tqdm import tqdm

# Training parameters
M = 10  # number of steps
dataset_size = 300  # number of data points
d = 1 / M  # time step (sampling interval)
# Distribution parameters
VAR = 0.3  # variance
R = 9e7  # plotting range
COMP = 1000
device = torch.device('cpu')
# Lorentz force parameters
C_para = 1.5e7  # parallel force magnitude
C_perp = 1.5e8  # perpendicular force magnitude
c = 3e8  # speed of light
m = 1  # mass of particles
C_spiral = 2e4

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
    def spiral_function(theta):
        return ((theta * 5) + 10) * C_spiral
    num_turns = 2
    angles = [k * (2 * np.pi / COMP) for k in range(COMP * num_turns)]
    radii = [spiral_function(theta) for theta in angles]
    vertices_1 = [[r * np.cos(theta), r * np.sin(theta)] for r, theta in zip(radii, angles)]
    target_mix = Categorical(torch.tensor([1 / len(vertices_1)] * len(vertices_1)))
    target_comp = MultivariateNormal(
        torch.tensor(vertices_1).float(),
        VAR * torch.stack([torch.eye(2) for _ in range(len(vertices_1))])
    )
    target_model = MixtureSameFamily(target_mix, target_comp)
    samples_0 = target_model.sample([size])  # Sample from the target distribution

    # Initialize velocity as equal to position
    velocity = samples_0.clone() * 6e1
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


# # Save Dataset

# In[2]:


def Save_Dataset(dataset, test_dataset):
    # Save dataset
    # torch.save(dataset, "dataset/train_spiral.pth")
    # torch.save(test_dataset, "dataset/test_spiral.pth")
    # Plotting
    plt.figure(figsize=(4.1, 4))
    # plt.xlim(-R / 3e9, R / 3e9)
    # plt.ylim(-R / 3e9, R / 3e9)
    plt.xlim(-R / 3e8, R / 3e8)
    plt.ylim(-R / 3e8, R / 3e8)
    plt.title(r'Spiral Dataset', fontsize=19)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # Plot the initial and final samples
    plt.scatter(test_dataset.pf[:, 0].cpu().numpy() / 3e8, test_dataset.pf[:, 1].cpu().numpy() / 3e8, alpha=1, c='#D9A0B3', label=r'$\pi_1$')
    plt.scatter(test_dataset.p0[:, 0].cpu().numpy() / 3e8, test_dataset.p0[:, 1].cpu().numpy() / 3e8, alpha=1, c='#2E59A7', label=r'$\pi_0$')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'size': 12})
    plt.tight_layout()
    # plt.savefig('figures/dataset_spiral.pdf', format='pdf', bbox_inches='tight')

def Plot_Trajectories(pm):
    # Determine the number of trajectories to plot (dataset_size / 10)
    total_samples = pm.shape[0]
    num_trajectories = max(1, int(total_samples / 20))
    # Randomly select indices
    indices = torch.randperm(total_samples)[:num_trajectories]
    # Create figure with similar style as before
    plt.figure(figsize=(4.1, 4))
    # Use the same axis limits (scaled as in Save_Dataset)
    plt.xlim(-R / 3e8, R / 3e8)
    plt.ylim(-R / 3e8, R / 3e8)
    plt.title("Trajectories generated\nby (O1 + O2) model", fontsize=19)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # Prepare a color cycle to distinguish trajectories
    colors = plt.cm.jet(np.linspace(0, 1, num_trajectories))
    # Plot each selected trajectory
    for i, idx in enumerate(indices):
        # Each trajectory is of shape (M+1, 2)
        traj = pm[idx].cpu().numpy()
        # Plot the trajectory as a line with markers at each time step.
        plt.plot(traj[:, 0] / 3e8, traj[:, 1] / 3e8, marker='o',
                 markersize=3, color=colors[i], label=f'Traj {idx.item()}')
    # Optionally, add a legend (if not too crowded)
    if num_trajectories <= 10:
        plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('figures/trajectory_12_spiral.pdf', format='pdf', bbox_inches='tight')

@torch.no_grad()
def draw_plot(rectified_flow, test_dataset, N=None):
  traj, _ = rectified_flow.sample_ode(test_dataset, N=N)
  plt.figure(figsize=(4,4.2))
  plt.xlim(-R / 3e8,R / 3e8)
  plt.ylim(-R / 3e8,R / 3e8)
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)
  plt.scatter(test_dataset.pf[:, 0].cpu().numpy() / 3e8, test_dataset.pf[:, 1].cpu().numpy() / 3e8, c='#D9A0B3', label=r'$\pi_1$', alpha=0.5)
  plt.scatter(traj[0][:, 0].cpu().numpy() / 3e8, traj[0][:, 1].cpu().numpy() / 3e8, c='#2E59A7' , label=r'$\pi_0$', alpha=1.0)
  plt.scatter(traj[-1][:, 0].cpu().numpy() / 3e8, traj[-1][:, 1].cpu().numpy() / 3e8, c='#9966CC' , label='Generated', alpha=1.0)
  plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'size': 9})
  plt.title('Spiral optimized\nwith (O1 + O2) losses', fontsize=19)
  plt.tight_layout()
  plt.savefig('figures/12_spiral.pdf', format='pdf', bbox_inches='tight')


# # Models

# In[3]:


class First_MLP(nn.Module):
    def __init__(self, input_dim = 2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)
    def forward(self, input_dim, t):
        inputs = torch.cat([input_dim, t], dim=1)
        x = self.fc1(inputs)
        # x = self.act(x)
        x = self.fc2(x)
        # x = self.act(x)
        x = self.fc3(x)
        return x
class Second_MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + input_dim + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)
    def forward(self, first_order_input, x_input, t):
        inputs = torch.cat([first_order_input, x_input, t], dim=1)
        x = self.fc1(inputs)
        # x = self.act(x)
        x = self.fc2(x)
        # x = self.act(x)
        x = self.fc3(x)
        return x

class RectifiedFlow2():
  def __init__(self, model = None, num_steps=1000):
    self.model = model
    self.first_order_model = self.model[0]
    self.second_order_model = self.model[1]
    self.N = num_steps
  def get_train_tuple(self, dataset): # Need to rewrite
    t = torch.randint(0, self.N, (dataset.p0.shape[0], 1))
    z_t = dataset.pm[torch.arange(dataset_size), t.squeeze(), :]
    First_target = dataset.v[torch.arange(dataset_size), t.squeeze(), :]
    Second_target = dataset.a[torch.arange(dataset_size), t.squeeze(), :]
    return z_t, t, First_target, Second_target
  def frist_and_second_order_predict(self, z_t, t):
    first_order_pred = self.first_order_model(z_t, t)
    second_order_pred = self.second_order_model(first_order_pred, z_t, t)
    return first_order_pred, second_order_pred
  @torch.no_grad()
  def sample_ode(self, dataset, N=None):
    if N is None:
      N = self.N
    dt = 1./N
    traj = [] # to store the trajectory
    z = dataset.p0.detach().clone()
    batchsize = z.shape[0]
    traj.append(z.detach().clone())
    for i in range(N):
      t = torch.ones((batchsize,1)) * i / N
      first_order_pred, second_order_pred = self.frist_and_second_order_predict(z, t)
      z = z.detach().clone() + first_order_pred * dt + 0.5 * second_order_pred * dt**2
      traj.append(z.detach().clone())
    first_order_loss = torch.sqrt((dataset.pf - z).abs().pow(2).sum(dim=1))
    first_order_loss_mean = first_order_loss.mean()
    print("Average L2 Norm:", first_order_loss_mean.item())
    traj_tensor = torch.stack(traj, dim=1)
    return traj, traj_tensor


# # Training Functions

# In[4]:


def train_rectified_flow(rectified_flow, optimizer, dataset, inner_iters):
  loss_curve = []
  for _ in tqdm(range(inner_iters+1)):
    optimizer.zero_grad()
    z_t, t, first_order_gt, second_order_gt = rectified_flow.get_train_tuple(dataset)
    first_order_pred, second_order_pred = rectified_flow.frist_and_second_order_predict(z_t, t / rectified_flow.N)
    first_order_loss = (first_order_gt - first_order_pred).abs().pow(2).sum(dim=1)
    second_order_loss = (second_order_gt - second_order_pred).abs().pow(2).sum(dim=1)
    first_order_loss_mean = first_order_loss.mean()
    second_order_loss_mean = second_order_loss.mean()
    loss = first_order_loss_mean + second_order_loss_mean
    loss.backward()
    optimizer.step()
    loss_curve.append(np.log(loss.item())) ## to store the loss curve
  return rectified_flow, loss_curve


# # Create and Save Dataset

# In[5]:


dataset = Create_Dataset(dataset_size)
test_dataset = Create_Dataset(int(dataset_size * 3))
Save_Dataset(dataset, test_dataset)


# # Training 

# In[6]:


input_dim = 2
reflow_iterations = 4000
model1 = First_MLP(input_dim, hidden_num=100).to(device)
model2 = Second_MLP(input_dim, hidden_num=100).to(device)
modellist = nn.ModuleList([model1, model2]).to(device)
rectified_flow_2 = RectifiedFlow2(modellist, num_steps=M)
optimizer = torch.optim.Adam(rectified_flow_2.model.parameters(), lr=5e-3)
rectified_flow_2, loss_curve = train_rectified_flow(rectified_flow_2, optimizer, dataset, reflow_iterations)


# # Result

# In[7]:


draw_plot(rectified_flow_2, test_dataset)
Plot_Trajectories(rectified_flow_2.sample_ode(test_dataset)[1])
# Plot_Trajectories(test_dataset.pm)

