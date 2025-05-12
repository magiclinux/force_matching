#!/usr/bin/env python
# coding: utf-8

# # Create Dataset Function

# In[15]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm

# Training parameters
M = 10  # number of steps
dataset_size = 500  # number of data points
d = 1 / M  # time step (sampling interval)
# Distribution parameters
VAR = 0.3  # variance
R = 3e8  # plotting range
device = torch.device('cpu')
# Lorentz force parameters
C_para = 1e7  # parallel force magnitude
C_perp = 7e8  # perpendicular force magnitude
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
    # Define the rectangle dimensions
    width = 1e1    # short axis (x-direction)
    height = 2e6   # long axis (y-direction)
    # Generate uniformly distributed samples for x and y.
    x = (torch.rand(size) - 0.5) * width
    y = (torch.rand(size) - 0.5) * height
    # Combine into a tensor of shape (size, 2)
    samples_0 = torch.stack([x, y], dim=1)
    # Define the velocity field
    V = 3e8
    direction = torch.zeros_like(samples_0)
    direction[:, 0] = torch.where(samples_0[:, 1] > 0, torch.tensor(1.0, device=samples_0.device), torch.tensor(-1.0, device=samples_0.device))
    length = V * torch.abs(torch.sin(3.1415 * samples_0[:, 1:2] / 1e6))
    velocity = length * direction
    # Store trajectory data
    all_positions = [samples_0.clone()]
    all_velocities = [velocity.clone()]
    all_para_forces = []
    all_perp_forces = []
    all_accelerations = []
    # Simulation of motion with Lorentz force
    for t in range(M - 1):
        # Compute Lorentz factor (gamma)
        speed_sq = (velocity ** 2).sum(dim=1, keepdim=True)
        gamma = 1 / torch.sqrt(1 - speed_sq / c**2)
        # Compute parallel force (same direction as velocity)
        velocity_norm = torch.norm(velocity, dim=1, keepdim=True)
        velocity_unit = velocity / (velocity_norm + 1e-8)  # Normalize to avoid division by zero
        ### C_para = C_para * torch.abs(torch.sin(3.1415 * samples_0[:, 1:2] / 1e6))
        F_para = C_para * velocity_unit * torch.abs(torch.sin(torch.full((samples_0.shape[0], 1), 3.1415 * t / 10)))
        tmpc_para = C_para * torch.abs(torch.sin(torch.full((samples_0.shape[0], 1), 3.1415 * t / 10)))
        # Compute perpendicular force (random "up" or "down" direction)
        perp_directions = torch.ones((size, 1), dtype=torch.int)
        velocity_perp = torch.cat([-velocity[:, 1:2], velocity[:, 0:1]], dim=1)  # Rotate 90 degrees
        velocity_perp = velocity_perp / (velocity_norm + 1e-8)  # Normalize
        ### C_perp = C_perp * torch.abs(torch.sin(3.1415 * samples_0[:, 1:2] / 1.25e5))
        F_perp = C_perp * velocity_perp * perp_directions * torch.abs(torch.sin(torch.full((samples_0.shape[0], 1), 3.1415 * 8 * t / 10)))
        tmpc_perp = C_perp * torch.abs(torch.sin(torch.full((samples_0.shape[0], 1), 3.1415 * 8 * t / 10)))
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
        all_para_forces.append(tmpc_para.clone())
        all_perp_forces.append(tmpc_perp.clone())
        all_accelerations.append(acceleration.clone())
    # Compute final acceleration step
    speed_sq = (velocity ** 2).sum(dim=1, keepdim=True)
    gamma = 1 / torch.sqrt(1 - speed_sq / c**2)
    velocity_norm = torch.norm(velocity, dim=1, keepdim=True)
    velocity_unit = velocity / (velocity_norm + 1e-8)
    # C_para = C_para * torch.abs(torch.sin(3.1415 * samples_0[:, 1:2] / 1e6))
    F_para = C_para * velocity_unit * torch.abs(torch.sin(torch.full((samples_0.shape[0], 1), 3.1415 * t * 9 / 10)))
    tmpc_para = C_para * torch.abs(torch.sin(torch.full((samples_0.shape[0], 1), 3.1415 * 9 / 10)))
    perp_directions = torch.ones((size, 1), dtype=torch.int)
    velocity_perp = torch.cat([-velocity[:, 1:2], velocity[:, 0:1]], dim=1)
    velocity_perp = velocity_perp / (velocity_norm + 1e-8)
    # C_perp = C_perp * torch.abs(torch.sin(3.1415 * samples_0[:, 1:2] / 1.25e5))
    F_perp = C_perp * velocity_perp * perp_directions * torch.abs(torch.sin(torch.full((samples_0.shape[0], 1), 3.1415 * 8 * 9 / 10)))
    tmpc_perp = C_perp * torch.abs(torch.sin(torch.full((samples_0.shape[0], 1), 3.1415 * 8 * 9 / 10)))
    a_para = F_para / (m * gamma**3)
    a_perp = F_perp / (m * gamma)
    acceleration = a_para + a_perp
    all_accelerations.append(acceleration.clone())
    all_para_forces.append(tmpc_para.clone())
    all_perp_forces.append(tmpc_perp.clone())
    # Convert lists to tensors
    all_positions = torch.stack(all_positions, dim=1)  # Shape: (size, M+1, 2)
    all_velocities = torch.stack(all_velocities, dim=1)  # Shape: (size, M+1, 2)
    all_para_forces = torch.stack(all_para_forces, dim=1)  # Shape: (size, M, 2)
    all_perp_forces = torch.stack(all_perp_forces, dim=1)  # Shape: (size, M, 2)
    all_accelerations = torch.stack(all_accelerations, dim=1)  # Shape: (size, M, 2)
    samples_1 = all_positions[:, -1]  # Final positions after M steps
    ### C_para = C_para * torch.abs(torch.sin(3.1415 * samples_0[:, 1:2] / 1e6))
    # all_para_forces = (C_para * torch.full((samples_0.shape[0], 1), 3.1415 * t / 10)).unsqueeze(1).repeat(1, 10, 1)
    ### C_perp = C_perp * torch.abs(torch.sin(3.1415 * samples_0[:, 1:2] / 1.25e5))
    # all_perp_forces = (C_perp * torch.full((samples_0.shape[0], 1), 3.1415 * 8 * t / 10)).unsqueeze(1).repeat(1, 10, 1)
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


# # Save Data

# In[16]:


def Save_Dataset(dataset, test_dataset):
    # Save dataset
    # torch.save(dataset, "dataset/train_halfmoons.pth")
    # torch.save(test_dataset, "dataset/test_halfmoons.pth")
    # Plotting
    plt.figure(figsize=(4.3, 4))
    plt.xlim(-R / 3e8, R / 3e8)
    plt.ylim(-R / 3e8, R / 3e8)
    plt.title(r'Halfmoons Dataset', fontsize=19)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # Plot the initial and final samples
    plt.scatter(test_dataset.pf[:, 0].cpu().numpy() / 3e8, test_dataset.pf[:, 1].cpu().numpy() / 3e8, alpha=0.5, c='#D9A0B3', label=r'$\pi_1$')
    plt.scatter(test_dataset.p0[:, 0].cpu().numpy() / 3e8, test_dataset.p0[:, 1].cpu().numpy() / 3e8, alpha=1.0, c='#2E59A7', label=r'$\pi_0$')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'size': 12})
    plt.tight_layout()
    # plt.savefig('figures/dataset_halfmoons.pdf', format='pdf', bbox_inches='tight')

def Plot_Trajectories(pm):
    # Determine the number of trajectories to plot (dataset_size / 10)
    total_samples = pm.shape[0]
    num_trajectories = max(1, int(total_samples / 20))
    # Randomly select indices
    indices = torch.randperm(total_samples)[:num_trajectories]
    # Create figure with similar style as before
    plt.figure(figsize=(4.3, 4))
    # Use the same axis limits (scaled as in Save_Dataset)
    plt.xlim(-R / 3e8, R / 3e8)
    plt.ylim(-R / 3e8, R / 3e8)
    plt.title("Trajectories generated\nby O1 model", fontsize=19)
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
    plt.savefig('figures/trajectory_1_halfmoons.pdf', format='pdf', bbox_inches='tight')
  
@torch.no_grad()
def draw_plot(rectified_flow, test_dataset, N=None):
  traj, _ = rectified_flow.sample_ode(test_dataset, N=N)
  plt.figure(figsize=(4.1,4))
  plt.xlim(-R / 3e8,R / 3e8)
  plt.ylim(-R / 3e8,R / 3e8)
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)
  plt.scatter(test_dataset.pf[:, 0].cpu().numpy() / 3e8, test_dataset.pf[:, 1].cpu().numpy() / 3e8, c='#D9A0B3', label=r'$\pi_1$', alpha=1.0)
  plt.scatter(traj[0][:, 0].cpu().numpy() / 3e8, traj[0][:, 1].cpu().numpy() / 3e8, c='#2E59A7' , label=r'$\pi_0$', alpha=1.0)
  plt.scatter(traj[-1][:, 0].cpu().numpy() / 3e8, traj[-1][:, 1].cpu().numpy() / 3e8, c='#9966CC' , label='Generated', alpha=1.0)
  plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'size': 9})
  plt.title('Halfmoons optimized\n with O1 loss', fontsize=19)
  plt.tight_layout()
  plt.savefig('figures/1_halfmoons.pdf', format='pdf', bbox_inches='tight')


# # Models

# In[17]:


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

class RectifiedFlow1():
  def __init__(self, model = None, num_steps=1000):
    self.model = model
    self.N = num_steps
  def get_train_tuple(self, dataset): # Need to rewrite
    t = torch.randint(0, 10, (dataset.p0.shape[0], 1))
    z_t = dataset.pm[torch.arange(dataset_size), t.squeeze(), :]
    First_target = dataset.v[:, t].squeeze()
    return z_t, t, First_target
  def frist_order_predict(self, z_t, t):
    first_order_pred = self.model(z_t, t)
    return first_order_pred
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
      pred = self.model(z, t)
      z = z.detach().clone() + pred * dt
      traj.append(z.detach().clone())
    first_order_loss = torch.sqrt((dataset.pf - z).abs().pow(2).sum(dim=1))
    first_order_loss_mean = first_order_loss.mean()
    print("Average L2 Norm:", first_order_loss_mean.item())
    traj_tensor = torch.stack(traj, dim=1)
    return traj, traj_tensor


# # Training Function

# In[18]:


def train_rectified_flow(rectified_flow, optimizer, dataset, inner_iters):
  loss_curve = []
  for _ in tqdm(range(inner_iters+1)):
    optimizer.zero_grad()
    z_t, t, first_order_gt = rectified_flow.get_train_tuple(dataset)
    first_order_pred = rectified_flow.frist_order_predict(z_t, t / rectified_flow.N)
    first_order_loss = (first_order_gt - first_order_pred).abs().pow(2).sum(dim=1)
    first_order_loss_mean = first_order_loss.mean()
    loss = first_order_loss_mean
    loss.backward()
    optimizer.step()
    loss_curve.append(np.log(loss.item())) ## to store the loss curve
  return rectified_flow, loss_curve


# # Create and Save Dataset

# In[19]:


dataset = Create_Dataset(dataset_size)
test_dataset = Create_Dataset(dataset_size * 3)
Save_Dataset(dataset, test_dataset)


# # Training

# In[20]:


input_dim = 2
reflow_iterations = 100
model = First_MLP(input_dim, hidden_num=100).to(device)
rectified_flow_1 = RectifiedFlow1(model, num_steps=M)
optimizer = torch.optim.Adam(rectified_flow_1.model.parameters(), lr=5e-3)
rectified_flow_1, loss_curve = train_rectified_flow(rectified_flow_1, optimizer, dataset, reflow_iterations)


# # Result

# In[21]:


draw_plot(rectified_flow_1, test_dataset)
Plot_Trajectories(rectified_flow_1.sample_ode(test_dataset)[1])

