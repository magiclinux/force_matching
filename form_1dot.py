#!/usr/bin/env python
# coding: utf-8

# # Create Dataset Function

# In[1]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm

# Training parameters
M = 10  # number of steps
dataset_size = 400  # number of data points
d = 1 / M  # time step (sampling interval)
# Distribution parameters
VAR = 0.3  # variance
R = 9e7  # plotting range
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


# # Save Dataset

# In[2]:


def Save_Dataset(dataset, test_dataset):
    # Save dataset
    torch.save(dataset, "dataset/train_onedot.pth")
    torch.save(test_dataset, "dataset/test_onedot.pth")
    # Plotting
    plt.figure(figsize=(4.1, 4))
    plt.xlim(-R / 3e8, R / 3e8)
    plt.ylim(-R / 3e8, R / 3e8)
    plt.title(r'Onedot Dataset', fontsize=19)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # Plot the initial and final samples
    plt.scatter(test_dataset.pf[:, 0].cpu().numpy() / 3e8, test_dataset.pf[:, 1].cpu().numpy() / 3e8, alpha=1.0, c='#D9A0B3', label=r'$\pi_1$')
    plt.scatter(test_dataset.p0[:, 0].cpu().numpy() / 3e8, test_dataset.p0[:, 1].cpu().numpy() / 3e8, alpha=1.0, c='#2E59A7', label=r'$\pi_0$')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'size': 12})
    plt.tight_layout()
    plt.savefig('figures/dataset_onedot.pdf', format='pdf', bbox_inches='tight')

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
    plt.title("Trajectories\ngenerated with ForM", fontsize=19)
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
    plt.savefig('figures/trajectory_form_onedot.pdf', format='pdf', bbox_inches='tight')

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
  plt.title('Onedot optimized\nwith ForM', fontsize=19)
  plt.tight_layout()
  plt.savefig('figures/form_onedot.pdf', format='pdf', bbox_inches='tight')


# # Models

# In[3]:


class First_MLP(nn.Module):
    def __init__(self, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, 1, bias=True)
        self.act = lambda x: torch.tanh(x)
    def forward(self, t):
        x = self.fc1(t)
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
    
class ForM(): 
  def __init__(self, model = None, num_steps=1000):
    self.model = model
    self.first_order_model = self.model[0]
    self.second_order_model = self.model[1]
    self.N = num_steps
  def get_train_tuple(self, dataset):
    t = torch.randint(0, self.N, (dataset.p0.shape[0], 1))
    para_force_gt = dataset.para[torch.arange(dataset_size), t.squeeze(), :]
    perp_force_gt = dataset.perp[torch.arange(dataset_size), t.squeeze(), :]
    return t, para_force_gt, perp_force_gt
  def para_and_perp_force_predict(self, t):
    para_force_pred = self.first_order_model(t)
    perp_force_pred = self.second_order_model(t)
    return para_force_pred, perp_force_pred
  @torch.no_grad()
  def sample_ode(self, dataset, N=None):
    if N is None:
      N = self.N
    traj = [] # to store the trajectory
    z = dataset.p0.detach().clone()
    batchsize = z.shape[0]
    velocity = z.clone()
    traj.append(z.detach().clone())
    for i in range(N):
      t = torch.ones((batchsize,1)) * i / N
      para_force_pred, perp_force_pred = self.para_and_perp_force_predict(t)
      speed_sq = (velocity ** 2).sum(dim=1, keepdim=True)
      gamma = 1 / torch.sqrt(1 - speed_sq / c**2)
      velocity_norm = torch.norm(velocity, dim=1, keepdim=True)
      velocity_unit = velocity / (velocity_norm + 1e-8)
      F_para = para_force_pred * velocity_unit
      perp_directions = torch.ones((z.shape[0], 1), dtype=torch.int)
      velocity_perp = torch.cat([-velocity[:, 1:2], velocity[:, 0:1]], dim=1)  # Rotate 90 degrees
      velocity_perp = velocity_perp / (velocity_norm + 1e-8)  # Normalize
      F_perp = perp_force_pred * velocity_perp * perp_directions 
      a_para = F_para / (m * gamma**3)
      a_perp = F_perp / (m * gamma)
      acceleration = a_para + a_perp
      velocity_new = velocity + acceleration * d
      z = z.detach().clone() + (velocity + velocity_new) * d / 2
      velocity = velocity_new
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
    t, para_force_gt, perp_force_gt = rectified_flow.get_train_tuple(dataset)
    para_force_pred, perp_force_pred = rectified_flow.para_and_perp_force_predict(t / rectified_flow.N)
    para_loss = (para_force_gt - para_force_pred).abs().pow(2).sum(dim=1)
    perp_loss = (perp_force_gt - perp_force_pred).abs().pow(2).sum(dim=1)
    para_loss_mean = para_loss.mean()
    perp_loss_mean = perp_loss.mean()
    loss = para_loss_mean + perp_loss_mean
    loss.backward()
    optimizer.step()
    loss_curve.append(np.log(loss.item()))
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
model1 = First_MLP(hidden_num=100).to(device)
model2 = First_MLP(hidden_num=100).to(device)
modellist = nn.ModuleList([model1, model2]).to(device)
ForM_model = ForM(modellist, num_steps=M)
optimizer = torch.optim.Adam(ForM_model.model.parameters(), lr=5e-3)
ForM_model, loss_curve = train_rectified_flow(ForM_model, optimizer, dataset, reflow_iterations)


# # Result

# In[7]:


draw_plot(ForM_model, test_dataset)
Plot_Trajectories(ForM_model.sample_ode(test_dataset)[1])

