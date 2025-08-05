import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from gym import spaces

import os
import time
import random
import argparse

from collections import deque, namedtuple
from typing import Tuple, Optional



class Landmark:

    def __init__(self,world,radius):
        # setable paramaters
        self.w = world
        self.radius = radius
        
        # variables
        self.pos = np.array([0,0],dtype=np.float32)

    def reset(self,low=None,high=None):
        low = -self.w.size if low is None else low
        high = self.w.size if high is None else high
        self.pos = np.random.uniform(low,high,self.pos.shape)
  
class Agent:

    def __init__(self,world,raduis,lsd,asd,acd):
        # setable paramaters
        self.w = world
        self.radius = raduis

        self.lsd = lsd
        self.asd = asd
        self.acd = acd
        
        # variables
        self.pos = np.array([0,0],dtype=np.float32)
        self.vel = np.array([0,0],dtype=np.float32)
        self.dist = None

        self.last_force = np.array([0,0],dtype=np.float32)
        self.last_vel = np.array([0,0],dtype=np.float32)

    def reset(self,low=None,high=None):
        low = -self.w.size if low is None else low
        high = self.w.size if high is None else high
        self.pos = np.random.uniform(low,high,self.pos.shape)
        self.vel = np.zeros(self.vel.shape,dtype=np.float32)
        self.dist = None

    def apply_force(self,force,scale=2.0):
        scaled = force*scale
        self.last_vel = self.vel.copy()
        self.vel += scaled*self.w.dt
        speed = np.linalg.norm(self.vel)
        if speed > self.w.max_speed:
            self.vel = self.vel/speed*self.w.max_speed
        self.pos += self.vel*self.w.dt
        # self.pos = np.clip(self.pos, -self.w.size + self.radius, self.w.size - self.radius)
        self.last_force = scaled*self.w.dt

def cos_center_to_outer(agent:Agent,landmark:Landmark):
    center_vector = landmark.pos-agent.pos
    l2_of_center_vector = np.linalg.norm(center_vector)
    if l2_of_center_vector>landmark.radius:
        speed_of_agent = np.linalg.norm(agent.vel)
        # TODO cos_VnC hesaplanirken bir hata oluyor
        sin_TnC = landmark.radius/max(l2_of_center_vector,1e-6) # cos of tagent vector and center vector
        cos_VnC = np.clip(np.dot(center_vector,agent.vel)/max(l2_of_center_vector*speed_of_agent,1e-6),-1,1) # cos of vel vector and center vector
        cos_TnC = np.sqrt(1-np.power(sin_TnC,2))
        sin_VnC = np.sqrt(1-np.power(cos_VnC,2))

        outter_sin = sin_VnC*cos_TnC - cos_VnC*sin_TnC
        if outter_sin > 0:
            return cos_TnC*cos_VnC+sin_TnC*sin_VnC
        else:
            return 1
    return 1

def cos_outter_to_outter(agent:Agent,landmark:Landmark):
    rel_pos_vec = landmark.pos-agent.pos
    rel_pos_len = np.linalg.norm(rel_pos_vec)
    if rel_pos_len > agent.radius + landmark.radius:
        cos = (agent.radius + landmark.radius)/(rel_pos_len+1e-6)
        sin = np.sqrt(1-np.power(cos,2))
        unit_rel_pos_vec = rel_pos_vec/rel_pos_len
        
        if (agent.vel-unit_rel_pos_vec)[0] < 0:
            rotation_matrix = np.array([[cos,sin],[-sin,cos]]) # negative way
            rotation_matrix90 = np.array([[0,-1],[1,0]]) # positive way
        else:
            rotation_matrix = np.array([[cos,-sin],[sin,cos]]) # positive way
            rotation_matrix90 = np.array([[0,1],[-1,0]]) # negative way


        tangent_vec = np.matmul(rotation_matrix90,np.matmul(rotation_matrix,unit_rel_pos_vec))
        if (agent.vel-tangent_vec)[0] < 0:
            cos = tangent_vec.dot(agent.vel)/(np.linalg.norm(tangent_vec)*np.linalg.norm(agent.vel)+1e-6)
        else:
            cos = 1
        return cos
    return 0

def cos_outter_to_outterV2(agent:Agent,landmark:Landmark):
    rel_pos_vec = landmark.pos-agent.pos
    rel_pos_len = np.linalg.norm(rel_pos_vec)
    if rel_pos_len > agent.radius + landmark.radius:
        sin = (agent.radius + landmark.radius)/(rel_pos_len+1e-6) # inner tagent angle
        cos = np.sqrt(1-np.power(sin,2))
        unit_rel_pos_vec = rel_pos_vec/rel_pos_len
        
        rotation_matrix,relative_position_factor = (np.array([[cos,-sin],[sin,cos]]), 1) if np.cross(agent.vel,unit_rel_pos_vec) < 0 else (np.array([[cos,sin],[-sin,cos]]),-1)
        tangent_vec = np.matmul(rotation_matrix,unit_rel_pos_vec)
        if np.cross(agent.vel,tangent_vec)*relative_position_factor < 0:
            cos = tangent_vec.dot(agent.vel)/(np.linalg.norm(tangent_vec)*np.linalg.norm(agent.vel)+1e-6)
        else:
            cos = 1
        return 4 if cos > 0.999 else cos
    return 0




class BaseWorld:

    def __init__(self,size=10, max_episode_steps=200):
        
        # world paramaters
        self.max_speed = None
        self.dt = None
        self.n_agents = None
        self.n_landmarks = None

        # entity paramaters
        agent_raduis = 1
        landmark_raduis = 2
        goal_raduis = 0.5

        # setable world paramaters
        self.size = size
        self.max_episode_steps = max_episode_steps

        # variables
        self.agents = None
        self.landmarks = None
        self.goal:Landmark = None

        self.current_step = None
        self._state_dim = None

    @property
    def state_dim(self):
        if self._state_dim is None:
            self._state_dim = self.observation(self.agents[0]).shape[0] 
        return self._state_dim

    def other_agents(self,agent):
        return [o_agent for o_agent in self.agents if o_agent is not agent]

    def reset(self):
        invalid = []
        self.goal.reset()
        invalid.append((self.goal.pos,self.goal.radius))
        for landmark in self.landmarks:
            while 1:
                landmark.reset()
                for cnt,radius in invalid:
                    if np.linalg.norm(landmark.pos-cnt) < radius+landmark.radius:
                        break
                else:
                    invalid.append((landmark.pos,landmark.radius))
                    break
        
        for agent in self.agents:
            while 1:
                agent.reset()
                for cnt,radius in invalid:
                    if np.linalg.norm(agent.pos-cnt) < radius+agent.radius:
                        break
                else:
                    invalid.append((agent.pos,agent.radius))
                    agent.dist = np.linalg.norm(agent.pos-self.goal.pos)
                    break
    
    def observation(self, agent: Agent):
        obs = [
            agent.pos[0], agent.pos[1],
            agent.vel[0], agent.vel[1],
        ]
        
        goal_relative = self.goal.pos - agent.pos
        goal_distance = np.linalg.norm(goal_relative)
        goal_direction = goal_relative / max(goal_distance, 0.01)
        cos = cos_outter_to_outterV2(agent,self.goal)
        # goal_direction = (np.degrees(np.arctan2(goal_relative[1],goal_relative[0]))+360)%360
        
        obs.extend([
            goal_relative[0], goal_relative[1],
            goal_distance,
            goal_direction[0], goal_direction[1],
            # goal_direction
            cos
        ])
        
    
        for landmark in self.landmarks:
            obstacle_relative = landmark.pos - agent.pos
            obstacle_distance = np.linalg.norm(obstacle_relative)
            obstacle_direction = obstacle_relative / max(obstacle_distance, 0.01)
            cos = cos_outter_to_outterV2(agent,landmark)
            # obstacle_direction = (np.degrees(np.arctan2(obstacle_relative[1],obstacle_relative[0]))+360)%360
            
        
            obs.extend([
                obstacle_relative[0], obstacle_relative[1],
                obstacle_distance,
                obstacle_direction[0], obstacle_direction[1],
                # obstacle_direction,
                max(0, landmark.radius + agent.radius - obstacle_distance),
                cos
            ])
        
    
        for o_agent in self.other_agents(agent):
            other_relative = o_agent.pos - agent.pos
            other_distance = np.linalg.norm(other_relative)
            other_direction = other_relative / max(other_distance, 0.01)
            # other_direction = (np.degrees(np.arctan2(other_relative[1],other_relative[0]))+360)%360
            
            obs.extend([
                other_relative[0], other_relative[1],
                other_distance,
                other_direction[0], other_direction[1],
                # other_direction,
                o_agent.vel[0], o_agent.vel[1],
                o_agent.vel[0]-agent.vel[0], o_agent.vel[1]-agent.vel[1],
            ])
        
        return np.array(obs, dtype=np.float32)

    def reward(self,agent:Agent):
        raise NotImplementedError()

class SimpleNavigationEnv:
    
    def __init__(self, world_size=10.0,max_step=200):
        
        self.world = World(size=world_size, max_episode_steps=max_step)
        
        self.state_dim = self.world.state_dim
        self.action_dim = 2
        
        low = np.array([-world_size] * self.state_dim)
        high = np.array([world_size] * self.state_dim)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.fig = None
        self.ax = None
        
    def reset(self,a=1000000) -> np.ndarray:
        self.world.current_step = 0
        self.world.reset(a)
        return [self.world.observation(agent) for agent in self.world.agents]
    
        
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.world.current_step += 1
        
        
        # Action'ı clamp et
        # Fizik simulation
        for i, agent in enumerate(self.world.agents):
            action = np.clip(actions[i], -1.0, 1.0)
            agent.apply_force(action)
        
        # Reward hesaplama ve terminal condition check
        rewards = []; dones = []; infos = [] 
        for agent in self.world.agents:
            reward,done,info = self.world.reward(agent)
            rewards.append(reward)
            dones.append(done)
            infos.append(info) 

        # Observation
        states = [self.world.observation(agent) for agent in self.world.agents]
        
        return states, rewards, dones, infos
    
    
    def render(self, mode='human'):
        """Environment'ı görselleştir"""
        try:
            
            # Backend kontrol et ve ayarla
            if self.fig is None:
                # Interactive backend ayarla
                if mode == 'human':
                    try:
                        matplotlib.use('TkAgg') # En uyumlu backend
                    except:
                        try:
                            matplotlib.use('Qt5Agg')
                        except:
                            try:
                                matplotlib.use('Agg') # Fallback
                                print("⚠️ Display bulunamadı, render edilmiyor")
                                return
                            except:
                                print("❌ Matplotlib backend sorunu, render skip ediliyor")
                                return
                
                self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
                plt.ion() # Interactive mode ON
                
                # Window başlığı
                self.fig.canvas.manager.set_window_title('MADDPG Navigation Environment')
        
        except ImportError:
            if not hasattr(self, '_render_warning_shown'):
                print("⚠️ Matplotlib bulunamadı. Render için 'pip install matplotlib' çalıştırın")
                self._render_warning_shown = True
            return
        except Exception as e:
            if not hasattr(self, '_render_error_shown'):
                print(f"⚠️ Render hatası: {e}")
                self._render_error_shown = True
            return
        
        try:
            self.ax.clear()
            # World boundaries
            self.ax.set_xlim(-self.world.size, self.world.size)
            self.ax.set_ylim(-self.world.size, self.world.size)
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_facecolor('#f8f9fa') # Light background
            
            # Goal (yeşil daire)
            goal_circle = patches.Circle(self.world.goal.pos, self.world.goal.radius, color='green', alpha=0.8, label='Goal', zorder=3)
            self.ax.add_patch(goal_circle)
            
            # Obstacles (kırmızı kareler)
            for i, obs_pos in enumerate([x.pos for x in self.world.landmarks]):
                obs_area_circle = patches.Circle(obs_pos, self.world.landmarks[0].radius+self.world.agents[0].lsd , color='red', alpha=0.3, zorder=3)
                obs_circle = patches.Circle(obs_pos, self.world.landmarks[0].radius, color='red', alpha=0.8, label=f'Obstacle {i+1}', zorder=3)
                self.ax.add_patch(obs_area_circle)
                self.ax.add_patch(obs_circle)
            

            for i, agent in enumerate(self.world.agents):
                agent_asc = patches.Circle(agent.pos, agent.radius + agent.asd, color='blue', alpha=0.1, zorder=4)
                agent_acc = patches.Circle(agent.pos, agent.radius + agent.acd, color='blue', alpha=0.1, zorder=4)
                agent_circle = patches.Circle(agent.pos, agent.radius, color='blue', alpha=0.9, label=f'Agent {i+1}', zorder=4)
                self.ax.add_patch(agent_asc)
                self.ax.add_patch(agent_acc)
                self.ax.add_patch(agent_circle)
            
                if np.linalg.norm(agent.vel) > 0.1:
                    vel_scale = 2.0 # Velocity görünürlüğü için scale
                    self.ax.arrow(agent.pos[0], agent.pos[1], 
                                    agent.vel[0] * vel_scale, agent.vel[1] * vel_scale,
                                    head_width=0.3, head_length=0.2, fc='navy', ec='navy', 
                                    alpha=0.7, linewidth=2, zorder=4)
                    self.ax.arrow(agent.pos[0], agent.pos[1], agent.last_force[0]*vel_scale*2,agent.last_force[1]*vel_scale*2,head_width = 0.3, head_length=0.2,fc="red",ec="red",alpha=0.7,linewidth=2, zorder=4)
                    
            
            # Path trail (son birkaç pozisyonu göster)
            # Distance to goal line
            # Info panel
            info_text = f'Step: {self.world.current_step}/{self.world.max_episode_steps}\n'
            for i,agent in enumerate(self.world.agents):
                if not hasattr(self, 'position_history'):
                    self.position_history = [deque(maxlen=50) for _ in range(self.world.n_agents)]
                self.position_history[i].append(agent.pos.copy())
            
                if len(self.position_history[i]) > 1:
                    positions = np.array(self.position_history[i])
                    self.ax.plot(positions[:, 0], positions[:, 1], 
                                color='lightblue', alpha=0.6, linewidth=2, zorder=1)
            

                self.ax.plot([agent.pos[0], self.world.goal.pos[0]], 
                            [agent.pos[1], self.world.goal.pos[1]], 
                            color='gray', linestyle='--', alpha=0.4, linewidth=1, zorder=1)
            
                distance = np.linalg.norm(agent.pos - self.world.goal.pos)
                speed = np.linalg.norm(agent.vel)
                
                info_text += f'Dist: {distance:.2f} Speed: {speed:.2f}\n'
                            
            # Info box with background
            self.ax.text(-self.world.size + 3.75, self.world.size - 0.1, info_text,
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                                alpha=0.9, edgecolor="gray"),
                        verticalalignment='top', zorder=6)
            
            # Progress bar
            progress = self.world.current_step / self.world.max_episode_steps
            bar_width = 3.0
            bar_height = 0.3
            bar_x = self.world.size - bar_width - 0.5
            bar_y = self.world.size - 0.8
            
            # Progress background
            progress_bg = patches.Rectangle((bar_x, bar_y), bar_width, bar_height,
                                            color='lightgray', alpha=0.7, zorder=2)
            self.ax.add_patch(progress_bg)
            
            # Progress fill
            progress_fill = patches.Rectangle((bar_x, bar_y), bar_width * progress, bar_height,
                                            color='orange' if progress < 0.8 else 'red', 
                                            alpha=0.8, zorder=3)
            self.ax.add_patch(progress_fill)
            
            self.ax.text(bar_x + bar_width/2, bar_y + bar_height + 0.2, 
                        f'Progress: {progress*100:.0f}%',
                        horizontalalignment='center', fontsize=9, fontweight='bold')
            
            # Title and legend
            self.ax.set_title('MADDPG Navigation Environment', fontsize=14, fontweight='bold', pad=20)
            self.ax.legend(loc='upper left', framealpha=0.9)
            
            # Grid styling
            self.ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Update display
            plt.tight_layout()
            
            if mode == 'human':
                plt.draw()
                plt.pause(0.001) # Çok kısa pause
                
                # Non-blocking show
                if hasattr(self.fig.canvas, 'flush_events'):
                    self.fig.canvas.flush_events()
            
            elif mode == 'rgb_array':
                # Convert plot to RGB array
                self.fig.canvas.draw()
                buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                return buf
                    
        except Exception as e:
            if not hasattr(self, '_render_error_shown'):
                print(f"⚠️ Render display hatası: {e}")
                print("💡 Çözüm: Farklı matplotlib backend deneyin veya render=False kullanın")
                self._render_error_shown = True
    
    def close(self):
        """Environment'ı kapat"""
        try:
            if hasattr(self, 'fig') and self.fig is not None:
                import matplotlib.pyplot as plt
                plt.close(self.fig)
                self.fig = None
                self.ax = None
                print("🔒 Render window kapatıldı")
        except:
            pass # Silent fail on close


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, action_low=None, action_high=None):
        super(Actor, self).__init__()
        
        # Action scaling parametreleri
        if action_low is None:
            self.action_low = torch.tensor([-1.0] * action_dim)
        else:
            self.action_low = torch.tensor(action_low, dtype=torch.float32)
            
        if action_high is None:
            self.action_high = torch.tensor([1.0] * action_dim)
        else:
            self.action_high = torch.tensor(action_high, dtype=torch.float32)
        
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.tanh(self.fc5(x)) # [-1, 1] aralığında
        
        # Action scaling: [-1,1] -> [action_low, action_high]
        scaled_action = x * self.action_scale.to(x.device) + self.action_bias.to(x.device)
        return scaled_action

class Critic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(total_state_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1)
        
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)



class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.buffer = deque(maxlen=int(max_size))
    
    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def size(self):
        return len(self.buffer)
    

Experience = namedtuple('Experience', ['states', 'actions', 'rewards', 'next_states', 'dones', 'priority'])

class PriBuffer:

    def __init__(self, capacity, alpha = 0.6, beta = 0.4, beta_increment = 0.001, epsilon = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
    def add(self, states, actions, rewards, next_states, dones, td_error= None):

        priority = self.max_priority if td_error is None else abs(td_error) + self.epsilon
        
        experience = Experience(states, actions, rewards, next_states, dones, priority)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities[self.position] = priority
        self.max_priority = max(self.max_priority, priority)
        
        self.position = (self.position + 1) % self.capacity
        
    def _get_rank_probabilities(self) -> np.ndarray:

        size = len(self.buffer)
        
        sorted_indices = np.argsort(self.priorities[:size])[::-1]
        ranks = np.empty(size, dtype=np.int32)
        ranks[sorted_indices] = np.arange(size) + 1
        
       
        probabilities = 1.0 / (ranks ** self.alpha)
        probabilities = probabilities / probabilities.sum()
        
        return probabilities
        
    def sample(self, batch_size):

        size = len(self.buffer)
        if size == 0:
            raise ValueError("Buffer boş!")
            
        probabilities = self._get_rank_probabilities()
        indices = np.random.choice(size, batch_size, p=probabilities)
        
        weights = (size * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        
        batch = [self.buffer[idx] for idx in indices]
        
        return batch, indices, weights
        
    def update_priorities(self, indices, td_errors):

        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + self.epsilon
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return len(self.buffer)
    
    def size(self):
        return len(self.buffer)

class MADDPG:
    def __init__(self, num_agents, state_dims, action_dims, hidden_dim=64, 
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.95, tau=0.01, 
                 action_low=None, action_high=None, buffer_size=1e6):
        
        self.num_agents = num_agents
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.gamma = gamma
        self.tau = tau
        
        if action_low is None:
            self.action_low = [[-1.0] * action_dims[i] for i in range(num_agents)]
        else:
            self.action_low = action_low
            
        if action_high is None:
            self.action_high = [[1.0] * action_dims[i] for i in range(num_agents)]
        else:
            self.action_high = action_high
        
        self.total_state_dim = sum(state_dims)
        self.total_action_dim = sum(action_dims)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for i in range(num_agents):
            actor = Actor(state_dims[i], action_dims[i], hidden_dim, 
                         self.action_low[i], self.action_high[i]).to(self.device)
            target_actor = Actor(state_dims[i], action_dims[i], hidden_dim, 
                               self.action_low[i], self.action_high[i]).to(self.device)
            
            critic = Critic(self.total_state_dim, self.total_action_dim, hidden_dim).to(self.device)
            target_critic = Critic(self.total_state_dim, self.total_action_dim, hidden_dim).to(self.device)
            
            target_actor.load_state_dict(actor.state_dict())
            target_critic.load_state_dict(critic.state_dict())
            
            actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
            critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)
            
            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)
            self.actor_optimizers.append(actor_optimizer)
            self.critic_optimizers.append(critic_optimizer)
        
        self.memory = PriBuffer(int(buffer_size))
        
        self.noise_std = 0.2
        
    def select_action(self, states, add_noise=True):
        actions = []
        
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.actors[i](state).cpu().numpy().flatten()
            
            if add_noise:
                noise = np.random.normal(0, self.noise_std, size=action.shape)
                action = action + noise
                
                action = np.clip(action, self.action_low[i], self.action_high[i])
            
            actions.append(action)
        
        return actions
    
    def store_transition(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)
    
    def learn(self, batch_size=64):
        
        if self.memory.size() < batch_size:
            return
        
        batch, indices, weights = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(np.array([exp.states for exp in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([exp.actions for exp in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([exp.rewards for exp in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([exp.next_states for exp in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([exp.dones for exp in batch])).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        batch_states = []
        batch_actions = []
        batch_next_states = []
        
        for i in range(self.num_agents):
            agent_states = states[:, i, :self.state_dims[i]]
            agent_actions = actions[:, i, :self.action_dims[i]]
            agent_next_states = next_states[:, i, :self.state_dims[i]]
            
            batch_states.append(agent_states)
            batch_actions.append(agent_actions)
            batch_next_states.append(agent_next_states)
        
        all_states = torch.cat(batch_states, dim=1)
        all_actions = torch.cat(batch_actions, dim=1)
        all_next_states = torch.cat(batch_next_states, dim=1)
        
        td_errors = torch.zeros((batch_size,self.num_agents),dtype=torch.float)
        for agent_idx in range(self.num_agents):
            #CRITIC
            target_next_actions = []
            for i in range(self.num_agents):
                target_action = self.target_actors[i](batch_next_states[i])
                target_next_actions.append(target_action)
            
            target_next_actions = torch.cat(target_next_actions, dim=1)
            
            target_q = self.target_critics[agent_idx](all_next_states, target_next_actions)
            target_q = rewards[:, agent_idx].unsqueeze(1) + \
                      (self.gamma * target_q * (1 - dones[:, agent_idx].unsqueeze(1)))
            
            current_q = self.critics[agent_idx](all_states, all_actions)
            
            # TDerror
            temp = (target_q.detach()-current_q)[:,0]
            td_errors[:,agent_idx] = abs(temp)

            critic_loss = (weights*(temp**2)).mean()
            self.critic_optimizers[agent_idx].zero_grad()
            critic_loss.backward(retain_graph=(True if agent_idx < self.num_agents-1 else False))
            torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), 1.0)
            self.critic_optimizers[agent_idx].step()
            
            
            #ACTOR UPDATE
            current_actions = []
            for i in range(self.num_agents):
                if i == agent_idx:
                    current_action = self.actors[i](batch_states[i])
                else:
                    current_action = batch_actions[i].detach()
                current_actions.append(current_action)
            
            current_actions = torch.cat(current_actions, dim=1)
            
            actor_loss = -(weights.unsqueeze(1) * self.critics[agent_idx](all_states, current_actions)).mean()
            
            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 1.0)
            self.actor_optimizers[agent_idx].step()
            
            self._soft_update(self.critics[agent_idx], self.target_critics[agent_idx])
            self._soft_update(self.actors[agent_idx], self.target_actors[agent_idx])
        
        self.memory.update_priorities(indices,td_errors.mean(dim=0).detach().cpu().numpy())
    
    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save_models(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'target_actors': [target_actor.state_dict() for target_actor in self.target_actors],
            'target_critics': [target_critic.state_dict() for target_critic in self.target_critics],
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizers': [opt.state_dict() for opt in self.critic_optimizers],
            'noise_std': self.noise_std
        }
        torch.save(checkpoint, filepath)
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath):
        """Model yükleme"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.critics[i].load_state_dict(checkpoint['critics'][i])
            self.target_actors[i].load_state_dict(checkpoint['target_actors'][i])
            self.target_critics[i].load_state_dict(checkpoint['target_critics'][i])
            self.actor_optimizers[i].load_state_dict(checkpoint['actor_optimizers'][i])
            self.critic_optimizers[i].load_state_dict(checkpoint['critic_optimizers'][i])
        
        if 'noise_std' in checkpoint:
            self.noise_std = checkpoint['noise_std']
        
        print(f"Models loaded from {filepath}")


class TrainingLogger:
    """Training metrics logger"""
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.episode_rewards = []
        self.success_rates = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        
    def log_episode(self, episode, reward, success, length):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        # Success rate (son 100 episode)
        successes = [success] if not hasattr(self, '_success_buffer') else self._success_buffer + [success]
        if len(successes) > 100:
            successes = successes[-100:]
        self._success_buffer = successes
        self.success_rates.append(np.mean(successes) * 100)
        
    def plot_metrics(self, save_path="training_results.png"):
        """Training metrics plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        ax1.plot(self.episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
        if len(self.episode_rewards) >= 50:
            moving_avg = np.convolve(self.episode_rewards, np.ones(50)/50, mode='valid')
            ax1.plot(range(49, len(self.episode_rewards)), moving_avg, 
                    color='red', linewidth=2, label='Moving Average (50)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Success rate
        ax2.plot(self.success_rates, color='green', linewidth=2)
        ax2.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Target (90%)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate (Rolling 100 episodes)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Episode lengths
        ax3.plot(self.episode_lengths, alpha=0.5, color='orange')
        if len(self.episode_lengths) >= 50:
            moving_avg = np.convolve(self.episode_lengths, np.ones(50)/50, mode='valid')
            ax3.plot(range(49, len(self.episode_lengths)), moving_avg, 
                    color='red', linewidth=2, label='Moving Average (50)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Episode Length')
        ax3.set_title('Episode Lengths')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Buffer size
        ax4.text(0.5, 0.5, f'Final Success Rate: {self.success_rates[-1]:.1f}%\n'
                            f'Best Reward: {max(self.episode_rewards):.2f}\n'
                            f'Avg Episode Length: {np.mean(self.episode_lengths[-100:]):.1f}',
                 transform=ax4.transAxes, fontsize=12, 
                 verticalalignment='center', horizontalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        ax4.axis('off')
        ax4.set_title('Training Summary')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training plots saved to {save_path}")

class FileLogger:

    target_dir = ""
    def_file = "log.log"

    @staticmethod
    def log_time():
        return datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")

    @staticmethod
    def log(*args,sep = " ", end = "\n", file=None, reset=False):
        if file is None:file = FileLogger.def_file
        text = sep.join(list(map(lambda x:str(x),args)))+end
        path = FileLogger.target_dir+"/"+file
        if os.path.exists(path) and not reset:
            with open(path, "a", encoding="utf-8") as f: f.write(FileLogger.log_time() + " " + text)
        else:
            with open(path, "w", encoding="utf-8") as f: f.write(FileLogger.log_time() + " " + text)
            
        print(text,sep="",end="")
    
    @staticmethod
    def abslog(*args,sep = " ", end = "\n", file=None, reset=False):
        if file is None:file = FileLogger.def_file
        text = sep.join(list(map(lambda x:str(x),args)))+end
        path = FileLogger.target_dir+"/"+file
        if os.path.exists(path) and not reset:
            with open(path, "a", encoding="utf-8") as f: f.write(FileLogger.log_time() + " " + text)
        else:
            with open(path, "w", encoding="utf-8") as f: f.write(FileLogger.log_time() + " " + text)


epp = 0
def train_agent(args):
    global epp
    print("="*60)
    print("MADDPG Navigation Training Başlıyor!")
    print("="*60)
    FileLogger.target_dir = args.log_dir
    
    env = SimpleNavigationEnv(world_size=args.world_size,max_step=args.max_steps_per_episode)

    num_agents = env.world.n_agents
    state_dims = [env.state_dim for _ in range(num_agents)]
    action_dims = [env.action_dim for _ in range(num_agents)]
    action_low = [[-1.0, -1.0] for _ in range(num_agents)]
    action_high = [[1.0, 1.0] for _ in range(num_agents)]
    
    print(f"Environment: State dim={env.state_dim}, Action dim={env.action_dim}")
    print(f"Action range: {action_low[0]} to {action_high[0]}")
    
    # MADDPG agent oluştur
    cause = {"steps":0,"agent":0, "reached":0,"defind":0,"obstacle":0}
    maddpg = MADDPG(
        num_agents=num_agents,
        state_dims=state_dims,
        action_dims=action_dims,
        hidden_dim=128,
        lr_actor= 1e-4,
        lr_critic= 1e-3,
        gamma= 0.95,
        tau= 0.01,
        action_low=action_low,
        action_high=action_high,
        buffer_size=100000
    )
    
    logger = TrainingLogger(args.log_dir)
    
    print(f"Training for {args.num_episodes} episodes...")
    start_time = time.time()
    best_success_rate = 0

    
    for episode in range(args.num_episodes):
        # Episode başlat
        states = env.reset(episode)
        
        episode_reward = np.zeros((num_agents,))
        episode_success = np.zeros((num_agents,))
        step = 0
        if episode >= 2000:
            epp = 1
        
        for step in range(args.max_steps_per_episode):
            # Action selection
            actions = maddpg.select_action(states, add_noise=True)
            # Environment step
            next_states, rewards, dones, infos = env.step(actions)
            
            # Store transition
            maddpg.store_transition(
                states=states, 
                actions=actions, 
                rewards=rewards, 
                next_states=next_states, 
                dones=dones
            )
            
            # Learn
            if episode > args.warmup_episodes:
                maddpg.learn(batch_size=args.batch_size)
            
            # Update
            states = next_states
            episode_reward += np.array(rewards)
            
            # Visualization (bazı episode'lar için)
            if args.render and episode % args.render_interval == 0:
                try:
                    env.render()
                    time.sleep(0.01) # Slow down for visualization
                except:
                    pass
            
            break_flag = False
            for i,done in enumerate(dones):
                if done:
                    episode_success[i] = 1 if infos[i].get('success',False) else 0
                    cause[infos[i].get("reason","not_defind").split("_")[1]] += 1
                    break_flag = True
            if break_flag:
                break

        # Episode logging
        logger.log_episode(episode, episode_reward.sum(), episode_success.max(), step + 1)
        
        # Progress logging
        if episode % args.log_interval == 0:
            avg_reward = np.mean(logger.episode_rewards[-100:]) if len(logger.episode_rewards) >= 100 else np.mean(logger.episode_rewards)
            success_rate = logger.success_rates[-1] if logger.success_rates else 0
            elapsed_time = time.time() - start_time
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward.sum():6.2f} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Success Rate: {success_rate:5.1f}% | "
                  f"Steps: {step+1:3d} | "
                  f"Time: {elapsed_time/60:.1f}m | "
                  f"Result: {'✅' if episode_success.max() else '❌'}")
            
            FileLogger.abslog("causes:",cause)
            
            # Model kaydet (best success rate)
            if success_rate > best_success_rate and episode > args.warmup_episodes:
                best_success_rate = success_rate
                maddpg.save_models(f"{args.model_dir}/best_model.pth")
                print(f" 🎯 New best success rate: {success_rate:.1f}%")
        
        # Periodic model save
        if episode % args.save_interval == 0 and episode > 0:
            maddpg.save_models(f"{args.model_dir}/checkpoint_episode_{episode}.pth")
        
        # Early stopping
        if logger.success_rates and logger.success_rates[-1] > 95 and episode > 500:
            print(f"🎉 Training completed! Success rate > 95% at episode {episode}")
            break
        
        # Exploration noise decay
        if episode % 200 == 0 and episode > 0:
            maddpg.noise_std = max(0.05, maddpg.noise_std * 0.99)
            print(f" 🔧 Noise reduced to {maddpg.noise_std:.3f}")
    
    # Final save
    maddpg.save_models(f"{args.model_dir}/final_model.pth")
    
    # Results
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("Training Completed!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Final success rate: {logger.success_rates[-1]:.1f}%")
    print(f"Best success rate: {best_success_rate:.1f}%")
    print(f"Average episode length: {np.mean(logger.episode_lengths[-100:]):.1f}")
    print("="*60)
    
    # Plot results
    logger.plot_metrics(f"{args.log_dir}/training_results.png")
    
    env.close()
    return maddpg, logger


def test_agent(model_path, num_episodes=10, render=True):
    """Train edilmiş agent'ı test et"""
    print(f"Testing agent from {model_path}")
    
    # Environment ve MADDPG oluştur
    env = SimpleNavigationEnv(world_size=10,max_step=200)

    num_agents = env.world.n_agents
    maddpg = MADDPG(
        num_agents=num_agents,
        state_dims=[env.state_dim for _ in range(num_agents)],
        action_dims=[env.action_dim for _ in range(num_agents)],
        action_low=[[-1.0, -1.0] for _ in range(num_agents)],
        action_high=[[1.0, 1.0] for _ in range(num_agents)],
        hidden_dim=128
    )
    
    # Model yükle
    maddpg.load_models(model_path)
    
    # Test
    success_count = 0
    total_rewards = []
    for episode in range(num_episodes):
        states = env.reset()
        total_reward = np.zeros((num_agents,))
        
        print(f"Test Episode {episode + 1}")
        
        for step in range(200):
            # Trained action (no noise)
            actions = maddpg.select_action(states, add_noise=False)
            
            states, rewards, dones, infos = env.step(actions)
            total_reward += np.array(rewards)
            
            if render:
                try:
                    env.render()
                    time.sleep(0.03) # Test sırasında biraz daha yavaş
                except Exception as e:
                    if not hasattr(test_agent, '_render_warning_shown'):
                        print(f"⚠️ Visualization hatası: {e}")
                        print("💡 Render olmadan test devam ediyor...")
                        test_agent._render_warning_shown = True
                        render = False # Render'ı kapat



            break_flag = False
            for i,done in enumerate(dones):
                if done:
                    info = infos[i]
                    if info.get('success', False):
                        success_count += 1
                        print(f" ✅ SUCCESS! Reward: {total_reward.sum():.2f}, Steps: {step+1}")
                    else:
                        print(f" ❌ FAILED! Reason: {info.get('reason', 'unknown')}, Reward: {total_reward.sum():.2f}")
                    break_flag = True  
                    break
            if break_flag:
                break
            
        total_rewards.append(total_reward.sum())
    
    print(f"\nTest Results:")
    print(f"Success Rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    
    env.close()


class World(BaseWorld):

    def __init__(self,size=10, max_episode_steps=200):
        
        # world paramaters
        self.max_speed = 2.0
        self.dt = 0.1
        self.n_agents = 2
        self.n_landmarks = 2

        # entity paramaters
        self.agent_raduis = 0.75
        self.landmark_raduis = 1.5
        self.goal_raduis = 0.5

        self.lsd = self.landmark_raduis*1.30
        self.asd = self.agent_raduis*2
        self.acd = self.agent_raduis*6

        # setable world paramaters
        self.size = size
        self.max_episode_steps = max_episode_steps

        # variables
        self.agents = [Agent(self,self.agent_raduis,self.lsd,self.asd,self.acd) for _ in range(self.n_agents)]
        self.landmarks = [Landmark(self,self.landmark_raduis) for _ in range(self.n_landmarks)]
        self.goal = Landmark(self,self.goal_raduis)

        self.current_step = 0
        self._state_dim = None
        self._radius_buffer = {}

        self._radius_buffer["lsd-0-0"] = self.agent_raduis+self.lsd+self.landmark_raduis
        self._radius_buffer["asd-0-0"] = self.agent_raduis+self.asd+self.agent_raduis
        self._radius_buffer["acd-0-0"] = self.agent_raduis+self.acd+self.agent_raduis



    def get_radius(self,ca,cl=0,cg=0):
        key = f"{ca}-{cl}-{cg}"
        radius = self._radius_buffer.get(key,None)
        if radius is None:
            radius = ca*self.agent_raduis+cl*self.landmark_raduis+cg*self.goal_raduis
            self._radius_buffer[key] = radius
        return radius
    
    def _get_radius(self,key): # get radius
        return self._radius_buffer[key]

    def reward(self,agent:Agent):
        reward = 0.0
        done = False
        info = {}

        # pull_reward_vec = np.zeros((2,),np.float32)
        # push_reward_vec = np.zeros((2,),np.float32)

        distance_to_goal = np.linalg.norm(agent.pos - self.goal.pos)
        if distance_to_goal < (agent.radius + self.goal.radius):
            reward += 2000.0
            done = True
            info['success'] = True
            info['reason'] = 'goal_reached'
            return reward, done, info
        else:
            reward += np.clip(1/(distance_to_goal-self.get_radius(1,0,1))*np.exp(self.current_step/100)*(cos_outter_to_outterV2(agent,self.goal)+1),-100,100)

        penlaty_vec = np.zeros((2,),np.float32)
        for i, landmark in enumerate(self.landmarks):
            rel_vec = landmark.pos - agent.pos
            rel_vec_len = np.linalg.norm(rel_vec)
            if rel_vec_len < (agent.radius + landmark.radius):
                reward += -2000.0
                done = True
                info['success'] = False
                info['reason'] = f'hit_obstacle_{i}'
                return reward, done, info
            elif rel_vec_len < self.get_radius("lsd"):
                # reward += (-np.exp(-distance_to_landmark)*37.951)*(cos_outter_to_outterV2(agent,landmark)+1)
                penlaty_vec += (-np.exp(-rel_vec_len)*37.951)*rel_vec/(rel_vec_len+1e-6)*(cos_outter_to_outterV2(agent,landmark)+1)*np.linalg.norm(agent.vel)

        for i, o_agent in enumerate(self.other_agents(agent)):
            rel_vec = o_agent.pos - agent.pos
            rel_vec_len = np.linalg.norm(rel_vec)
            if rel_vec_len < (agent.radius + o_agent.radius):
                reward += -2000.0
                done = True
                info['success'] = False
                info['reason'] = f'hit_agent_{self.agents.index(o_agent)}'
                return reward, done, info
            elif rel_vec_len <= self.get_radius("asd"):
                penlaty_vec += -np.exp(-rel_vec_len)*18.9755*rel_vec/(rel_vec_len+1e-6)
            elif rel_vec_len <= self.get_radius("acd"):
                reward += np.abs(agent.vel.dot(o_agent.vel)/(np.linalg.norm(o_agent.vel)+1e-6))*cos_outter_to_outterV2(agent,self.goal)
                pass
            else:
                penlaty_vec += np.log(rel_vec_len)*rel_vec/(rel_vec_len+1e-6)*3

        reward += -np.linalg.norm(penlaty_vec)

        distance_reward = (agent.dist - distance_to_goal)
        reward += distance_reward
        agent.dist = distance_to_goal

        reward -= np.arccos(np.clip(agent.vel.dot(agent.last_vel)/(np.linalg.norm(agent.vel)*np.linalg.norm(agent.last_vel)+1e-6),-1,1))

        # reward += np.linalg.norm(agent.vel)-2



        if self.current_step >= self.max_episode_steps:
            done = True
            info['success'] = False
            info['reason'] = 'max_steps'

        info['distance_to_goal'] = distance_to_goal

        return reward, done, info

    def reset(self,a=1000000):
        if True:
            self.duz_reset()
        else:
            if a >= 2000:
                self.reset1()
            else:
                self.reset2()
    def reset1(self):
        invalid = []
        self.goal.reset()
        invalid.append((self.goal.pos,self.goal.radius))
        
        # for agent in self.agents:
        #     while 1:
        #         agent.reset()
        #         for cnt,radius in invalid:
        #             if np.linalg.norm(agent.pos-cnt) < radius+agent.radius+self.landmarks[0].radius+self.goal.radius+self.lsd:
        #                 break
        #         else:
        #             invalid.append((agent.pos,agent.radius))
        #             agent.dist = np.linalg.norm(agent.pos-self.goal.pos)
        #             break

        agent = self.agents[0]
        while 1:
            agent.reset()
            if np.linalg.norm(agent.pos-self.goal.pos) < self.get_radius(1,1,2)+self.lsd:
                continue
            agent.dist = np.linalg.norm(agent.pos-self.goal.pos)
            break
        self.landmarks[0].pos = (self.agents[0].pos + self.goal.pos)/2
        
        agent = self.agents[1]
        while 1:
            agent.reset()
            if np.linalg.norm(agent.pos-self.goal.pos) < self.get_radius(1,1,2)+self.lsd or \
               np.linalg.norm(agent.pos-self.agents[0].pos) < agent.asd+self.get_radius(2,0,1) or \
               np.linalg.norm(agent.pos-self.landmarks[0].pos) < agent.lsd+self.get_radius(1,1,1) or \
               np.linalg.norm((self.agents[1].pos + self.goal.pos)/2-self.agents[0].pos) < agent.lsd+self.get_radius(1,1,1):
                continue
            agent.dist = np.linalg.norm(agent.pos-self.goal.pos)
            break
        
        self.landmarks[1].pos = (self.agents[1].pos + self.goal.pos)/2

    
    def duz_reset(self):
        
        invalid = []
        l = np.random.uniform(-10,10,(2,))
        
        self.agents[0].reset()
        self.agents[0].pos = l + np.array([1.5*self.agents[0].radius,0],dtype=np.float32)
        self.agents[1].reset()
        self.agents[1].pos = l + np.array([-1.5*self.agents[1].radius,0],dtype=np.float32)
        for agent in self.agents:
            agent.dist = np.linalg.norm(agent.pos-self.goal.pos)
            invalid.append((agent.pos,agent.radius))
            
        while 1:
            self.goal.reset()
            for cnt,r in invalid:
                if np.linalg.norm(self.goal.pos-cnt) < self.goal.radius+r:
                    break
            else:
                invalid.append((self.goal.pos,self.goal.radius))
                break
        
        for l in self.landmarks:
            while 1:
                l.reset()
                for cnt,r in invalid:
                    if np.linalg.norm(l.pos-cnt) < l.radius+r:
                        break
                else:
                    invalid.append((l.pos,l.radius))
                    break
        
        

    # def reset(self):
    #     invalid = []
    #     self.goal.reset()
    #     invalid.append((self.goal.pos,self.goal.radius))
        
    #     agent = self.agents[0]
    #     while 1:
    #         agent.reset()
    #         if np.linalg.norm(agent.pos-self.goal.pos) < self.get_radius(5,1,2)+self.lsd:
    #             continue
    #         agent.dist = np.linalg.norm(agent.pos-self.goal.pos)
    #         break
        
    #     agent = self.agents[1]
    #     angle = np.random.uniform(0,2*np.pi)
    #     agent.pos = self.agents[0].pos + np.array([np.cos(angle),np.sin(angle)])*np.random.uniform(self.get_radius("asd"),self.get_radius("acd"))
    #     agent.vel = np.zeros(agent.vel.shape, dtype=np.float32)
    #     agent.dist = np.linalg.norm(agent.pos-self.goal.pos)

        
    #     agent = self.agents[0] if self.agents[0].dist < self.agents[1].dist else self.agents[1]
    #     landmark = self.landmarks[0]
    #     landmark.pos = (self.agents[0].pos + self.goal.pos)/2

    #     random_angle = np.random.uniform(-np.pi/4,np.pi/4)
    #     cos_r,sin_r = np.cos(random_angle),np.sin(random_angle)
    #     self.landmarks[1].pos = np.matmul([[cos_r,-sin_r],[sin_r,cos_r]],(landmark.pos-self.goal.pos))+self.goal.pos

    def reset2(self):
        invalid = []
        self.goal.reset()
        invalid.append((self.goal.pos,self.goal.radius))
        
        agent = self.agents[0]
        while 1:
            agent.reset()
            if np.linalg.norm(agent.pos-self.goal.pos) < self.get_radius(1,1,2)+self.lsd:
                continue
            agent.dist = np.linalg.norm(agent.pos-self.goal.pos)
            break
        
        landmark = self.landmarks[0]
        landmark.pos = (agent.pos + self.goal.pos)/2
        dist = agent.dist/2

        agent = self.agents[1]
        while 1:
            angle = np.random.uniform(0,2*np.pi)
            agent.pos = self.agents[0].pos + np.array([np.cos(angle),np.sin(angle)])*np.random.uniform(self.get_radius("asd"),self.get_radius("acd"))
            if np.linalg.norm(agent.pos-self.goal.pos) < self.agents[0].dist+self.get_radius("lsd"):
                continue
            agent.vel = np.zeros(agent.vel.shape, dtype=np.float32)
            agent.dist = np.linalg.norm(agent.pos-self.goal.pos)
            break
        

        random_rotation_angle = np.random.uniform(-np.pi/3,np.pi/3)
        cos_r,sin_r = np.cos(random_rotation_angle),np.sin(random_rotation_angle)
        self.landmarks[1].pos = np.matmul([[cos_r,-sin_r],[sin_r,cos_r]],(landmark.pos-self.goal.pos)/dist)*np.random.uniform(self.get_radius(0,1.5,1),dist)+self.goal.pos

def main():
    parser = argparse.ArgumentParser(description='MADDPG Navigation Training')
    
    # Environment args
    parser.add_argument('--world_size', type=float, default=10.0, help='World size')
    parser.add_argument('--max_speed', type=float, default=2.0, help='Max agent speed')
    
    # Training args
    parser.add_argument('--num_episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--max_steps_per_episode', type=int, default=200, help='Max steps per episode')
    parser.add_argument('--warmup_episodes', type=int, default=50, help='Warmup episodes before learning')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    
    # Logging args
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=500, help='Model save interval')
    parser.add_argument('--render', action='store_true', help='Render during training')
    parser.add_argument('--render_interval', type=int, default=100, help='Render interval')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Model directory')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help='Model path for testing')
    parser.add_argument('--test_episodes', type=int, default=10, help='Number of test episodes')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    if args.mode == 'train':
        train_agent(args)
    elif args.mode == 'test':
        test_agent(args.model_path, args.test_episodes, render=True)


if __name__ == "__main__":
    # Eğer script olarak çalıştırılırsa
    if len(os.sys.argv) == 1 or "colab" in os.sys.argv[0]:
        # Kullanıcıdan mode seçimi al
        print("="*60)
        print("MADDPG Navigation - Quick Start")
        print("="*60)
        print("1. Train new agent (default)")
        print("2. Test existing agent")
        print("3. Train with visualization")
        print("4. Test with custom episodes")
        print("="*60)
        
        choice = input("Seçiminizi yapın (1-4) [default: 1]: ").strip()
        if not choice:
            choice = "1"
        
        if choice == "1":
            # Default training
            class TrainArgs:
                world_size = 10.0
                max_speed = 2.0
                dt = 0.1
                num_episodes = 10000
                max_steps_per_episode = 200
                warmup_episodes = 50
                batch_size = 64
                hidden_dim = 256
                lr_actor = 1e-4
                lr_critic = 1e-3
                gamma = 0.95
                tau = 0.005
                buffer_size = 100000
                log_interval = 50
                save_interval = 200
                render = False
                render_interval = 100
                log_dir = 'logs'
                model_dir = 'models'
                
            args = TrainArgs()
            print("🚀 Default training başlıyor...")
            print(f"Episodes: {args.num_episodes}, Hidden: {args.hidden_dim}, Visualization: {'ON' if args.render else 'OFF'}")
            train_agent(args)
            
        elif choice == "2":
            # Default testing
            print("🧪 Default testing başlıyor...")
            model_path = "models/best_model.pth"
            test_episodes = 50
            
            # Model var mı kontrol et
            if not os.path.exists(model_path):
                print(f"❌ Model bulunamadı: {model_path}")
                print("Önce training yapmanız gerekiyor!")
                print("Alternatif model paths:")
                if os.path.exists("models"):
                    models = [f for f in os.listdir("models") if f.endswith('.pth')]
                    for model in models:
                        print(f" - models/{model}")
                else:
                    print(" models/ klasörü bulunamadı")
            else:
                print(f"Model: {model_path}, Episodes: {test_episodes}, Visualization: ON")
                test_agent(model_path, test_episodes, render=True)
                
        elif choice == "3":
            # Training with visualization
            class VisualTrainArgs:
                world_size = 10.0
                max_speed = 2.0
                dt = 0.1
                num_episodes = 500 # Daha az episode (görselleştirme için)
                max_steps_per_episode = 200
                warmup_episodes = 25
                batch_size = 64
                hidden_dim = 128
                lr_actor = 1e-4
                lr_critic = 1e-3
                gamma = 0.95
                tau = 0.01
                buffer_size = 100000
                log_interval = 25 # Daha sık log
                save_interval = 100
                render = True # Visualization ON
                render_interval = 50 # Her 50 episode render et
                log_dir = 'logs'
                model_dir = 'models'
                
            args = VisualTrainArgs()
            print("🎬 Training with visualization başlıyor...")
            print(f"Episodes: {args.num_episodes}, Visualization: ON (her {args.render_interval} episode)")
            print("⚠️ Matplotlib penceresi açılacak - kapatmayın!")
            train_agent(args)
            
        elif choice == "4":
            # Custom test
            print("🎯 Custom testing...")
            model_path = input("Model path [default: models/best_model.pth]: ").strip()
            if not model_path:
                model_path = "models/best_model.pth"
                
            test_episodes_input = input("Test episodes [default: 10]: ").strip()
            test_episodes = int(test_episodes_input) if test_episodes_input else 10
            
            render_input = input("Show visualization? (y/n) [default: y]: ").strip().lower()
            render = render_input != 'n'
            
            # Model var mı kontrol et
            if not os.path.exists(model_path):
                print(f"❌ Model bulunamadı: {model_path}")
                if os.path.exists("models"):
                    models = [f for f in os.listdir("models") if f.endswith('.pth')]
                    if models:
                        print("Mevcut modeller:")
                        for i, model in enumerate(models):
                            print(f" {i+1}. models/{model}")
                        choice = input("Hangi modeli kullanmak istiyorsunuz? (1-{}): ".format(len(models)))
                        try:
                            model_idx = int(choice) - 1
                            if 0 <= model_idx < len(models):
                                model_path = f"models/{models[model_idx]}"
                            else:
                                print("Geçersiz seçim!")
                                exit(1)
                        except:
                            print("Geçersiz seçim!")
                            exit(1)
                    else:
                        print("Hiç model bulunamadı!")
                        exit(1)
                else:
                    print("models/ klasörü bulunamadı!")
                    exit(1)
            
            print(f"Model: {model_path}, Episodes: {test_episodes}, Visualization: {'ON' if render else 'OFF'}")
            test_agent(model_path, test_episodes, render=render)
            
        else:
            print("❌ Geçersiz seçim! 1-4 arasında bir sayı girin.")
            
    else:
        print(os.sys.argv)
        raise
        main()