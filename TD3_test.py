# Library Imports
import gym
import numpy as np
import os
from gym import Env
import random
import torch
import torch.nn as nn
from torch import optim as optim
from torch.nn import functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from collections import deque


class Critic(nn.Module):
    """Defines a Critic Deep Learning Network"""

    def __init__(self, input_dim: int, beta: float = None, density: int = 256, name: str = 'critic'):
        super(Critic, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name

        # Architecture
        self.H1 = torch.nn.Linear(input_dim, density, dtype=torch.float32)
        self.H2 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.H3 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.Q = torch.nn.Linear(density, 1, dtype=torch.float32)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = device
        self.to(device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        value = torch.cat((state, action),dim = 1)
        value = F.relu(self.H1(value))
        value = F.relu(self.H2(value))
        value = F.relu(self.H3(value))
        value = self.Q(value)
        return value


class Actor(nn.Module):
    """Defines a Actor Deep Learning Network"""

    def __init__(self, input_dim: int, n_actions: int, alpha: float = None, density: int = 256, name='actor'):
        super(Actor, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name

        # Architecture
        self.H1 = torch.nn.Linear(input_dim, density, dtype=torch.float32)
        self.H2 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.mu = torch.nn.Linear(density, n_actions, dtype=torch.float32)

        self.optimizer = torch.optim.Adam(self.parameters(),lr=alpha)
        self.device = device
        self.to(device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action = F.relu(self.H1(state))
        action = F.relu(self.H2(action))
        action = torch.tanh(self.mu(action))*2

        return action

class TD3:
    def __init__(self,env,n_games: int = 10, training: bool = True,
                 alpha=0.0005, beta=0.0005, gamma=0.99, tau=0.004,
                 batch_size: int = 128, noise: str = 'normal'):
        """Initialize an Agent object.
        Params
        =====
            alpha & beta :learning rate
            n_actions(int):dimension of each action
            obs_shape(int):dimension of each state
            n_games(int): the total amount of iterations
            gamma(float):discount rate
            buffer_len(int):the length of Replaybuffer
            seed(int):random seed to evaluate the training effect
            per_alpha(int),per_beta(int):per_alpha and per_beta are hypeparameters to represent
            how often PER(prioritiy experience replay) is used to train the model. In general, alpha corresponds with beta
            tau (float) : the update degree of target_network and original_network
        """
        self.directory = r'D:\WHD\model_dict'
        self.gamma = torch.tensor(gamma, dtype=torch.float32, device=device)
        self.tau = tau
        self.n_actions: int = env.action_space.shape[0]
        self.obs_shape: int = env.observation_space.shape[0]
        self.n_games = n_games
        self.gamma = gamma
        self.is_training = training
        self.memory = deque(maxlen=100_000)

        self.batch_size = batch_size
        self.noise = noise
        self.noise_clip = 1.5
        self.max_action = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)
        self.load = False
        self.policy_delay = 3

        self.actor = Actor(self.obs_shape, self.n_actions, alpha, name='actor').to(device)
        self.actor_target = Actor(self.obs_shape,self.n_actions,alpha,name='actor_target').to(device)

        self.critic_1 = Critic(self.obs_shape+self.n_actions,beta).to(device)
        self.critic_1_target = Critic(self.obs_shape+self.n_actions,beta).to(device)
        self.critic_2 = Critic(self.obs_shape+self.n_actions,beta).to(device)
        self.critic_2_target = Critic(self.obs_shape+self.n_actions,beta).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def _add_exploration_noise(self, action: torch.Tensor) -> torch.Tensor:
        mean = torch.zeros_like(action)
        std = torch.ones_like(action)
        noise = torch.normal(mean,std).clamp(-self.noise_clip,self.noise_clip).to("cuda")

        return noise
    def sample(self,batch_size):

        memory_for_training = random.sample(self.memory,batch_size)

        return memory_for_training

    def choose_action(self, observation: torch.tensor) -> np.ndarray:
        self.actor.eval()

        state = torch.as_tensor(observation, dtype=torch.float32, device=device)
        action = self.actor.forward(state)
        action = torch.as_tensor(action,dtype=torch.float32,device=device)
        if self.is_training:
            action += self._add_exploration_noise(action)
        return action.detach().cpu().numpy()

    def train(self):
        gamma = self.gamma
        replayer_buffer = self.memory
        batch_size = self.batch_size
        #Sample replay buffer
        for i in range(self.n_games):
            memory_total = np.array(replayer_buffer,dtype='object')
            idx_for_training = random.sample(range(len(memory_total)),batch_size)
            memory_for_training = memory_total[idx_for_training]

            state = torch.tensor(np.vstack(memory_for_training[:, 0]), dtype=torch.float32).to(device)

            # the shape of states are (length of memory,1)
            action = torch.tensor(np.vstack(memory_for_training[:, 1]), dtype=torch.float32).to(device)
            next_state = torch.tensor(np.vstack(memory_for_training[:,2]),dtype=torch.float32).to("cuda")
            rewards = torch.tensor(np.vstack(memory_for_training[:, 3]), dtype=torch.float32).to(device)
            done = torch.tensor(np.vstack(memory_for_training[:, 4]), dtype=torch.float32).to(device)
            #Select next action through target_actor network
            noise = self._add_exploration_noise(action)
            next_action = self.actor_target(next_state)+ noise
            next_action = next_action.clamp(-self.max_action,self.max_action)
            #compute target q_value：
            target_q1 = self.critic_1_target(next_state,next_action)
            target_q2 = self.critic_2_target(next_state,next_action)
            target_q = torch.min(target_q1,target_q2)
            target_q = rewards + ((1 - done) * gamma * target_q).detach()
            # Optimize Critic 1:
            current_q1 = self.critic_1(state,action)
            loss_q1 = F.mse_loss(current_q1,target_q)
            self.critic_1.optimizer.zero_grad()
            loss_q1.backward()
            self.critic_1.optimizer.step()
            #self.writer.add_scalar('Loss/Q1_loss', loss_q1, global_step=self.num_critic_update_iteration)
            # Optimize Critic 2:
            current_q2 = self.critic_2(state,action)
            loss_q2 = F.mse_loss(current_q2,target_q)

            self.critic_2.zero_grad()
            loss_q2.backward()
            self.critic_2.optimizer.step()

            #Delayed policy updates:
            if (i+1) % self.policy_delay == 0:
                #Compute actor loss:
                actor_loss = - self.critic_1(state,self.actor(state)).mean()

                #Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                #self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(),self.actor_target.parameters()):
                    target_param.data.copy_((1 - self.tau)* target_param.data +self.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_((1-self.tau) * target_param.data + self.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(),self.critic_2_target.parameters()):
                    target_param.data.copy_((1-self.tau) * target_param + self.tau*param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1
        self.save_models()
        print("No. {} training episode".format(self.num_training))
    def save_models(self):
        torch.save(self.actor.state_dict(),self.directory + 'actor.pth')
        torch.save(self.actor_target.state_dict(),self.directory+'actor_target.pth')
        torch.save(self.critic_1.state_dict(),self.directory + 'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(),self.directory + 'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(),self.directory + 'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(),self.directory + 'critic_2_target.pth')
        self.load = True
        print("==================================================")
        print("Model has been saved...")
        print("==================================================")

    def load_models(self):
        self.actor.load_state_dict(torch.load(self.directory + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(self.directory + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(self.directory + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(self.directory + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(self.directory + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(self.directory + 'critic_2_target.pth'))
        print("==================================================")
        print("Model has been loaded...")
        print("==================================================")

if __name__ == '__main__':
    env = gym.make('Pendulum-v1', render_mode="human")
    agent = TD3(env)
    agent.load_models()


    #Test the training effect
    agent.load_models()
    for i in range(10):
        state = env.reset()[0]
        for t in range(1000):
            action = agent.choose_action(state)
            next_state,reward,done,_,_ = env.step(action)
            env.render()
            state = next_state


