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

class Sumtree(object):
    def __init__(self, capacity,alpha,beta):
        """To initialize a priority buffer by sumtree.
        Parameters
        ----------
        capacity (int): the leaf amount of sumtree
        tree (int): the total amount of sumtree
        state(ndarray): np.array to store state
        reward(ndarray): np.array to store reward
        action(ndarray): np.array to store action
        next_state(ndarray): np.array to store next_state
        done(ndarray): np.array to store whether the game is terminated
        full_sign(int): a binary value to represent whether the memory is full
        idx(int): the index of the value in memory
        alpha & beta(float): hypeparameters to adjust the training effect

        """

        self.capacity = capacity  # capacity是叶子节点个数，
        self.tree = np.zeros(2 * capacity)  # 从1开始编号[1,capacity]
        self.state = np.zeros((capacity,3))
        self.action = np.zeros(capacity)
        self.next_state = np.zeros((capacity,3))
        self.reward = np.zeros(capacity)
        self.done = np.zeros(capacity)
        self.alpha = alpha
        self.beta = beta
        self.full_sign = 0
        self.idx = 0
    def add(self, data,state,action,next_state,reward,done):
        #here data represents the priority and it is equal to td error
        if self.idx >= self.capacity:
            self.idx = self.idx % self.capacity
        idx = self.idx
        self.state[idx] = state.cpu().data
        self.action[idx] = action.cpu().data
        self.next_state[idx] = next_state.cpu().data
        self.reward[idx] = reward
        self.done[idx] = done
        pos = self.idx + self.capacity
        self._updatetree(pos,data)
        self.idx += 1
        print(self.idx)
        if self.idx == self.capacity:
            self.full_sign = 1
    def _updatetree(self,pos,data):
        #idx is the index in array
        #pos is the index in SumTree
        change = data - self.tree[pos]
        self.tree[pos] = data
        self._propagate(change,pos)
    def _propagate(self,change,pos):
        pos = pos // 2
        self.tree[pos] += change  # 更新父节点的值，是向上传播的体现
        if pos != 1:
            self._propagate(change,pos)
    def _total(self):
        return self.tree[1]
    def get(self, s):
        idx = self._retrieve(1, s)
        return idx
    def _retrieve(self, idx, s):
        left = 2 * idx
        right = left + 1
        if left >= self.capacity-1:
            if self.tree[left] >= s:
                return left
            else:
                return right
        if s <= self.tree[left]:
            return self._retrieve(left, s)  # 往左孩子处查找
        else:
            return self._retrieve(right, s - self.tree[left])  # 往右孩子处查找
    def update_priority(self,priority):
        for idxx , pri in zip(self.indices,priority):
            self._updatetree(idxx+self.capacity,pri)

    def sample(self,batch_size):
        """Sample a batch of experiences from memory based on their priority."""
        assert self.beta > 0
        if self.full_sign == 1:
            prios = self.tree[self.capacity:2 * self.capacity]
        else:
            prios = self.tree[self.capacity:self.idx+self.capacity+1]
        probs = prios ** self.alpha
        probs /= probs.sum()
        every_range_len = max(self.tree[1] // batch_size,1)
        indices = []
        for i in range(batch_size):
            indices.append(int(i * every_range_len + int(random.random() * every_range_len)))
        indices = [self._retrieve(1,value) for value in indices ]
        indices = np.array(indices)
        self.indices = indices
        experiences = [self.tree[idx] for idx in indices]
        total = self.capacity if self.full_sign == 1 else self.idx
        indices -= self.capacity
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights,dtype=np.float32)
        states = torch.from_numpy(np.vstack(self.state[indices])).float().to(device)
        actions = torch.from_numpy(np.vstack(self.action[indices])).float().to(device)
        rewards = torch.from_numpy(np.vstack(self.reward[indices])).float().to(device)
        next_states = torch.from_numpy(np.vstack(self.next_state[indices])).float().to(device)
        dones = torch.from_numpy(np.vstack(self.done[indices])).float().to(device)
        indices = torch.from_numpy(np.vstack([index for index in indices]).astype(np.uint8)).int().to(device)
        weights = torch.from_numpy(np.vstack([weight for weight in weights]).astype(np.float32)).float().to(device)
        return (states,actions,next_states,rewards,dones,indices,weights)

    def __len__(self):
        if self.full_sign == 1:
            return self.idx
        else:
            return self.capacity


class Critic(nn.Module):
    """Defines a Critic Deep Learning Network"""

    def __init__(self, input_dim: int, beta: float = None, density: int = 512, name: str = 'critic'):
        super(Critic, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name

        # Architecture
        self.H1 = torch.nn.Linear(input_dim, density, dtype=torch.float32)
        self.H2 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.drop = torch.nn.Dropout(p=0.1)
        self.H3 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.H4 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.Q = torch.nn.Linear(density, 1, dtype=torch.float32)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = device
        self.to(device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        value = torch.cat((state, action),dim = 1)
        value = F.relu(self.H1(value))
        value = F.relu(self.H2(value))
        value = self.drop(value)
        value = F.relu(self.H3(value))
        value = F.relu(self.H4(value))
        value = self.Q(value)
        return value

class Actor(nn.Module):
    """Defines a Actor Deep Learning Network"""

    def __init__(self, input_dim: int, n_actions: int, alpha: float = None, density: int = 512, name='actor'):
        super(Actor, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name

        # Architecture
        self.H1 = torch.nn.Linear(input_dim, density, dtype=torch.float32)
        self.H2 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.drop = torch.nn.Dropout(p=0.1)
        self.H3 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.H4 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.mu = torch.nn.Linear(density, n_actions, dtype=torch.float32)

        self.device = device
        self.to(device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action = F.relu(self.H1(state))
        action = F.relu(self.H2(action))
        action = self.drop(action)
        action = F.relu(self.H3(action))
        action = F.relu(self.H4(action))
        action = torch.tanh(self.mu(action))*2
        return action

class TD3:
    def __init__(self,env,n_games: int = 300, training: bool = True,
                 alpha=0.0001, beta=0.002, gamma=0.99, tau=0.004,
                 batch_size: int = 64, noise: str = 'normal',
                 per_alpha: float = 0.6, per_beta: float = 0.4,
                 ):
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
        self.directory = r'E:\WHD\model_dict'
        self.gamma = torch.tensor(gamma, dtype=torch.float32, device=device)
        self.tau = tau
        self.n_actions: int = env.action_space.shape[0]
        self.obs_shape: int = env.observation_space.shape[0]
        self.n_games = n_games
        self.optim_steps: int = 0
        self.buffer_len: int = 16_384
        self.gamma = gamma
        self.is_training = training
        self.memory = Sumtree(self.buffer_len,per_alpha,per_beta)
        self.batch_size = batch_size
        self.noise = noise
        self.noise_clip = 1.5
        self.max_action = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)
        self.load =False
        self.policy_delay = 3

        self.actor = Actor(self.obs_shape, self.n_actions, alpha, name='actor').to(device)
        self.actor_target = Actor(self.obs_shape,self.n_actions,alpha,name='actor_target').to(device)

        self.critic_1 = Critic(self.obs_shape+self.n_actions,beta).to(device)
        self.critic_1_target = Critic(self.obs_shape+self.n_actions,beta).to(device)
        self.critic_2 = Critic(self.obs_shape+self.n_actions,beta).to(device)
        self.critic_2_target = Critic(self.obs_shape+self.n_actions,beta).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr = alpha)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(),lr = beta)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(),lr = beta)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())


        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def _add_exploration_noise(self, action: torch.Tensor) -> torch.Tensor:
        mean = torch.zeros_like(action)
        std = torch.ones_like(action)
        noise = torch.normal(mean,std).clamp(-self.noise_clip,self.noise_clip)

        return noise

    def choose_action(self, observation: np.ndarray) -> np.ndarray:
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
            state,action,next_state,reward,done,indice,weight = replayer_buffer.sample(batch_size)
            """state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(done).to(device)
            reward = torch.FloatTensor(reward).to(device)"""


            #Select next action through target_actor network
            noise = self._add_exploration_noise(action)
            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.max_action,self.max_action)

            #compute target q_value：
            target_q1 = self.critic_1_target(next_state,next_action)
            target_q2 = self.critic_2_target(next_state,next_action)
            target_q = torch.min(target_q1,target_q2)
            target_q = reward + ((1 - done) * gamma * target_q).detach()


            # Optimize Critic 1:
            current_q1 = self.critic_1(state,action)
            loss_q1 = F.mse_loss(current_q1,target_q)
            self.critic_1_optimizer.zero_grad()
            loss_q1.backward()
            self.critic_1_optimizer.step()
            #self.writer.add_scalar('Loss/Q1_loss', loss_q1, global_step=self.num_critic_update_iteration)

            # Optimize Critic 2:
            current_q2 = self.critic_2(state,action)
            loss_q2 = F.mse_loss(current_q2,target_q)
            self.critic_2_optimizer.zero_grad()
            loss_q2.backward()
            self.critic_2_optimizer.step()
            #self.writer.add_scalar('Loss/Q2_loss', loss_q2, global_step=self.num_critic_update_iteration)


            td_error = torch.add(abs(target_q - current_q1),abs(target_q-current_q2))
            td_error = torch.div(td_error,2)
            self.memory.update_priority(td_error)
            #Delayed policy updates:
            if (i+1) % self.policy_delay == 0:
                #Compute actor loss:
                actor_loss = - self.critic_1(state,self.actor(state)).mean()

                #Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
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

    if agent.load:
        agent.load_models()
    #Warm up stage and dont update
    state = env.reset()[0]
    state = torch.tensor(state,dtype=torch.float32)
    for t in range(1000):

        action = agent.choose_action(state)
        noise = np.random.normal(0,np.random.rand(),size=len(action))
        action = action + noise
        action = action.clip(env.action_space.low,env.action_space.high)
        next_state,reward,done,_,_ = env.step(action)
        next_action = agent.choose_action(next_state)

        state = torch.as_tensor(state,dtype=torch.float32,device=device).unsqueeze(0)
        action = torch.as_tensor(action,dtype=torch.float32,device=device).unsqueeze(0)
        next_state = torch.as_tensor(next_state,dtype=torch.float32,device=device).unsqueeze(0)
        next_action =torch.as_tensor(next_action,dtype=torch.float32,device=device).unsqueeze(0)

        q_state_1 = agent.critic_1.forward(state,action)
        q_state_2 = agent.critic_2(state,action)
        q_next_state_1 = agent.critic_1_target(next_state,next_action)
        q_next_state_2 = agent.critic_2_target(next_state,next_action)
        td_target = reward + (1-done) * min(q_next_state_1,q_next_state_2)
        td_error_1 = q_state_1 - td_target
        td_error_2 = q_state_2 - td_target
        td_error = (td_error_1+ td_error_2)/2

        state = state.squeeze(0)
        action = action.squeeze(0)
        td_error = td_error.squeeze(0)
        next_state = next_state.squeeze(0)

        agent.memory.add(td_error,state,action,next_state,reward,done)
        state = next_state

        if (t+1) % 10 == 0:
            print('No. {} Warm up episode'.format(t+1))

    #Start training
    for i in range(500):
        state = env.reset()[0]
        for t in range(1000):
            action = agent.choose_action(state)
            noise = np.random.normal(0, np.random.rand(), size=len(action))
            action = action + noise
            action = action.clip(env.action_space.low, env.action_space.high)
            next_state, reward, done, _, _ = env.step(action)
            next_action = agent.choose_action(next_state)

            state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            next_action = torch.as_tensor(next_action, dtype=torch.float32, device=device).unsqueeze(0)

            q_state_1 = agent.critic_1.forward(state, action)
            q_state_2 = agent.critic_2(state, action)
            q_next_state_1 = agent.critic_1_target(next_state, next_action)
            q_next_state_2 = agent.critic_2_target(next_state, next_action)
            td_target = reward + (1 - done) * min(q_next_state_1, q_next_state_2)
            td_error_1 = q_state_1 - td_target
            td_error_2 = q_state_2 - td_target
            td_error = (td_error_1 + td_error_2) / 2

            state = state.squeeze(0)
            action = action.squeeze(0)
            td_error = td_error.squeeze(0)
            next_state = next_state.squeeze(0)

            agent.memory.add(td_error, state, action, next_state, reward, done)
            state = next_state
        if agent.memory.full_sign == 1 or agent.memory.idx >= 1000:
            agent.train()
        print(agent.memory.idx)
        agent.save_models()