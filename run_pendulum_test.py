# -*- coding: utf-8 -*-
"""
@Time : 2023/6/14 20:05
@FILE : run_pendulum_test.py
@Author : WTC
"""
import gym
import numpy as np
import time
import torch.nn.functional as F
import torch.nn as nn
import torch
def run_test():

    env = gym.make('Pendulum-v1',render_mode = "human")
    action = [0]


    observation = env.reset()  # 状态
    actions = np.linspace(-2, 2, 10)
    for t in range(100):  #
        # action[0] =  random.uniform(-2,2)   #力矩  -2到2
        action[0] = 2
        observation, reward, done, _,_ = env.step(action)
        print(action, reward, done)

        # print('observation:',observation)
        # print('theta:',env.state)

        env.render()
        time.sleep(1)
        print("No.{}",t)
    env.close()
class Test():
    def __init__(self):
        self.test = test(10,1)
        self.directory = r'E:\WHD\model_dict'
    def save(self):
        torch.save(self.test.state_dict(), self.directory+'test.pth')

        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.test.load_state_dict(torch.load(self.directory + 'test.pth'))

        print("====================================")
        print("model has been loaded...")
        print("====================================")
class test(nn.Module):
    def __init__(self,input_dim = 10,output_dim = 1):
        super(test, self).__init__()

        self.fc1 = nn.Linear(input_dim,output_dim)
    def forward(self,state):
        a = F.relu(self.fc1(state))


if __name__ == '__main__':
    a = Test()
    model = torch.load(a.directory+'test.pth',torch.device('cpu'))
    print(len(model))
    print(type(model))
    for k in model.keys():
        print(k)
