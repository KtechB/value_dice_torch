import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
def weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(kaiming_uniform_(m.weight.data))
        m.bias.data.fill_(0)

class DenseNet(nn.Module):
    def __init__(self,dim_in, dim_out, unit_size = [10,10]):
        super(DenseNet, self).__init__()
        self.features = make_layers( input_dim =dim_in, unit_size = unit_size[:-1] )
        self.ouput_layer = nn.Linear(unit_size[-1], dim_out)
        self.features.apply(wight_init)
        self.output_layer.apply(weight_init)

    def forward(self, x):
        h = self.features(x)
        y = output_layer(h)
        return y


def make_layers(input_dim , unit_size):
    layers = []
    input_layer = nn.Linear(input_dim, unit_size[0])
    layers += [input_layer, nn.relu()]
    if len(unit_size) > 1
    for i in range(len(unit_size) - 1):
        linear = nn.Linear(unit_size[i], unit_size[1])
        layers += [linear, nn.relu()]
    
    return nn.Sequential(*layers)


def train(agent_traj, expert_traj, expert_init_s, pol, vf, optim_vf, optim_pol,batch_size = 32, step = 4, aplha = 0.1, gamma = 0.99):
    agent_iterator = agent_traj.iterate_step(
        batch_size=batch_size, step=step)
    expert_iterator = expert_traj.iterate_step(
        batch_size=batch_size, step=step)
    for agent_batch, expert_batch in zip(agent_iterator, expert_iterator):
        bellman_e = non_reward_bellman(pol, vf, expert_batch, gamma, sampling=1)
        bellman_a = non_reward_bellman(pol, vf, agent_batch, gamma, sampling=1)
        

        a_real, ac, pd_params = pol(expert_init_s_batch)
        
        v_s0 = vf(expert_init_s_batch, a_real)

        loss_log =torch.log(torch.sum((1-alpha) * torch.exp(bellman_e) + alpha * torch.exp(bellman_a), axis = 0))
        loss_linear = torch.sum((1-alpha) * (1-gamma) *  v_s0 + alpha * (bellman_a), axis = 0)

        loss =  loss_log - loss_linear
        optim_pol.zero_grad()
        optim_vf.zero_grad()
        loss.backward()
        optim_pol.step()
        optim_vf.step()
    
    
def non_reward_bellman(pol, vf, batch, gamma, sampling = 1):
    """
    Bellman loss with no reward
    
    ------------------------
    samplling:int  
        number of sampling action per one nextobs

    returns
    ----------
    bellman_loss: torch.Tensor

    """
    obs = batch["obs"]
    acs = batch["acs"]
    next_obs = batch["next_obs"]
    dones = batch["dones"]

    __, __, pd_params = pol(next_obs)
    pd = pol.pd


    next_acs = pd.sample(pd_params, torch.Size([sampling]))
    next_obs = next_obs.expand([sampling] + list(next_obs.size()))

    bellman_loss = vf(obs, acs) - 
                    gamma * vf(next_obs, next_abs) * (1 - dones)

        

    return bellman_loss