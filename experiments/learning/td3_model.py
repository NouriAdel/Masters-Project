# Theory: https://spinningup.openai.com/en/latest/algorithms/td3.html
# Reference Implementation: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/DDPG.ipynb
# Good implementation: https://github.com/henry32144/TD3-Pytorch
import numpy as np
import random
import copy
from collections import namedtuple, deque

from neural_network_AC_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.995           # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 0.1e-3       # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 1e-6     # L2 weight decay

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    # Shared crtic among all agents
    critic_local = None
    critic_target = None
    critic_optimizer = None
    
    
    def __init__(self, state_size, action_size, 
        random_seed, agent_size=1, max_action = 1, 
        min_action = -1, noise=0.2, noise_std=0.3, noise_clip=0.5):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            max_action (ndarray): the maximum valid value for each action vector
            min_action (ndarray): the minimum valid value for each action vector
            random_seed (int): random seed
            noise (float): the range to generate random noise while learning
            noise_std (float): the range to generate random noise while performing action
            noise_clip (float): to clip random noise into this range
        """

        self.state_size = state_size
        self.action_size = action_size
        self.agent_size = agent_size
        self.max_action = max_action
        self.min_action = min_action
        self.noise = noise
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        if Agent.critic_local is None:
            Agent.critic_local = Critic(state_size, action_size, random_seed).to(device)
        if Agent.critic_target is None:
            Agent.critic_target = Critic(state_size, action_size, random_seed).to(device)
            Agent.critic_target.load_state_dict(self.critic_local.state_dict())
        if Agent.critic_optimizer is None:
            Agent.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        self.critic_local = Agent.critic_local
        self.critic_target = Agent.critic_target 
        self.critic_optimizer = Agent.critic_optimizer

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
                          
    
    def step(self, states, actions, rewards, next_states, dones, 
        t, learn_every, n_experiences):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
              
        if self.agent_size > 1:
            for agent in range(self.agent_size):
                self.memory.add(states[agent,:], actions[agent,:], 
                    rewards[agent], next_states[agent,:], dones[agent])
        else:
            self.memory.add(states, actions, rewards, next_states, dones)
        

        if t % learn_every == 0:
            for i in range(n_experiences): #n times to update the network
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)





    def act(self, state, noise_reduction=0.0, add_noise=True):
        """Returns actions for given state as per current policy."""
        
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=self.action_size)
            noise *= noise_reduction
            actions += noise
            actions = np.clip(actions, self.min_action, self.max_action)
        self.actor_local.train()
        return  actions  
                                                                                                                                                         

    def learn(self, experiences, gamma=GAMMA):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            
            
        """
        # Sample replay buffer                                       
        states, actions, rewards, next_states, dones = experiences
        action_ = actions.cpu().numpy()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        
        actions_next = self.actor_target(next_states)

        noise = torch.FloatTensor(action_).data.normal_(0, self.noise).to(device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        
        actions_next = (actions_next + noise).clamp(-self.max_action, self.max_action)
        Q1_targets, Q2_targets = self.critic_target(next_states, actions_next)
        Q_targets_next = torch.min(Q1_targets, Q2_targets)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)).detach()
        # Compute critic loss
        
        
        
        Q1_expected, Q2_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q1_expected, Q_targets) + F.mse_loss(Q2_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #use gradient clipping when training the critic network from Benchmark Implementation  Attemp 3
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local.Q1(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor_local(state).cpu().data.numpy().flatten()

    def load(self, i):
        self.actor_local.load_state_dict(torch.load('td3_actor{}.pth'.format(i), map_location=torch.device('cpu')))
        self.critic_local.load_state_dict(torch.load('td3_critic{}.pth'.format(i), map_location=torch.device('cpu')))

        
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
