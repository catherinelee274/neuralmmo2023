# model_1 Vanilla Policy Gradient 
# pylint: disable=all


# pylint: disable=all

# source: look at https://github.com/lbarazza/VPG-PyTorch/blob/master/vpg.py 


# import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym
from collections import deque

# define policy network
class policy_net(nn.Module):
    def __init__(self, nS, nH, nA): # nS: state space size, nH: n. of neurons in hidden layer, nA: size action space
        super(policy_net, self).__init__()
        self.h = nn.Linear(nS, nH)
        self.out = nn.Linear(nH, nA)

    # define forward pass with one hidden layer with ReLU activation and sofmax after output layer
    def forward(self, x):
        x = F.relu(self.h(x))
        x = F.softmax(self.out(x), dim=1)
        return x

# create environment
env = gym.make("CartPole-v1")
# instantiate the policy
policy = policy_net(env.observation_space.shape[0], 20, env.action_space.n)
# create an optimizer
optimizer = torch.optim.Adam(policy.parameters())
# initialize gamma and stats
gamma=0.99
n_episode = 1
returns = deque(maxlen=100)
render_rate = 100 # render every render_rate episodes
while True:
    rewards = [] # TODO: replace
    actions = [] # TODO: replace
    states  = [] # TODO: replace
    # reset environment
    state = env.reset()
    while True:
        # render episode every render_rate epsiodes
        if n_episode%render_rate==0:
            env.render()

        # calculate probabilities of taking each action
        probs = policy(torch.tensor(state).unsqueeze(0).float())
        # sample an action from that set of probs
        sampler = Categorical(probs)
        action = sampler.sample()

        # use that action in the environment
        new_state, reward, done, info = env.step(action.item())
        # store state, action and reward
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = new_state
        if done:
            break

    # preprocess rewards
    rewards = np.array(rewards)
    # calculate rewards to go for less variance
    R = torch.tensor([np.sum(rewards[i:]*(gamma**np.array(range(i, len(rewards))))) for i in range(len(rewards))])
    # or uncomment following line for normal rewards
    #R = torch.sum(torch.tensor(rewards))

    # preprocess states and actions
    states = torch.tensor(states).float()
    actions = torch.tensor(actions)

    # calculate gradient
    probs = policy(states)
    sampler = Categorical(probs)
    log_probs = -sampler.log_prob(actions)   # "-" because it was built to work with gradient descent, but we are using gradient ascent
    pseudo_loss = torch.sum(log_probs * R) # loss that when differentiated with autograd gives the gradient of J(θ)
    # update policy weights
    optimizer.zero_grad()
    pseudo_loss.backward()
    optimizer.step()

    # calculate average return and print it out
    returns.append(np.sum(rewards))
    print("Episode: {:6d}\tAvg. Return: {:6.2f}".format(n_episode, np.mean(returns)))
    n_episode += 1

# close environment
env.close()



### REAL 

@dataclass
class VanillaPolicyGradient:
    env_creator: callable = None
    env_creator_kwargs: dict = None
    agent: nn.Module = None
    agent_creator: callable = None
    agent_kwargs: dict = None

    exp_name: str = os.path.basename(__file__)

    data_dir: str = 'data'
    record_loss: bool = False
    checkpoint_interval: int = 1
    seed: int = 1
    torch_deterministic: bool = True
    vectorization: ... = pufferlib.vectorization.Serial
    device: str = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    num_buffers: int = 1
    num_envs: int = 8
    num_cores: int = psutil.cpu_count(logical=False)
    cpu_offload: bool = True
    verbose: bool = True
    batch_size: int = 2**14
    policy_store: pufferlib.policy_store.PolicyStore = None
    policy_ranker: pufferlib.policy_ranker.PolicyRanker = None

    policy_pool: pufferlib.policy_pool.PolicyPool = None
    policy_selector: pufferlib.policy_ranker.PolicySelector = None

    # Wandb
    wandb_entity: str = None
    wandb_project: str = None
    wandb_extra_data: dict = None

    # Selfplay
    selfplay_learner_weight: float = 1.0
    selfplay_num_policies: int = 1

    def __post_init__(self, *args, **kwargs):
        # TODO: FILL 
        return None
    
    @pufferlib.utils.profile
    def evaluate(self, show_progress=False):
        # TODO: FILL 
        return None
    

    @pufferlib.utils.profile
    def train(
        self,
        batch_rows=32,
        update_epochs=4,
        bptt_horizon=16,
        gamma=0.99,
        gae_lambda=0.95,
        anneal_lr=True,
        norm_adv=True,
        clip_coef=0.1,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
    ):
        # TODO: FILL 
        return None
    
'''
Functions that remain the same across all
''' 

    def done_training(self):
        return self.update >= self.total_updates

    def close(self):
        for envs in self.buffers:
            envs.close()

        if self.wandb_entity:
            wandb.finish()

    def _save_checkpoint(self):
        if self.data_dir is None:
            return

        policy_name = f"{self.exp_name}.{self.update:06d}"
        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "agent_step": self.agent_step,
            "update": self.update,
            "learning_rate": self.learning_rate,
            "policy_checkpoint_name": policy_name,
            "wandb_run_id": self.wandb_run_id,
        }
        path = os.path.join(self.data_dir, f"trainer.pt")
        tmp_path = path + ".tmp"
        torch.save(state, tmp_path)
        os.rename(tmp_path, path)

        # NOTE: as the agent_creator has args internally, the policy args are not passed
        self.policy_store.add_policy(policy_name, self.agent)

        if self.policy_ranker:
            self.policy_ranker.add_policy_copy(
                policy_name, self.policy_pool._learner_name
            )