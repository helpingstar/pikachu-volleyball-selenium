import torch.nn as nn
from torch.distributions import Categorical


class Agent(nn.Module):
    def __init__(self, setting):
        super().__init__()
        # Critic Network
        critic_layers = [nn.Linear(35, setting["linear_size"]), nn.ReLU()]
        for _ in range(setting["n_layer"]):
            critic_layers.append(nn.Linear(setting["linear_size"], setting["linear_size"]))
            critic_layers.append(nn.ReLU())
        critic_layers.append(nn.Linear(setting["linear_size"], 1))
        self.critic = nn.Sequential(*critic_layers)

        # Actor Network
        actor_layers = [nn.Linear(35, setting["linear_size"]), nn.ReLU()]
        for _ in range(setting["n_layer"]):
            actor_layers.append(nn.Linear(setting["linear_size"], setting["linear_size"]))
            actor_layers.append(nn.ReLU())
        actor_layers.append(nn.Linear(setting["linear_size"], setting["n_action"]))
        self.actor = nn.Sequential(*actor_layers)

    def get_action(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action
