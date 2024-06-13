import torch.nn as nn
from torch.distributions import Categorical
import torch


class Agent(nn.Module):
    def __init__(self, setting):
        super().__init__()
        self.setting = setting
        # Critic Network
        critic_layers = [nn.Linear(35, setting["n_linear"]), nn.ReLU()]
        for _ in range(setting["n_layer"]):
            critic_layers.append(nn.Linear(setting["n_linear"], setting["n_linear"]))
            critic_layers.append(nn.ReLU())
        critic_layers.append(nn.Linear(setting["n_linear"], 1))
        self.critic = nn.Sequential(*critic_layers)

        # Actor Network
        actor_layers = [nn.Linear(35, setting["n_linear"]), nn.ReLU()]
        for _ in range(setting["n_layer"]):
            actor_layers.append(nn.Linear(setting["n_linear"], setting["n_linear"]))
            actor_layers.append(nn.ReLU())
        actor_layers.append(nn.Linear(setting["n_linear"], setting["n_action"]))
        self.actor = nn.Sequential(*actor_layers)

    def get_action(self, x):
        logits = self.actor(x)
        if self.setting["infer"] == "argmax":
            action = torch.argmax(logits)
        else:
            probs = Categorical(logits=logits)
            action = probs.sample()
        return action
