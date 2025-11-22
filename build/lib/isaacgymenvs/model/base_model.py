"""
TODO: clean up later, a lot of functions are not used here
"""

from abc import ABC, abstractmethod
import torch.nn as nn
import torch


class BaseModel(nn.Module, ABC):
    
    @abstractmethod
    def __init__(
        self, 
        normalize_state=False, 
        normalize_action=False,
        action_std=None,
        action_space="delta",
    ):
        super().__init__()
        self.is_normalize_state = normalize_state
        self.is_normalize_action = normalize_action
        self.action_std = action_std
        self.action_space = action_space
        if self.action_space == "delta" or self.action_space == "relative":
            if self.is_normalize_action:
                assert self.action_std is not None, \
                    "action_std must be provided if normalize_action is True and action_space is delta or relative"
                self.action_std = torch.tensor(action_std).float()
        
    @abstractmethod
    def forward_pass(self, obs, target):
        pass
    
    @abstractmethod
    def get_action(self, obs):
        pass

    def normalize_action(self, actions):
        if self.is_normalize_action:
            actions = actions / self.action_std.to(actions.device)
        return actions

    def decode_actions(self, current_angles, actions):
        """
        Decode actions based on the action space.
        
        Args:
            current_angles (torch.Tensor): Current joint angles of the robot, shape (B, 1, 7)
            actions (torch.Tensor): Predicted actions from the model, shape (B, S, 7)
                where B is batch size, S is sequence length
                
        Returns:
            torch.Tensor: Decoded actions based on the action space setting
        """
        # unnormalize actions if normalized
        if self.is_normalize_action:
            if self.action_space == "delta" or self.action_space == "relative":
                actions = actions * self.action_std.to(actions.device)
        # decode actions
        if self.action_space == "delta":
            abs_actions = torch.zeros_like(actions)
            abs_actions[:, 0] = current_angles.squeeze(1) + actions[:, 0]
            for i in range(1, actions.shape[1]):
                abs_actions[:, i] = abs_actions[:, i-1] + actions[:, i]
        elif self.action_space == "relative":
            abs_actions = current_angles + actions
        elif self.action_space == "absolute":
            abs_actions = actions
        
        return abs_actions
