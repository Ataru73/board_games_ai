import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class PolicyValueNet(nn.Module):
    def __init__(self, board_size=7, num_channels=12, num_res_blocks=10, num_filters=128):
        """
        Upgraded Architecture for 'Strong Amateur' level play.
        - num_res_blocks: Increased to 10 (was 4) for deeper calculation.
        - num_filters: Increased to 128 (was 64) for better pattern recognition.
        """
        super(PolicyValueNet, self).__init__()
        self.board_size = board_size
        
        # Initial Block
        self.conv_input = nn.Conv2d(num_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        # Residual Blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Policy Head
        # Reduces filters -> 2 channels
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        # Input size: 2 * 7 * 7 = 98
        self.policy_fc = nn.Linear(2 * board_size * board_size, 196) # 196 actions
        
        # Value Head
        # Reduces filters -> 1 channel
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        # Fully connected: Board Area (49) -> 128 -> 1
        self.value_fc1 = nn.Linear(1 * board_size * board_size, num_filters)
        self.value_fc2 = nn.Linear(num_filters, 1)

    def forward(self, x):
        # Input: (N, 12, 7, 7) - 4 stacked states * 3 channels each

        # Normalize input
        # Standardize dice values (1-6) to roughly 0-1 range
        x = x / 6.0
        
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        for block in self.res_blocks:
            x = block(x)
            
        # Policy Head
        x_policy = F.relu(self.policy_bn(self.policy_conv(x)))
        x_policy = x_policy.view(-1, 2 * self.board_size * self.board_size)
        x_policy = self.policy_fc(x_policy)
        policy = F.log_softmax(x_policy, dim=1)
        
        # Value Head
        x_value = F.relu(self.value_bn(self.value_conv(x)))
        x_value = x_value.view(-1, 1 * self.board_size * self.board_size)
        x_value = F.relu(self.value_fc1(x_value))
        value = torch.tanh(self.value_fc2(x_value))
        
        return policy, value