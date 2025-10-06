import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Standard residual block with two conv layers.
    Used throughout the network for deep feature extraction.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out

class AlphaZeroNetwork(nn.Module):
    """
    Neural network (p, v) = f_θ(s) from the paper.
    
    Takes board state s as input, outputs:
    - p: policy vector (probability distribution over moves)
    - v: value scalar (expected outcome from this position)
    
    Architecture:
    - Initial convolution
    - N residual blocks (paper uses 20 for Go, we'll use 19 for chess)
    - Policy head: outputs move probabilities
    - Value head: outputs position evaluation
    """
    def __init__(self, num_res_blocks=19, num_channels=256):
        super().__init__()
        
        # Initial convolution: 119 input planes -> num_channels
        self.conv_initial = nn.Conv2d(119, num_channels, 3, padding=1, bias=False)
        self.bn_initial = nn.BatchNorm2d(num_channels)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 73, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(73)
        # Output is 8x8x73 = 4672 logits
        
        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 119, 8, 8) board representation
        
        Returns:
            policy_logits: (batch_size, 4672) unnormalized log probabilities
            value: (batch_size, 1) position evaluation in [-1, 1]
        """
        # Initial convolution
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        
        # Residual tower
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_logits = policy.view(-1, 4672)  # Flatten to action space
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 64)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output in [-1, 1]
        
        return policy_logits, value
    
    def predict(self, board_state):
        """
        Single inference for MCTS.
        Returns policy probabilities and value.
        """
        self.eval()
        with torch.no_grad():
            board_tensor = torch.FloatTensor(board_state).unsqueeze(0)
            if torch.cuda.is_available():
                board_tensor = board_tensor.cuda()
            
            policy_logits, value = self.forward(board_tensor)
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value.cpu().numpy()[0][0]
        
        return policy_probs, value