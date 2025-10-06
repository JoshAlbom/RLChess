import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from representation import ActionEncoder

class AlphaZeroTrainer:
    """
    Trains the AlphaZero network using self-play data.
    
    Loss function from paper (Equation 1):
    l = (z - v)² - π^T log(p) + c||θ||²
    
    Where:
    - (z - v)²: mean squared error between predicted and actual outcome
    - π^T log(p): cross-entropy between search policy and network policy
    - c||θ||²: L2 regularization on weights
    """
    def __init__(self, network, learning_rate=0.2, weight_decay=1e-4):
        self.network = network
        self.optimizer = optim.SGD(
            network.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay  # L2 regularization (c in paper)
        )
        
        # Learning rate schedule: paper drops LR 3 times during training
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[100000, 300000, 500000],  # Training steps
            gamma=0.1  # Multiply LR by 0.1 at each milestone
        )
        
        self.action_encoder = ActionEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
    
    def train_step(self, states, target_policies, target_values):
        """
        Single training step on a batch.
        
        Args:
            states: (batch_size, 119, 8, 8) board states
            target_policies: (batch_size, 4672) search probabilities π
            target_values: (batch_size,) game outcomes z
        
        Returns:
            loss_dict: Dictionary with total loss and components
        """
        self.network.train()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        target_policies = torch.FloatTensor(target_policies).to(self.device)
        target_values = torch.FloatTensor(target_values).to(self.device)
        
        # Forward pass
        policy_logits, predicted_values = self.network(states)
        predicted_values = predicted_values.squeeze(-1)
        
        # Value loss: (z - v)²
        value_loss = F.mse_loss(predicted_values, target_values)
        
        # Policy loss: -π^T log(p)
        # Use log_softmax for numerical stability
        log_policies = F.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.sum(target_policies * log_policies, dim=1).mean()
        
        # Total loss (L2 regularization handled by weight_decay in optimizer)
        total_loss = value_loss + policy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item()
        }
    
    def train_from_buffer(self, replay_buffer, batch_size=4096, epochs=1):
        """
        Train on data from replay buffer.
        
        Args:
            replay_buffer: Deque of (state, policy, outcome) tuples
            batch_size: Mini-batch size (paper uses 4096)
            epochs: Number of passes through data
        """
        buffer_size = len(replay_buffer)
        num_batches = buffer_size // batch_size
        
        total_losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(buffer_size)
            
            for batch_idx in range(num_batches):
                # Get batch indices
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Extract batch
                batch = [replay_buffer[i] for i in batch_indices]
                states = np.array([ex[0] for ex in batch])
                policies = np.array([ex[1] for ex in batch])
                values = np.array([ex[2] for ex in batch])
                
                # Train on batch
                loss_dict = self.train_step(states, policies, values)
                total_losses.append(loss_dict['total_loss'])
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{num_batches}, "
                          f"Loss: {loss_dict['total_loss']:.4f} "
                          f"(Value: {loss_dict['value_loss']:.4f}, "
                          f"Policy: {loss_dict['policy_loss']:.4f})")
        
        return np.mean(total_losses)