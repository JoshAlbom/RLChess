import os
import pickle
import torch
from datetime import datetime
from train import AlphaZeroTrainer
from selfplay import SelfPlayWorker
from evaluation import Evaluator
from agent import AlphaZeroNetwork

class AlphaZeroTrainingPipeline:
    """
    Complete AlphaZero training pipeline.
    
    From paper:
    - Train for 700,000 steps
    - Each step: 4096 examples
    - Generate self-play games continuously
    - Update network from replay buffer
    """
    def __init__(
        self,
        network,
        training_steps=700000,
        games_per_iteration=100,
        mcts_simulations=800,
        batch_size=4096,
        eval_frequency=1000,
        checkpoint_frequency=10000
    ):
        self.network = network
        self.training_steps = training_steps
        self.games_per_iteration = games_per_iteration
        self.mcts_simulations = mcts_simulations
        self.batch_size = batch_size
        self.eval_frequency = eval_frequency
        self.checkpoint_frequency = checkpoint_frequency
        
        # Initialize components
        self.trainer = AlphaZeroTrainer(network)
        self.self_play_worker = SelfPlayWorker(
            network, games_per_iteration, mcts_simulations
        )
        self.evaluator = Evaluator(network)
        
        # Tracking
        self.training_history = {
            'losses': [],
            'elo_ratings': [],
            'eval_results': []
        }
    
    def train(self):
        """
        Main training loop.
        
        Process:
        1. Generate self-play games
        2. Train network on replay buffer
        3. Periodically evaluate
        4. Save checkpoints
        """
        print("=" * 60)
        print("Starting AlphaZero Training")
        print(f"Total training steps: {self.training_steps}")
        print(f"Batch size: {self.batch_size}")
        print(f"MCTS simulations: {self.mcts_simulations}")
        print("=" * 60)
        
        step = 0
        
        while step < self.training_steps:
            print(f"\n{'='*60}")
            print(f"Training Step {step}/{self.training_steps}")
            print(f"{'='*60}")
            
            # Phase 1: Generate self-play games
            print("\n[1/3] Generating self-play games...")
            self.self_play_worker.generate_games()
            
            # Phase 2: Train on replay buffer
            print("\n[2/3] Training network...")
            if len(self.self_play_worker.replay_buffer) >= self.batch_size:
                avg_loss = self.trainer.train_from_buffer(
                    self.self_play_worker.replay_buffer,
                    batch_size=self.batch_size,
                    epochs=1
                )
                self.training_history['losses'].append(avg_loss)
                print(f"Average loss: {avg_loss:.4f}")
            
            step += 1
            
            # Phase 3: Evaluate periodically
            if step % self.eval_frequency == 0:
                print("\n[3/3] Running evaluation...")
                self.evaluate_current_network()
            
            # Save checkpoint
            if step % self.checkpoint_frequency == 0:
                self.save_checkpoint(step)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        self.save_checkpoint(step, final=True)
    
    def evaluate_current_network(self):
        """
        Evaluate current network strength.
        In production: compare against previous best or Stockfish
        For training: track improvement over time
        """
        print("Running evaluation matches...")
        results = self.evaluator.play_match(
            use_mcts_for_both=True,
            mcts_simulations=400  # Faster for eval
        )
        
        print(f"Results: {results['wins']}W - {results['losses']}L - {results['draws']}D")
        
        # Calculate approximate Elo
        elo = self.evaluator.calculate_elo(
            results['wins'],
            results['losses'],
            results['draws']
        )
        print(f"Estimated Elo: {elo:.0f}")
        
        self.training_history['eval_results'].append(results)
        self.training_history['elo_ratings'].append(elo)
    
    def save_checkpoint(self, step, final=False):
        """Save model checkpoint and training history"""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "final" if final else f"step_{step}"
        
        # Save model
        model_path = os.path.join(checkpoint_dir, f"{prefix}_model_{timestamp}.pt")
        torch.save({
            'step': step,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'training_history': self.training_history
        }, model_path)
        
        print(f"Saved checkpoint: {model_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        print(f"Loaded checkpoint from step {checkpoint['step']}")
        return checkpoint['step']


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"PyTorch version: {torch.__version__}")
    # Initialize network
    print("Initializing AlphaZero network...")
    network = AlphaZeroNetwork(num_res_blocks=19, num_channels=256)
    
    # Create training pipeline
    pipeline = AlphaZeroTrainingPipeline(
        network=network,
        training_steps=700000,      # Paper uses 700k
        games_per_iteration=10,     # Adjust based on compute (orginal at 100)
        mcts_simulations=50,        # Paper uses 800
        batch_size=512,             # Paper uses 4096
        eval_frequency=100,         # Evaluate every 1000 steps
        checkpoint_frequency=10000   # Save every 10k steps
    )
    
    # Start training
    pipeline.train()