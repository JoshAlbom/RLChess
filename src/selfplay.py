import chess
import numpy as np
from collections import deque
from representation import ActionEncoder, ChessEnvironment
from mcts import MCTS

class SelfPlayGame:
    """
    Generates a single self-play game for training.
    
    Process:
    1. Start from initial position
    2. At each move, run MCTS to get search probabilities π
    3. Sample move from π (with temperature)
    4. Store (state, π, placeholder_value) tuple
    5. When game ends, fill in actual outcomes
    """
    def __init__(self, network, mcts_simulations=800):
        self.network = network
        self.mcts_simulations = mcts_simulations
        self.action_encoder = ActionEncoder()
    
    def play_game(self, temperature=1.0):
        """
        Play one complete self-play game.
        
        Args:
            temperature: Controls exploration in move selection
                        Higher = more exploration (early game)
                        Lower = more exploitation (late game)
        
        Returns:
            training_examples: List of (state, π, z) tuples where:
                - state: board position (119, 8, 8)
                - π: search probabilities from MCTS (4672,)
                - z: final game outcome from that player's perspective
        """
        training_examples = []
        env = ChessEnvironment()
        env.reset()
        
        move_count = 0
        
        while True:
            move_count += 1
            
            # Get current canonical board state
            canonical_board = env.get_canonical_board()
            
            # Run MCTS to get search probabilities
            mcts = MCTS(self.network, num_simulations=self.mcts_simulations)
            search_probs = mcts.search(env, add_noise=True)
            
            # Store training example (outcome will be filled in later)
            training_examples.append({
                'state': canonical_board,
                'policy': search_probs,
                'player': env.board.turn  # Track which player made this move
            })
            
            # Select move based on search probabilities with temperature
            move = self.select_move_with_temperature(
                env, search_probs, temperature, move_count
            )
            
            # Execute move
            env.execute_move(move)
            
            # Check if game ended
            game_result = env.get_game_ended()
            if game_result != 0:
                # Game over - assign outcomes to all training examples
                return self.assign_outcomes(training_examples, game_result, env)
    
    def select_move_with_temperature(self, env, search_probs, temperature, move_count):
        """
        Select move from search probabilities using temperature.
        
        Temperature schedule from paper:
        - First 30 moves: sample proportionally from π (temp=1)
        - After move 30: select argmax (temp→0)
        
        This provides exploration early, exploitation late.
        """
        legal_moves = env.get_legal_moves()
        
        # Get action indices and probabilities for legal moves
        action_probs = []
        moves_list = []
        
        for move in legal_moves:
            action_idx = self.action_encoder.encode_move(move)
            if action_idx >= 0:
                action_probs.append(search_probs[action_idx])
                moves_list.append(move)
        
        action_probs = np.array(action_probs)
        
        # Apply temperature
        if move_count < 30:
            # Proportional sampling with temperature
            if temperature == 0:
                # Greedy selection
                move_idx = np.argmax(action_probs)
            else:
                # Apply temperature and sample
                action_probs = action_probs ** (1.0 / temperature)
                action_probs /= np.sum(action_probs)
                move_idx = np.random.choice(len(moves_list), p=action_probs)
        else:
            # After move 30, always pick best move
            move_idx = np.argmax(action_probs)
        
        return moves_list[move_idx]
    
    def assign_outcomes(self, training_examples, final_result, env):
        """
        Assign game outcomes to all training examples.
        
        Key insight: Each example gets outcome from that position's
        player perspective. If white won (+1 from white's view),
        then black's examples get -1.
        
        Args:
            training_examples: List of dicts with state, policy, player
            final_result: Game outcome from final position's perspective
            env: Final environment state
        
        Returns:
            List of (state, policy, outcome) tuples ready for training
        """
        final_examples = []
        
        # Determine actual winner
        if final_result == 0.0001:
            # Draw
            outcome_white = 0
            outcome_black = 0
        else:
            # Someone won - figure out who
            # final_result is from the perspective of the player who just moved
            last_player = env.board.turn
            
            if final_result == -1:
                # Last player lost, so previous player won
                if last_player == chess.WHITE:
                    outcome_white = -1
                    outcome_black = 1
                else:
                    outcome_white = 1
                    outcome_black = -1
            else:
                # Last player won (rare - usually means checkmate)
                if last_player == chess.WHITE:
                    outcome_white = 1
                    outcome_black = -1
                else:
                    outcome_white = -1
                    outcome_black = 1
        
        # Assign outcomes to each example based on player
        for example in training_examples:
            if example['player'] == chess.WHITE:
                outcome = outcome_white
            else:
                outcome = outcome_black
            
            final_examples.append((
                example['state'],
                example['policy'],
                outcome
            ))
        
        return final_examples


class SelfPlayWorker:
    """
    Manages parallel self-play game generation.
    Paper uses 5,000 TPUs generating games simultaneously.
    
    For practical implementation, we'll use multiprocessing on CPUs/GPUs.
    """
    def __init__(self, network, games_per_iteration=100, mcts_simulations=800):
        self.network = network
        self.games_per_iteration = games_per_iteration
        self.mcts_simulations = mcts_simulations
        self.replay_buffer = deque(maxlen=500000)  # Store recent games
    
    def generate_games(self):
        """
        Generate a batch of self-play games.
        
        In production: parallelize across multiple processes/machines
        For this implementation: sequential for clarity
        """
        all_examples = []
        
        for game_num in range(self.games_per_iteration):
            print(f"Generating self-play game {game_num + 1}/{self.games_per_iteration}")
            
            game_generator = SelfPlayGame(self.network, self.mcts_simulations)
            examples = game_generator.play_game()
            
            all_examples.extend(examples)
            
            # Add to replay buffer
            for example in examples:
                self.replay_buffer.append(example)
        
        print(f"Generated {len(all_examples)} training examples")
        return all_examples
    
    def get_training_batch(self, batch_size=4096):
        """
        Sample random batch from replay buffer for training.
        Paper uses mini-batch size of 4096.
        """
        if len(self.replay_buffer) < batch_size:
            batch_size = len(self.replay_buffer)
        
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        # Separate into states, policies, outcomes
        states = np.array([example[0] for example in batch])
        policies = np.array([example[1] for example in batch])
        outcomes = np.array([example[2] for example in batch])
        
        return states, policies, outcomes