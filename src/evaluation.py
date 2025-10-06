import chess
import numpy as np
from representation import ActionEncoder, ChessEnvironment
from mcts import MCTS

class Evaluator:
    """
    Evaluates AlphaZero against opponents.
    Paper evaluates against Stockfish with specific settings.
    
    For training: evaluate against previous version of self
    For benchmarking: evaluate against known engines
    """
    def __init__(self, network, opponent_network=None, games_per_eval=100):
        self.network = network
        self.opponent_network = opponent_network
        self.games_per_eval = games_per_eval
        self.action_encoder = ActionEncoder()
    
    def play_match(self, use_mcts_for_both=True, mcts_simulations=800):
        """
        Play evaluation match between current network and opponent.
        
        Returns:
            results: Dict with wins, losses, draws
        """
        results = {'wins': 0, 'losses': 0, 'draws': 0}
        
        for game_num in range(self.games_per_eval):
            # Alternate colors
            player1_is_white = (game_num % 2 == 0)
            
            result = self.play_single_game(
                player1_is_white,
                use_mcts_for_both,
                mcts_simulations
            )
            
            # Update results from player1 (our network) perspective
            if result == 1:
                results['wins'] += 1
            elif result == -1:
                results['losses'] += 1
            else:
                results['draws'] += 1
            
            if (game_num + 1) % 10 == 0:
                print(f"Evaluation: {game_num + 1}/{self.games_per_eval} games complete")
        
        return results
    
    def play_single_game(self, player1_is_white, use_mcts, mcts_simulations):
        """
        Play one game between networks.
        
        Returns:
            1 if player1 (current network) won
            -1 if player1 lost
            0 if draw
        """
        env = ChessEnvironment()
        env.reset()
        
        while True:
            # Determine which network to use
            if (env.board.turn == chess.WHITE) == player1_is_white:
                # Player 1's turn (current network)
                network = self.network
            else:
                # Player 2's turn (opponent)
                network = self.opponent_network if self.opponent_network else self.network
            
            # Get move
            if use_mcts:
                mcts = MCTS(network, num_simulations=mcts_simulations)
                search_probs = mcts.search(env, add_noise=False)
                move = self.get_best_move(env, search_probs)
            else:
                move = self.get_network_move(env, network)
            
            # Execute move
            env.execute_move(move)
            
            # Check if game ended
            game_result = env.get_game_ended()
            if game_result != 0:
                # Convert result to player1's perspective
                if not player1_is_white:
                    game_result = -game_result
                
                if game_result == 0.0001:
                    return 0  # Draw
                elif game_result > 0:
                    return 1  # Player1 won
                else:
                    return -1  # Player1 lost
    
    def get_best_move(self, env, search_probs):
        """Select best move from search probabilities (greedy)"""
        legal_moves = env.get_legal_moves()
        best_prob = -1
        best_move = None
        
        for move in legal_moves:
            action_idx = self.action_encoder.encode_move(move)
            if action_idx >= 0 and search_probs[action_idx] > best_prob:
                best_prob = search_probs[action_idx]
                best_move = move
        
        return best_move
    
    def get_network_move(self, env, network):
        """Get move directly from network policy (no MCTS)"""
        canonical_board = env.get_canonical_board()
        policy_probs, _ = network.predict(canonical_board)
        
        # Mask illegal moves
        legal_mask = self.action_encoder.get_legal_actions_mask(env.board)
        policy_probs = policy_probs * legal_mask
        policy_probs /= np.sum(policy_probs)
        
        # Get best legal move
        return self.get_best_move(env, policy_probs)
    
    def calculate_elo(self, wins, losses, draws, baseline_elo=3000):
        """
        Calculate Elo rating based on win/loss/draw record.
        Paper uses this to track improvement during training.
        
        Args:
            wins, losses, draws: Match results
            baseline_elo: Opponent's Elo rating
        
        Returns:
            Estimated Elo rating
        """
        total_games = wins + losses + draws
        if total_games == 0:
            return baseline_elo
        
        # Calculate win percentage (draws count as 0.5)
        score = (wins + 0.5 * draws) / total_games
        
        # Avoid edge cases
        score = np.clip(score, 0.01, 0.99)
        
        # Elo formula
        elo_diff = 400 * np.log10(score / (1 - score))
        return baseline_elo + elo_diff