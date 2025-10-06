import math
import numpy as np
from collections import defaultdict
from representation import ActionEncoder, ChessEnvironment

class MCTSNode:
    """
    Represents a single node in the MCTS tree.
    Each node corresponds to a board state.
    """
    def __init__(self, prior_prob):
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_prob = prior_prob  # P(s,a) from neural network
        self.children = {}  # Maps actions to child nodes
    
    def value(self):
        """Q(s,a): average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def is_leaf(self):
        """Check if this is a leaf node (not yet expanded)"""
        return len(self.children) == 0

class MCTS:
    """
    Monte Carlo Tree Search as described in AlphaZero paper.
    
    Key differences from traditional MCTS:
    - Uses neural network for evaluation instead of rollouts
    - Uses neural network policy to guide selection
    - No random playouts
    """
    def __init__(self, network, num_simulations=800, c_puct=1.0):
        """
        Args:
            network: AlphaZeroNetwork for evaluation
            num_simulations: Number of MCTS simulations per move (paper uses 800)
            c_puct: Exploration constant for PUCT formula
        """
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.root = None
    
    def copy_env(self, env):
        """
        Create a copy of environment for simulation.
        We need this because MCTS simulations modify the board state.
        """
        env_copy = ChessEnvironment()
        env_copy.board = env.board.copy()
        return env_copy
    
    def search(self, env, add_noise=True):
        """
        Run MCTS from current position.
        
        Returns:
            π: search probabilities over actions (policy target for training)
        """
        # Initialize root if needed
        root_state = env.get_canonical_board()
        self.root = MCTSNode(prior_prob=1.0)
        
        # Add Dirichlet noise to root for exploration (during training)
        if add_noise:
            self.add_exploration_noise(env)
        
        # Run simulations
        for _ in range(self.num_simulations):
            env_copy = self.copy_env(env)
            self._simulate(env_copy, self.root)
        
        # Extract visit counts as policy
        action_visits = np.zeros(4672)
        total_visits = sum(child.visit_count for child in self.root.children.values())
        
        for action, child in self.root.children.items():
            action_visits[action] = child.visit_count
        
        # Normalize to get probabilities
        search_probs = action_visits / total_visits if total_visits > 0 else action_visits
        
        return search_probs
    
    def _simulate(self, env, node):
        """
        Single MCTS simulation from node to leaf.
        
        The paper describes this as:
        1. Select actions using PUCT formula until reaching leaf
        2. Expand leaf using neural network
        3. Backup values up the tree
        """
        # Check if game ended
        game_result = env.get_game_ended()
        if game_result != 0:
            # Terminal node: return actual game outcome
            return -game_result  # Negated for opponent's perspective
        
        # If leaf node, expand it
        if node.is_leaf():
            return self._expand_and_evaluate(env, node)
        
        # Select best action using PUCT
        action_idx = self._select_action(env, node)
        action_encoder = ActionEncoder()
        move = action_encoder.decode_action(action_idx)
        
        if move is None:
            # Shouldn't happen, but handle gracefully
            return 0

        # Execute move
        env.execute_move(move)
        
        # Recurse on child
        if action_idx not in node.children:
            node.children[action_idx] = MCTSNode(prior_prob=0.0)

        value = self._simulate(env, node.children[action_idx])
        
        # Backup: update node statistics
        node.visit_count += 1
        node.total_value += value
        
        return -value  # Negate for alternating players
    
    def _expand_and_evaluate(self, env, node):
        """
        Expand leaf node using neural network evaluation.
        
        From paper: Neural network f_θ(s) returns:
        - Prior probabilities p for actions
        - Value estimate v for position
        """
        board_state = env.get_canonical_board()
        policy_probs, value = self.network.predict(board_state)
        
        # Mask illegal moves
        action_encoder = ActionEncoder()
        legal_mask = action_encoder.get_legal_actions_mask(env)
        policy_probs = policy_probs * legal_mask
        policy_sum = np.sum(policy_probs)
        if policy_sum > 0:
            policy_probs /= policy_sum  # Renormalize
        
        # Create child nodes for all legal actions
        legal_actions = env.get_legal_moves()
        
        
        for move in legal_actions:
            action_idx = action_encoder.encode_move(move)
            if action_idx >= 0:
                node.children[action_idx] = MCTSNode(
                    prior_prob=policy_probs[action_idx]
                )
        
        return value
    
    def _select_action(self, env, node):
        """
        Select action with highest PUCT value.
        
        PUCT formula from paper:
        a = argmax_a (Q(s,a) + U(s,a))
        where U(s,a) = c_puct * P(s,a) * sqrt(Σ_b N(s,b)) / (1 + N(s,a))
        
        This balances exploitation (Q) and exploration (U).
        """
        best_score = -float('inf')
        best_action = None
        
        sqrt_total_visits = math.sqrt(sum(
            child.visit_count for child in node.children.values()
        ))
        
        for action, child in node.children.items():
            # Q(s,a): average value
            q_value = child.value()
            
            # U(s,a): exploration bonus
            u_value = self.c_puct * child.prior_prob * sqrt_total_visits / (1 + child.visit_count)
            
            # PUCT score
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def add_exploration_noise(self, env):
        """
        Add Dirichlet noise to root node prior probabilities.
        From paper: Dir(α) where α = 0.15 for chess
        
        This ensures exploration during self-play training.
        """
        legal_actions = env.get_legal_moves()
        action_encoder = ActionEncoder()
        
        # Generate Dirichlet noise
        alpha = 0.15  # Paper specifies 0.15 for chess
        noise = np.random.dirichlet([alpha] * len(legal_actions))
        
        # Mix noise with prior (75% prior, 25% noise)
        epsilon = 0.25
        for i, move in enumerate(legal_actions):
            action_idx = action_encoder.encode_move(move)
            if action_idx in self.root.children:
                child = self.root.children[action_idx]
                child.prior_prob = (1 - epsilon) * child.prior_prob + epsilon * noise[i]