import chess
import numpy as np

class ChessEnvironment:
    """
    Handles all chess game logic and state management.
    """
    def __init__(self):
        self.board = chess.Board()
    
    def reset(self):
        """Reset to starting position"""
        self.board = chess.Board()
        return self.get_canonical_board()
    
    def get_legal_moves(self):
        """
        Returns all legal moves in current position.
        Used to mask illegal moves in the policy output.
        """
        return list(self.board.legal_moves)
    
    def execute_move(self, move):
        """
        Execute a move and return new state.
        Move should be in UCI format (e.g., 'e2e4')
        """
        self.board.push(move)
        return self.get_canonical_board()
    
    def get_game_ended(self):
        """
        Returns game outcome from current player's perspective:
        0 if not ended, 1 if current player won, -1 if lost, 0.0001 for draw
        """
        if not self.board.is_game_over():
            return 0
        
        result = self.board.result()
        if result == "1/2-1/2":
            return 0.0001  # Draw
        elif (result == "1-0" and self.board.turn == chess.WHITE) or \
             (result == "0-1" and self.board.turn == chess.BLACK):
            return -1  # Current player lost
        else:
            return 1  # Current player won
    
    def get_canonical_board(self):
        """
        Returns board from current player's perspective.
        This is crucial: the network always sees the board as if it's
        playing white, which makes learning symmetric.
        """
        board_state = self.board_to_planes()
        if self.board.turn == chess.BLACK:
            # Flip board vertically and swap piece colors
            board_state = self.flip_perspective(board_state)
        return board_state
    
    def flip_perspective(self, planes):
        """
        Flip board representation for black's perspective.
        
        What this does:
        - Flips board vertically (rank 0 <-> rank 7)
        - Swaps white and black piece planes (planes 0-5 <-> planes 6-11)
        - Swaps castling rights planes
        - Inverts color plane
        """
        flipped = np.zeros_like(planes)
        
        # For each time step in history (if using history)
        for t in range(8):  # 8 historical positions
            offset = t * 14  # 14 planes per position
            
            # Flip board vertically for all piece planes
            for i in range(12):  # 12 piece planes (6 for each side)
                if i < 6:
                    # White pieces become black pieces
                    flipped[offset + i + 6] = np.flip(planes[offset + i], axis=0)
                else:
                    # Black pieces become white pieces
                    flipped[offset + i - 6] = np.flip(planes[offset + i], axis=0)
        
        # Handle auxiliary planes (last 7 planes: castling, color, counts)
        # Castling rights: swap white and black (planes 112-115)
        flipped[112] = planes[114]  # White kingside <- Black kingside
        flipped[113] = planes[115]  # White queenside <- Black queenside
        flipped[114] = planes[112]  # Black kingside <- White kingside
        flipped[115] = planes[113]  # Black queenside <- White queenside
        
        # Color plane (116): invert (0 -> 1, 1 -> 0)
        flipped[116] = 1.0 - planes[116]
        
        # Move count and no-progress count stay the same
        flipped[117] = planes[117]
        flipped[118] = planes[118]
        
        return flipped

    def board_to_planes(self):
        """
        Convert board to neural network input format.
        Based on Table S1 from paper: 119 planes for chess
        
        Structure:
        - 6 planes for each piece type (P,N,B,R,Q,K) for current player
        - 6 planes for opponent pieces
        - 2 planes for castling rights (kingside/queenside) for current player
        - 2 planes for opponent castling rights
        - 1 plane for en passant squares
        - 1 plane for color (all 1s if white to move, all 0s if black)
        - 1 plane for move count
        - 1 plane for no-progress count (50-move rule)
        
        This is repeated for T=8 historical positions (8 * 14 = 112 planes)
        + 7 auxiliary planes = 119 total
        """
        planes = np.zeros((119, 8, 8), dtype=np.float32)
        
        # Current position encoding (first 14 planes)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                plane_idx = self.piece_to_plane(piece)
                planes[plane_idx, rank, file] = 1.0
        
        # Castling rights (planes 112-115)
        planes[112, :, :] = float(self.board.has_kingside_castling_rights(chess.WHITE))
        planes[113, :, :] = float(self.board.has_queenside_castling_rights(chess.WHITE))
        planes[114, :, :] = float(self.board.has_kingside_castling_rights(chess.BLACK))
        planes[115, :, :] = float(self.board.has_queenside_castling_rights(chess.BLACK))
        
        # Color plane (plane 116)
        planes[116, :, :] = float(self.board.turn == chess.WHITE)
        
        # Move count (plane 117)
        planes[117, :, :] = self.board.fullmove_number / 100.0  # Normalize
        
        # No-progress count for 50-move rule (plane 118)
        planes[118, :, :] = self.board.halfmove_clock / 100.0  # Normalize
        
        return planes
    
    def piece_to_plane(self, piece):
        """Map chess piece to plane index"""
        piece_to_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        base_idx = piece_to_idx[piece.piece_type]
        # Add 6 if it's opponent's piece
        if piece.color != self.board.turn:
            base_idx += 6
        return base_idx
    
class ActionEncoder:
    """
    Encodes/decodes between chess moves and neural network action indices.
    Paper describes: 8x8x73 = 4,672 possible actions
    
    Breakdown of 73 planes:
    - 56 planes: Queen-like moves (7 distances x 8 directions)
    - 8 planes: Knight moves
    - 9 planes: Underpromotions (3 directions x 3 piece types)
    """
    def __init__(self):
        self.action_size = 4672  # 8 * 8 * 73
        self.build_action_mapping()
    
    def build_action_mapping(self):
        """
        Create bidirectional mapping between moves and action indices.
        This is computationally expensive but done once at initialization.
        """
        self.move_to_index = {}
        self.index_to_move = {}
        
        idx = 0
        for from_square in chess.SQUARES:
            from_rank, from_file = divmod(from_square, 8)
            
            # Queen moves (56 planes)
            for direction in range(8):  # N, NE, E, SE, S, SW, W, NW
                for distance in range(1, 8):
                    to_rank, to_file = self.apply_direction(
                        from_rank, from_file, direction, distance
                    )
                    if 0 <= to_rank < 8 and 0 <= to_file < 8:
                        to_square = to_rank * 8 + to_file
                        move = chess.Move(from_square, to_square)
                        
                        plane = direction * 7 + (distance - 1)
                        action_idx = from_square * 73 + plane
                        
                        self.move_to_index[move.uci()] = action_idx
                        self.index_to_move[action_idx] = move
            
            # Knight moves (8 planes)
            knight_moves = [
                (-2, -1), (-2, 1), (-1, -2), (-1, 2),
                (1, -2), (1, 2), (2, -1), (2, 1)
            ]
            for knight_idx, (dr, df) in enumerate(knight_moves):
                to_rank, to_file = from_rank + dr, from_file + df
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    to_square = to_rank * 8 + to_file
                    move = chess.Move(from_square, to_square)
                    
                    plane = 56 + knight_idx
                    action_idx = from_square * 73 + plane
                    
                    self.move_to_index[move.uci()] = action_idx
                    self.index_to_move[action_idx] = move
            
            # Underpromotions (9 planes)
            # Handle pawn promotions to N, B, R from 7th rank
            if from_rank == 6:  # 7th rank (0-indexed)
                for direction in [-1, 0, 1]:  # left diagonal, straight, right diagonal
                    to_file = from_file + direction
                    if 0 <= to_file < 8:
                        to_square = 7 * 8 + to_file  # 8th rank
                        for promo_idx, promo_piece in enumerate([
                            chess.KNIGHT, chess.BISHOP, chess.ROOK
                        ]):
                            move = chess.Move(
                                from_square, to_square, promotion=promo_piece
                            )
                            plane = 64 + direction + 1 + promo_idx * 3
                            action_idx = from_square * 73 + plane
                            
                            self.move_to_index[move.uci()] = action_idx
                            self.index_to_move[action_idx] = move
    
    def apply_direction(self, rank, file, direction, distance):
        """Apply a direction vector for queen-like moves"""
        direction_map = {
            0: (1, 0),   # N
            1: (1, 1),   # NE
            2: (0, 1),   # E
            3: (-1, 1),  # SE
            4: (-1, 0),  # S
            5: (-1, -1), # SW
            6: (0, -1),  # W
            7: (1, -1)   # NW
        }
        dr, df = direction_map[direction]
        return rank + dr * distance, file + df * distance
    
    def encode_move(self, move):
        """Convert chess.Move to action index"""
        return self.move_to_index.get(move.uci(), -1)
    
    def decode_action(self, action_idx):
        """Convert action index to chess.Move"""
        return self.index_to_move.get(action_idx, None)
    
    def get_legal_actions_mask(self, board):
        """
        Create binary mask of legal actions for current position.
        This is critical: illegal moves must have probability 0.
        """
        mask = np.zeros(self.action_size, dtype=np.float32)
        for move in board.legal_moves:
            action_idx = self.encode_move(move)
            if action_idx >= 0:
                mask[action_idx] = 1.0
        return mask