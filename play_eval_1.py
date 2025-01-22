import torch
import torch.nn as nn
import chess
import numpy as np
from collections import defaultdict
import time

import torch
import torch.nn as nn

class ChessModel(nn.Module):
    def __init__(self, num_moves):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Added max pooling
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Added max pooling
        )
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 512),  # Adjusted for downscaling
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_moves)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        policy = self.policy_head(x)
        return torch.log_softmax(policy, dim=1), None
import torch
import chess
import numpy as np
import time

class ChessEngine:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        if model_path:
            checkpoint = torch.load(model_path, map_location=device)
            self.num_moves = len(checkpoint['move_mapping'])
            self.model = ChessModel(self.num_moves)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(device)
            self.model.eval()
            
            self.move_mapping = checkpoint['move_mapping']
            self.reverse_mapping = {v: k for k, v in self.move_mapping.items()}
        else:
            self.num_moves = 1968
            self.model = None
            self.move_mapping = {}
            self.reverse_mapping = {}

        # Piece values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

    def board_to_tensor(self, board):
        pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
        board_state = np.zeros((8, 8, 13), dtype=np.float32)
        
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                rank, file = i // 8, i % 8
                piece_idx = pieces.index(piece.symbol())
                board_state[rank, file, piece_idx] = 1
        
        board_state[:, :, -1] = float(board.turn)
        
        return torch.FloatTensor(board_state).permute(2, 0, 1).unsqueeze(0)

    def evaluate_position(self, board):
        """Simple material evaluation"""
        if board.is_checkmate():
            return -20000 if board.turn else 20000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0
        piece_map = board.piece_map()
        
        # Material count only
        for square, piece in piece_map.items():
            value = self.piece_values[piece.piece_type]
            if piece.color:
                score += value
            else:
                score -= value

        return score if board.turn else -score

    def alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board)

        if maximizing_player:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    def get_model_predictions(self, board, temperature=1.0):
        with torch.no_grad():
            tensor = self.board_to_tensor(board).to(self.device)
            policy_logits, _ = self.model(tensor)
            
            # Apply temperature scaling
            scaled_logits = policy_logits / temperature
            probabilities = torch.softmax(scaled_logits, dim=-1).cpu().numpy()[0]
        return probabilities


    def get_best_move(self, board, depth=5, time_limit=10, num_model_moves=5, debug_output=True):

        start_time = time.time()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        if debug_output:
            print("\nDEBUG - Getting model predictions...")
        predictions = self.get_model_predictions(board)
        
        # Get top moves according to the model
        move_scores = []
        if debug_output:
            print("\nDEBUG - Scoring legal moves:")
            
        for move in legal_moves:
            move_str = move.uci()
            if move_str in self.move_mapping:
                move_idx = self.move_mapping[move_str]
                prediction = predictions[move_idx]
                move_scores.append((move, prediction))
                if debug_output:
                    print(f"Move: {board.san(move)} ({move_str}) - Model score: {prediction:.4f}")
            else:
                move_scores.append((move, 0.0001))
                if debug_output:
                    print(f"Move: {board.san(move)} ({move_str}) - Not in model vocabulary")
        
        # Sort moves by model score
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        if debug_output:
            print(f"\nDEBUG - Top {num_model_moves} moves according to model:")
            for i, (move, score) in enumerate(move_scores[:num_model_moves]):
                print(f"{i+1}. {board.san(move)} - Score: {score:.4f}")
        
        # Select top N moves to analyze
        top_moves = move_scores[:num_model_moves] if len(move_scores) > num_model_moves else move_scores
        
        # Search with alpha-beta
        best_move = None
        best_eval = float('-inf')
        
        if debug_output:
            print("\nDEBUG - Searching moves with alpha-beta:")
            
        for move, model_score in top_moves:
            # Check time limit
            if (time.time() - start_time) >= time_limit:
                if debug_output:
                    print(f"Time limit ({time_limit}s) reached, stopping search.")
                break
            
            # Evaluate position
            board.push(move)
            eval = -self.alpha_beta(board, depth-1, float('-inf'), float('inf'), False)
            board.pop()
            
            if debug_output:
                print(f"Move: {board.san(move)} - Model score: {model_score:.4f} - Eval: {eval/100:.2f}")
            
            # Update best move if better evaluation found
            if eval > best_eval:
                best_eval = eval
                best_move = move
                if debug_output:
                    print(f"New best move: {board.san(best_move)} with eval {best_eval/100:.2f}")
        
        return best_move

def play_game(engine, human_color=chess.WHITE):
    board = chess.Board()
    
    def print_board():
        print("\n   a b c d e f g h")
        print("  ----------------")
        board_str = str(board).split('\n')
        for i, row in enumerate(board_str):
            print(f"{8-i} |{row}| {8-i}")
        print("  ----------------")
        print("   a b c d e f g h\n")
    
    while not board.is_game_over():
        print_board()
        print(f"\nEvaluation: {engine.evaluate_position(board)/100:.2f}")
        
        if board.turn == human_color:
            while True:
                try:
                    move_str = input("\nEnter your move (e.g., 'e2e4' or 'Nf3' or 'O-O'): ").strip()
                    
                    if move_str.upper() in ['O-O', 'O-O-O']:
                        moves = list(board.legal_moves)
                        castle_move = None
                        for move in moves:
                            if board.is_castling(move):
                                if (move_str.upper() == 'O-O' and chess.square_file(move.to_square) == 6) or \
                                   (move_str.upper() == 'O-O-O' and chess.square_file(move.to_square) == 2):
                                    castle_move = move
                                    break
                        if castle_move:
                            board.push(castle_move)
                            break
                        else:
                            print("Illegal castling move!")
                            continue
                    
                    try:
                        move = chess.Move.from_uci(move_str)
                        if move in board.legal_moves:
                            board.push(move)
                            break
                        else:
                            raise ValueError("Illegal move")
                    except ValueError:
                        try:
                            move = board.parse_san(move_str)
                            board.push(move)
                            break
                        except ValueError:
                            print("Invalid move! Please use UCI (e.g., 'e2e4') or algebraic notation (e.g., 'Nf3')")
                            continue
                except Exception as e:
                    print(f"Error: {str(e)}")
                    print("Please try again.")
        else:
            print("\nEngine is thinking...")
            move = engine.get_best_move(board)
            if move:
                print(f"Engine plays: {board.san(move)} ({move.uci()})")
                board.push(move)
            else:
                print("Engine couldn't find a move!")
                break

    print("\nGame Over!")
    print(f"Result: {board.result()}")
    print_board()
    
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        print(f"\nCheckmate! {winner} wins!")
    elif board.is_stalemate():
        print("\nStalemate! Game is drawn.")
    elif board.is_insufficient_material():
        print("\nDraw due to insufficient material.")
    elif board.is_fifty_moves():
        print("\nDraw due to fifty-move rule.")
    elif board.is_repetition():
        print("\nDraw due to threefold repetition.")
    else:
        print("\nGame drawn.")

def main():
    print("Chess Engine Starting...")
    print("\nAttempting to load trained model...")
    
    try:
        engine = ChessEngine('chess_model_v2.pth')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Could not load model: {str(e)}")
        print("Continuing without neural network support...")
        engine = ChessEngine()

    while True:
        print("\nMenu:")
        print("1. Play as White")
        print("2. Play as Black")
        print("3. Quit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            play_game(engine, chess.WHITE)
        elif choice == '2':
            play_game(engine, chess.BLACK)
        elif choice == '3':
            print("\nThanks for playing!")
            break
        else:
            print("\nInvalid choice! Please try again.")

if __name__ == "__main__":
    main()