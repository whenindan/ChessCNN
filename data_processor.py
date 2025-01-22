import chess
import chess.pgn
import json
import numpy as np

class ChessDataProcessor:
    def __init__(self, max_moves=60):
        self.max_moves = max_moves
        # Piece to plane mapping (6 pieces * 2 colors + 1 empty = 13 planes)
        self.piece_to_plane = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        
    def board_to_tensor(self, board):
        """Convert chess board to 8x8x13 tensor."""
        tensor = np.zeros((8, 8, 13), dtype=np.float32)
        
        # Fill piece planes
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank, file = chess.square_rank(square), chess.square_file(square)
                plane = self.piece_to_plane[piece.symbol()]
                tensor[rank, file, plane] = 1
                
        # Fill auxiliary plane (empty squares)
        tensor[:, :, 12] = 1 - np.sum(tensor[:, :, :12], axis=2)
        
        return tensor
        
    def process_game(self, game, player_name):

        sequences = []
        current_sequence = []
        
        board = game.board()
        moves_list = list(game.mainline_moves())
        
        is_player_white = (game.headers["White"] == player_name)
        player_color = chess.WHITE if is_player_white else chess.BLACK
        
        for move_idx, move in enumerate(moves_list):
            if len(current_sequence) >= self.max_moves:
                if len(current_sequence) > 0:
                    sequences.append(current_sequence)
                current_sequence = []
                
            # Only process positions where it's the player's turn
            if board.turn == player_color:
                position_tensor = self.board_to_tensor(board)
                
                # Get context features
                move_number = board.fullmove_number
                castling_rights = [
                    board.has_kingside_castling_rights(chess.WHITE),
                    board.has_queenside_castling_rights(chess.WHITE),
                    board.has_kingside_castling_rights(chess.BLACK),
                    board.has_queenside_castling_rights(chess.BLACK)
                ]
                
                # Add position data
                current_sequence.append({
                    'position': position_tensor,
                    'move_number': move_number,
                    'castling_rights': castling_rights,
                    'next_move': move.uci(),
                    'game_id': game.headers.get("Site", ""),
                    'player_elo': game.headers.get(f"{game.headers['White']} if is_player_white else {game.headers['Black']}Elo", "")
                })
            
            board.push(move)
            
        # Add the last sequence if it's not empty
        if len(current_sequence) > 0:
            sequences.append(current_sequence)
            
        return sequences
    
    def process_pgn_file(self, pgn_file, player_name):
        all_sequences = []
        
        with open(pgn_file, "r") as file:
            while True:
                game = chess.pgn.read_game(file)
                if game is None:
                    break
                    
                # Only process games where the player participated
                if (game.headers.get("White") == player_name or 
                    game.headers.get("Black") == player_name):
                    game_sequences = self.process_game(game, player_name)
                    all_sequences.extend(game_sequences)
        
        return all_sequences
    
    def save_to_json(self, data, output_file):

        # Convert numpy arrays to lists for JSON serialization
        serializable_data = []
        for sequence in data:
            serializable_sequence = []
            for position_data in sequence:
                position_data = position_data.copy()
                position_data['position'] = position_data['position'].tolist()
                serializable_sequence.append(position_data)
            serializable_data.append(serializable_sequence)
            
        with open(output_file, "w") as file:
            json.dump(serializable_data, file, indent=4)

if __name__ == "__main__":
    processor = ChessDataProcessor()
    pgn_file_path = "brainlet888.pgn"
    output_file_path = "brainlet888_sequences.json"
    player_name = "brainlet888"
    
    # Process the PGN file
    sequences = processor.process_pgn_file(pgn_file_path, player_name)
    
    # Save to JSON
    processor.save_to_json(sequences, output_file_path)
    print(f"Processed {len(sequences)} sequences for player '{player_name}'")