import chess
import chess.pgn
import json

def process_pgn_file(pgn_file, player_name):

    data = []

    with open(pgn_file, "r") as file:
        while True:
            game = chess.pgn.read_game(file)
            if game is None:
                break

            # Get player names
            white_player = game.headers.get("White", "")
            black_player = game.headers.get("Black", "")

            # Determine if the player is involved
            if white_player == player_name or black_player == player_name:
                board = game.board()
                for move in game.mainline_moves():
                    board_state = board.fen()  # Current board state as FEN
                    board.push(move)  # Make the move on the board

                    # Check if this move is made by the selected player
                    current_player = white_player if board.turn == chess.BLACK else black_player
                    if current_player == player_name:
                        data.append({
                            "board_state": board_state,
                            "next_move": move.uci()  # Store move in UCI format
                        })

    return data

def save_to_json(data, output_file):

    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    # Replace with your PGN file path
    pgn_file_path = "brainlet888.pgn"
    output_file_path = "brainlet888_data.json"

    # Filter by player name
    player_name = "brainlet888"

    # Process the PGN file
    extracted_data = process_pgn_file(pgn_file_path, player_name)

    # Save to JSON
    save_to_json(extracted_data, output_file_path)

    print(f"Data for player '{player_name}' has been extracted and saved to '{output_file_path}'.")
