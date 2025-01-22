import pygame
import chess
import os
from play_eval_1 import ChessEngine  # Import your existing ChessEngine class

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = 800
BOARD_SIZE = 600
SQUARE_SIZE = BOARD_SIZE // 8
PIECE_SIZE = SQUARE_SIZE - 10
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_SQUARE = (240, 217, 181)  # Light brown
DARK_SQUARE = (181, 136, 99)    # Dark brown
HIGHLIGHT = (186, 202, 43)      # Highlight color for selected piece
MOVE_HINT = (106, 135, 77)      # Highlight color for possible moves

class ChessGUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Chess Game")
        self.clock = pygame.time.Clock()
        
        # Load chess pieces images
        self.pieces_images = {}
        self.load_pieces()
        
        # Initialize game state
        self.board = chess.Board()
        try:
            self.engine = ChessEngine('chess_model_v2.pth')
            print("Neural network model loaded successfully!")
        except Exception as e:
            print(f"Could not load model: {str(e)}")
            print("Continuing without neural network support...")
            self.engine = ChessEngine()
        
        self.selected_square = None
        self.valid_moves = []
        self.player_color = chess.WHITE
        self.is_game_over = False
        self.status_message = ""
        
    def load_pieces(self):
        """Load chess piece images from black and white subfolders"""
        # Black pieces (lowercase)
        black_pieces = ['p', 'n', 'b', 'r', 'q', 'k']
        # White pieces (uppercase)
        white_pieces = ['P', 'N', 'B', 'R', 'Q', 'K']
        
        # Load black pieces
        for piece in black_pieces:
            try:
                image = pygame.image.load(os.path.join('img', 'black', f'{piece}.png'))
                self.pieces_images[piece] = pygame.transform.scale(image, (PIECE_SIZE, PIECE_SIZE))
            except Exception as e:
                print(f"Couldn't load black piece image: {piece}")
                print(f"Error: {str(e)}")
                # Create placeholder for missing image
                surface = pygame.Surface((PIECE_SIZE, PIECE_SIZE))
                surface.fill(WHITE)
                pygame.draw.rect(surface, BLACK, surface.get_rect(), 2)
                font = pygame.font.SysFont('Arial', 30)
                text = font.render(piece, True, BLACK)
                text_rect = text.get_rect(center=surface.get_rect().center)
                surface.blit(text, text_rect)
                self.pieces_images[piece] = surface
                
        # Load white pieces
        for piece in white_pieces:
            try:
                image = pygame.image.load(os.path.join('img', 'white', f'{piece}.png'))
                self.pieces_images[piece] = pygame.transform.scale(image, (PIECE_SIZE, PIECE_SIZE))
            except Exception as e:
                print(f"Couldn't load white piece image: {piece}")
                print(f"Error: {str(e)}")
                # Create placeholder for missing image
                surface = pygame.Surface((PIECE_SIZE, PIECE_SIZE))
                surface.fill(WHITE)
                pygame.draw.rect(surface, BLACK, surface.get_rect(), 2)
                font = pygame.font.SysFont('Arial', 30)
                text = font.render(piece, True, BLACK)
                text_rect = text.get_rect(center=surface.get_rect().center)
                surface.blit(text, text_rect)
                self.pieces_images[piece] = surface

    def get_square_from_pos(self, pos):
        """Convert mouse position to chess square"""
        x, y = pos
        board_x = (WINDOW_SIZE - BOARD_SIZE) // 2
        board_y = (WINDOW_SIZE - BOARD_SIZE) // 2
        
        if (x < board_x or x >= board_x + BOARD_SIZE or 
            y < board_y or y >= board_y + BOARD_SIZE):
            return None
            
        file = (x - board_x) // SQUARE_SIZE
        rank = 7 - ((y - board_y) // SQUARE_SIZE)
        
        if self.player_color == chess.BLACK:
            file = 7 - file
            rank = 7 - rank
            
        return chess.square(file, rank)

    def get_square_rect(self, square):
        """Get pygame Rect for a chess square"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        if self.player_color == chess.BLACK:
            file = 7 - file
            rank = 7 - rank
            
        board_x = (WINDOW_SIZE - BOARD_SIZE) // 2
        board_y = (WINDOW_SIZE - BOARD_SIZE) // 2
        
        x = board_x + (file * SQUARE_SIZE)
        y = board_y + ((7 - rank) * SQUARE_SIZE)
        
        return pygame.Rect(x, y, SQUARE_SIZE, SQUARE_SIZE)

    def draw_board(self):
        """Draw the chess board"""
        self.screen.fill(GRAY)
        board_x = (WINDOW_SIZE - BOARD_SIZE) // 2
        board_y = (WINDOW_SIZE - BOARD_SIZE) // 2
        
        # Draw squares
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                rect = self.get_square_rect(square)
                color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, color, rect)
                
                # Highlight selected square
                if square == self.selected_square:
                    pygame.draw.rect(self.screen, HIGHLIGHT, rect, 3)
                
                # Highlight valid moves
                if square in self.valid_moves:
                    pygame.draw.rect(self.screen, MOVE_HINT, rect, 3)

        # Draw pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                rect = self.get_square_rect(square)
                piece_img = self.pieces_images[piece.symbol()]
                img_rect = piece_img.get_rect(center=rect.center)
                self.screen.blit(piece_img, img_rect)

        # Draw status message
        if self.status_message:
            font = pygame.font.SysFont('Arial', 24)
            text = font.render(self.status_message, True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE//2, 30))
            self.screen.blit(text, text_rect)

    def handle_click(self, pos):
        """Handle mouse click events"""
        if self.is_game_over:
            return
            
        square = self.get_square_from_pos(pos)
        if square is None:
            return
            
        # If it's not the player's turn, ignore the click
        if self.board.turn != self.player_color:
            return
            
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.player_color:
                self.selected_square = square
                self.valid_moves = [move.to_square 
                                  for move in self.board.legal_moves 
                                  if move.from_square == square]
        else:
            # Create the move
            move = None
            # Check if this is a pawn promotion move
            if (self.board.piece_at(self.selected_square) and 
                self.board.piece_at(self.selected_square).piece_type == chess.PAWN and
                ((square >= 56 and self.player_color == chess.WHITE) or
                 (square <= 7 and self.player_color == chess.BLACK))):
                move = chess.Move(self.selected_square, square, chess.QUEEN)
            else:
                move = chess.Move(self.selected_square, square)
                
            # Check if move is legal and get SAN before pushing
            if move in self.board.legal_moves:
                san = self.board.san(move)
                self.board.push(move)
                self.status_message = f"You played: {san}"
                
                if not self.board.is_game_over():
                    # Engine's turn
                    print("\nEngine is thinking...")
                    engine_move = self.engine.get_best_move(self.board)
                    if engine_move and engine_move in self.board.legal_moves:
                        san = self.board.san(engine_move)  # Get SAN before pushing the move
                        self.board.push(engine_move)
                        self.status_message = f"Engine played: {san}"
                    else:
                        print("Engine couldn't find a legal move!")
                    
                if self.board.is_game_over():
                    self.handle_game_over()
            
            self.selected_square = None
            self.valid_moves = []

    def handle_game_over(self):
        """Handle game over conditions"""
        self.is_game_over = True
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            self.status_message = f"Checkmate! {winner} wins!"
        elif self.board.is_stalemate():
            self.status_message = "Stalemate! Game is drawn."
        elif self.board.is_insufficient_material():
            self.status_message = "Draw due to insufficient material."
        elif self.board.is_fifty_moves():
            self.status_message = "Draw due to fifty-move rule."
        elif self.board.is_repetition():
            self.status_message = "Draw due to threefold repetition."
        else:
            self.status_message = "Game drawn."

    def run(self):
        """Main game loop"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset game
                        self.__init__()
                    elif event.key == pygame.K_f:  # Flip board
                        self.player_color = not self.player_color
            
            self.draw_board()
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()

def draw_menu(screen):
    """Draw the color selection menu"""
    screen.fill(GRAY)
    font = pygame.font.SysFont('Arial', 32)
    
    # Title
    title = font.render('Choose Your Color', True, BLACK)
    title_rect = title.get_rect(center=(WINDOW_SIZE//2, WINDOW_SIZE//4))
    screen.blit(title, title_rect)
    
    # White button
    white_btn = pygame.Rect(WINDOW_SIZE//4, WINDOW_SIZE//2, WINDOW_SIZE//2, 50)
    pygame.draw.rect(screen, WHITE, white_btn)
    pygame.draw.rect(screen, BLACK, white_btn, 2)
    white_text = font.render('Play as White', True, BLACK)
    white_text_rect = white_text.get_rect(center=white_btn.center)
    screen.blit(white_text, white_text_rect)
    
    # Black button
    black_btn = pygame.Rect(WINDOW_SIZE//4, WINDOW_SIZE//2 + 100, WINDOW_SIZE//2, 50)
    pygame.draw.rect(screen, LIGHT_SQUARE, black_btn)
    pygame.draw.rect(screen, BLACK, black_btn, 2)
    black_text = font.render('Play as Black', True, BLACK)
    black_text_rect = black_text.get_rect(center=black_btn.center)
    screen.blit(black_text, black_text_rect)
    
    pygame.display.flip()
    return white_btn, black_btn

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Chess Game")
    
    # Show menu and get color choice
    white_btn, black_btn = draw_menu(screen)
    color_chosen = False
    player_color = chess.WHITE
    
    while not color_chosen:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if white_btn.collidepoint(mouse_pos):
                    player_color = chess.WHITE
                    color_chosen = True
                elif black_btn.collidepoint(mouse_pos):
                    player_color = chess.BLACK
                    color_chosen = True
    
    # Start game with chosen color
    gui = ChessGUI()
    gui.player_color = player_color
    
    # If playing as black, let engine make first move
    if player_color == chess.BLACK:
        print("\nEngine is thinking...")
        engine_move = gui.engine.get_best_move(gui.board)
        if engine_move:
            san = gui.board.san(engine_move)
            gui.board.push(engine_move)
            gui.status_message = f"Engine played: {san}"
    
    gui.run()

if __name__ == "__main__":
    main()