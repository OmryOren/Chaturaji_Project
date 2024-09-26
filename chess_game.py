import numpy as np

class Piece:
    def __init__(self, piece_type, color):
        self.piece_type = piece_type
        self.color = color

class Player:
    def __init__(self, color, is_ai=True):
        self.color = color
        self.score = 0
        self.active = True
        self.is_ai = is_ai  # Flag to determine if this player is AI-controlled

class State:
    def __init__(self, state=None):
        if state is not None:
            self.board = state.board.copy()
            self.pieces = state.pieces.copy()
            self.turn_order = state.turn_order.copy()
            self.turn_index = state.turn_index
            self.players = {color: Player(color, is_ai=True) for color in state.players}
            self.no_capture_moves = state.no_capture_moves
            self.max_no_capture_moves = state.max_no_capture_moves
            self.board_history = state.board_history.copy()
        else:
            self.board = np.zeros((8, 8))
            self.pieces = {
                (0, 0): Piece('R', 'blue'), (1, 0): Piece('N', 'blue'), (2, 0): Piece('B', 'blue'), (3, 0): Piece('K', 'blue'), 
                (0, 1): Piece('P', 'blue'), (1, 1): Piece('P', 'blue'), (2, 1): Piece('P', 'blue'), (3, 1): Piece('P', 'blue'),

                (0, 4): Piece('K', 'yellow'), (0, 5): Piece('B', 'yellow'), (0, 6): Piece('N', 'yellow'), (0, 7): Piece('R', 'yellow'),
                (1, 4): Piece('P', 'yellow'), (1, 5): Piece('P', 'yellow'), (1, 6): Piece('P', 'yellow'), (1, 7): Piece('P', 'yellow'), 

                (4, 7): Piece('K', 'green'), (5, 7): Piece('B', 'green'), (6, 7): Piece('N', 'green'), (7, 7): Piece('R', 'green'),
                (4, 6): Piece('P', 'green'), (5, 6): Piece('P', 'green'), (6, 6): Piece('P', 'green'), (7, 6): Piece('P', 'green'),  
                
                (7, 3): Piece('K', 'red'), (7, 2): Piece('B', 'red'), (7, 1): Piece('N', 'red'), (7, 0): Piece('R', 'red'), 
                (6, 3): Piece('P', 'red'), (6, 2): Piece('P', 'red'), (6, 1): Piece('P', 'red'), (6, 0): Piece('P', 'red'), 
            }
            for (x, y), piece in self.pieces.items():
                self.board[x][y] = 1

            self.turn_order = ['red', 'blue', 'yellow', 'green']
            self.turn_index = 0
            self.players = {color: Player(color, is_ai=True) for color in self.turn_order}  # All players default to AI

            self.no_capture_moves = 0
            self.max_no_capture_moves = 50
            self.board_history = []

    def print_board(self):
        print("  a  b  c  d  e  f  g  h")
        for x in range(8):
            print(8 - x, end=' ')
            for y in range(8):
                if (x, y) in self.pieces:
                    piece = self.pieces[(x, y)]
                    if piece.piece_type == 'D':
                        print("DD", end=' ')
                    else:
                        print(f"{piece.piece_type[0]}{piece.color[0]}", end=' ')
                else:
                    print("..", end=' ')
            print()
        scores = self.get_scores()
        print("Scores:", scores)
        print()

    def get_scores(self):
        return {color: player.score for color, player in self.players.items()}

    def move_piece(self, start, end):
        piece = self.pieces[start]
        captured_piece = None
        if end in self.pieces:
            captured_piece = self.pieces[end]
            if captured_piece.color != 'dead':  # Capture enemy piece
                self.players[piece.color].score += self.get_piece_value(captured_piece.piece_type)
                self.no_capture_moves = 0  # Reset no capture move counter
        else:
            self.no_capture_moves += 1  # Increment no capture move counter

        self.pieces[end] = piece
        del self.pieces[start]
        self.board[start[0]][start[1]] = 0
        self.board[end[0]][end[1]] = 1

        if captured_piece and captured_piece.piece_type == 'K':
            self.capture_king(captured_piece.color)

        if piece.piece_type == 'P':
            if piece.color in ['red', 'green'] and end[0] == 0:
                self.pieces[end] = Piece('R', piece.color)
            elif piece.color in ['blue', 'yellow'] and end[0] == 7:
                self.pieces[end] = Piece('R', piece.color)

    def capture_king(self, color):
        if color == 'dead':
          self.players[self.turn_order[self.turn_index]].score += self.get_piece_value('K')
          return
        for pos, piece in list(self.pieces.items()):
            if piece.color == color:
                self.pieces[pos] = Piece('D', 'dead')  # Mark as dead
                self.board[pos[0]][pos[1]] = 1  # Keep the board occupied to show the dead piece
        self.players[color].active = False

    def get_piece_value(self, piece_type):
        piece_values = {'P': 1, 'N': 3, 'B': 5, 'R': 5, 'K': 3}
        return piece_values.get(piece_type, 0)

    def get_pawn_moves(self, x, y, color):
        moves = []
        direction = -1 if color in ['red', 'green'] else 1
        if color in ['red', 'yellow']:
            if 0 <= x + direction < 8 and self.board[x + direction][y] == 0:
                moves.append((x + direction, y))
        else:
            if 0 <= y + direction < 8 and self.board[x][direction + y] == 0:
                moves.append((x, y + direction))
        for dy in [-1, 1]:
            if color in ['red', 'yellow']:
                if 0 <= y + dy < 8 and 0 <= x + direction < 8 and self.board[x + direction][y + dy] == 1:
                    if self.pieces[(x + direction, y + dy)].color != color:
                        moves.append((x + direction, y + dy))
            else:
                if 0 <= x + dy < 8 and 0 <= y + direction < 8 and self.board[x + dy][y + direction] == 1:
                    if self.pieces[(x + dy, y + direction)].color != color:
                        moves.append((x + dy, y + direction))
        return moves

    def get_knight_moves(self, x, y, color):
        moves = []
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for dx, dy in knight_moves:
            if 0 <= x + dx < 8 and 0 <= y + dy < 8:
                if self.board[x + dx][y + dy] == 0 or self.pieces[(x + dx, y + dy)].color != color:
                    moves.append((x + dx, y + dy))
        return moves

    def get_bishop_moves(self, x, y, color):
        moves = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 0 <= nx < 8 and 0 <= ny < 8:
                if self.board[nx][ny] == 0:
                    moves.append((nx, ny))
                elif self.pieces[(nx, ny)].color != color:
                    moves.append((nx, ny))
                    break
                else:
                    break
                nx += dx
                ny += dy
        return moves

    def get_rook_moves(self, x, y, color):
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 0 <= nx < 8 and 0 <= ny < 8:
                if self.board[nx][ny] == 0:
                    moves.append((nx, ny))
                elif self.pieces[(nx, ny)].color != color:
                    moves.append((nx, ny))
                    break
                else:
                    break
                nx += dx
                ny += dy
        return moves

    def get_king_moves(self, x, y, color):
        moves = []
        king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in king_moves:
            if 0 <= x + dx < 8 and 0 <= y + dy < 8:
                if self.board[x + dx][y + dy] == 0 or self.pieces[(x + dx, y + dy)].color != color:
                    moves.append((x + dx, y + dy))
        return moves

    def get_all_possible_moves(self, player_color):
        moves = []
        for (x, y), piece in self.pieces.items():
            if piece.color == player_color:
                possible_moves = self.get_possible_moves(x, y, piece.piece_type, piece.color)
                for end in possible_moves:
                    moves.append(((x, y), end))
        moves.append(("surrender", None))
        return moves

    def get_possible_moves(self, x, y, piece_type, color):
        if piece_type == 'P':
            return self.get_pawn_moves(x, y, color)
        elif piece_type == 'N':
            return self.get_knight_moves(x, y, color)
        elif piece_type == 'B':
            return self.get_bishop_moves(x, y, color)
        elif piece_type == 'R':
            return self.get_rook_moves(x, y, color)
        elif piece_type == 'K':
            return self.get_king_moves(x, y, color)
        return []

    def is_repetition(self):
        if len(self.board_history) < 3:
            return False
        x = self.board_history[-1]
        y = self.board_history[-2]
        z = self.board_history[-3]
        return np.array_equal(x, y) and np.array_equal(y, z)

    def surrender(self, player_color):
        for pos, piece in list(self.pieces.items()):
            if piece.color == player_color:
                if piece.piece_type == 'K':
                    self.pieces[pos] = Piece('K', 'dead')
                    self.board[pos[0]][pos[1]] = 1
                else:
                    self.pieces[pos] = Piece('D', 'dead')
                    self.board[pos[0]][pos[1]] = 1
        self.players[player_color].active = False

    def play_game(self, ai_agents=None):
        while True:
            self.print_board()

            if self.check_end_conditions():
                break

            current_turn = self.turn_order[self.turn_index]
            current_player = self.players[current_turn]

            if not current_player.active:
                self.turn_index = (self.turn_index + 1) % len(self.turn_order)
                continue

            if current_player.is_ai and ai_agents:
                # AI-controlled player
                ai_agent = ai_agents[current_turn]
                action = ai_agent.choose_best_action(self)
            else:
                # Human-controlled player (you can replace this with a simulated decision)
                print(f"{current_turn}'s turn.")
                action = input("Enter a start position to make a move or 'surrender' to surrender: ").strip().lower()

            if action == 'surrender':
                self.surrender(current_turn)
                self.board_history.append(np.copy(self.board))
                self.turn_index = (self.turn_index + 1) % len(self.turn_order)
                continue
            
            if isinstance(action, tuple):
                start, end = action
                self.move_piece(start, end)

                if self.turn_index == len(self.turn_order) - 1:
                    self.board_history.append(np.copy(self.board))
                self.turn_index = (self.turn_index + 1) % len(self.turn_order)

        # After the game is finished, return the final state
        return self.get_scores()

    def check_end_conditions(self):
        active_count = sum(player.active for player in self.players.values())
        if active_count == 1:
            last_player = [color for color, player in self.players.items() if player.active][0]
            for pos, piece in self.pieces.items():
                if piece.piece_type == 'K' and piece.color != last_player:
                    self.players[last_player].score += self.get_piece_value('K')
            return True
        if self.turn_index == 0:
            if self.no_capture_moves >= self.max_no_capture_moves:
                return True
            if self.is_repetition():
                return True
        return False
