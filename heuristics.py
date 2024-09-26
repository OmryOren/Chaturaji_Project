from json import encoder
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
import random
from tensorflow import Tensor
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from chess_game import State

# Define the learning rate scheduler function
def lr_scheduler(epoch, lr):
    return lr * 0.99 if epoch > 0 else lr

# Build the autoencoder model
def build_autoencoder(board_shape, random_board_states):
    """Build the autoencoder model with dynamic learning rate."""
    
    # Encoder structure
    input_board = Input(shape=board_shape)
    x = Flatten()(input_board)  # Flatten the board input to a 1D vector
    x = Dense(1024, activation='relu')(x)
    x = Dense(600, activation='relu')(x)
    x = Dense(400, activation='relu')(x)
    x = Dense(200, activation='relu')(x)
    encoded = Dense(100, activation='relu')(x)  # Compressed latent representation

    # Decoder structure to reconstruct the board from the latent space
    x = Dense(200, activation='relu')(encoded)
    x = Dense(400, activation='relu')(x)
    x = Dense(600, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    decoded = Dense(np.prod(board_shape), activation='sigmoid')(x)
    decoded = Reshape(board_shape)(decoded)

    # Autoencoder Model: Maps input to reconstruction
    autoencoder = Model(input_board, decoded)
    encoder = Model(input_board, encoded)

    # Compile the model with an initial learning rate
    optimizer = Adam(learning_rate=0.01)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    # Pretrain the autoencoder with dynamic learning rate
    pretrain_autoencoder(autoencoder, random_board_states)

    return autoencoder, encoder

# Pretrain autoencoder function
def pretrain_autoencoder(autoencoder, board_states):
    """Pretrain the autoencoder with random board states and a dynamic learning rate."""
    board_states = np.array(board_states)
    autoencoder.trainable = True

    # Define the learning rate scheduler callback
    lr_callback = LearningRateScheduler(lr_scheduler)

    # Train the model with the learning rate scheduler
    autoencoder.fit(board_states, board_states, epochs=200, batch_size=500, shuffle=True, callbacks=[lr_callback])
class BaseHeuristic:
    def __init__(self):
        self.piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'K': 6}

    def material_balance(self, state, player_color):
        balance = 0
        for (x, y), piece in state.pieces.items():
            if piece.color == player_color:
                balance += self.piece_values.get(piece.piece_type, 0)
        return balance

    def mobility(self, state, player_color):
        moves = state.get_all_possible_moves(player_color)
        return len(moves)

    def king_safety(self, state, player_color):
        king_pos = None
        for pos, piece in state.pieces.items():
            if piece.color == player_color and piece.piece_type == 'K':
                king_pos = pos
                break

        if not king_pos:
            return 0

        safety_score = 0
        for (x, y), piece in state.pieces.items():
            if piece.color != player_color:
                possible_moves = state.get_possible_moves(x, y, piece.piece_type, piece.color)
                if king_pos in possible_moves:
                    safety_score -= 3  # King is in check

        for (x, y), piece in state.pieces.items():
            if piece.color == player_color and piece.piece_type != 'K':
                possible_moves = state.get_possible_moves(x, y, piece.piece_type, piece.color)
                for move in possible_moves:
                    if move == king_pos:
                        safety_score -= 1  # This piece is pinned

        return safety_score

    def center_control(self, state, player_color):
        center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
        control = 0
        for pos, piece in state.pieces.items():
            if piece.color == player_color and pos in center_squares:
                control += 1
        return control

    def threatened_piece_analysis(self, state, player_color):
        total_penalty = 0
        piece_values = self.piece_values

        for (x, y), piece in state.pieces.items():
            if piece.color != player_color or piece.piece_type == 'K':
                continue

            piece_value = piece_values.get(piece.piece_type, 0)
            attackers = 0
            protectors = 0
            lower_value_attackers = 0

            for (enemy_x, enemy_y), enemy_piece in state.pieces.items():
                if enemy_piece.color != player_color:
                    enemy_moves = state.get_possible_moves(enemy_x, enemy_y, enemy_piece.piece_type, enemy_piece.color)
                    if (x, y) in enemy_moves:
                        attackers += 1
                        enemy_value = piece_values.get(enemy_piece.piece_type, 0)
                        if enemy_value < piece_value:
                            lower_value_attackers += 1

            for (friend_x, friend_y), friend_piece in state.pieces.items():
                if friend_piece.color == player_color and friend_piece.piece_type != 'K':
                    friend_moves = state.get_possible_moves(friend_x, friend_y, friend_piece.piece_type, friend_piece.color)
                    if (x, y) in friend_moves:
                        protectors += 1

            if attackers > 0:
                total_penalty += 2 * attackers
            if protectors > 0:
                total_penalty -= protectors

            if lower_value_attackers > 0:
                total_penalty += 5 * lower_value_attackers

        return total_penalty

    def score_distance(self, state, player_color):
        player_score = state.players[player_color].score
        max_score = max(player.score for player in state.players.values())
        return max_score - player_score if max_score > player_score else 0

    def evaluate_state(self, state):
        """Return the heuristic values for all players."""
        heuristics_per_player = {}
        for player_color in state.turn_order:
            material = self.material_balance(state, player_color)
            mobility = self.mobility(state, player_color)
            king_safety = self.king_safety(state, player_color)
            center_control = self.center_control(state, player_color)
            threats_and_protection = self.threatened_piece_analysis(state, player_color)
            score_diff = self.score_distance(state, player_color)

            heuristics_per_player[player_color] = np.array([
                material,                 # Material balance
                mobility,                 # Mobility (number of legal moves)
                king_safety,              # King safety (penalizes for check and pins)
                center_control,           # Center control (number of pieces in center)
                threats_and_protection,   # Threats to and protection of non-king pieces
                score_diff                # Distance from first place in terms of score
            ])
        return heuristics_per_player

class HeuristicAI:
    def __init__(self, color, model, output_size):
        self.color = color
        random_board_states = self.create_random_board_states(100000)
        self.autoencoder, self.encoder = build_autoencoder((8, 8, 2), random_board_states)
        self.BaseHeuristic = BaseHeuristic()
        self.model = model
        self.output_size = output_size

    def get_state_input(self, state):
        """Return the input vector for the neural network, combining all game-related features."""
        
        # 1. Board Representation (latent space)
        board_flattened = self.flatten_board(state)
        board_latent = self.encoder.predict(board_flattened.reshape(1, 8, 8, 2)).flatten()
        
        # 2. Piece Information (piece counts and king status)
        piece_info = self.get_piece_info(state)
        
        # 3. Player Information (active status, score, turn order, current player)
        player_info = self.get_player_info(state)
        
        # 4. Turn Information (turn index and remaining players)
        turn_info = self.get_turn_info(state)
        
        # 5. Move Possibilities (number of possible moves)
        move_info = self.get_move_info(state)
        
        # 6. Game Status (check, king capture, surrender status)
        game_status = self.get_game_status(state)
        
        # 7. Heuristic Features (material balance, king safety, etc.)
        heuristic_values_red = self.BaseHeuristic.evaluate_state(state)["red"]
        heuristic_values_blue = self.BaseHeuristic.evaluate_state(state)["blue"]
        heuristic_values_yellow = self.BaseHeuristic.evaluate_state(state)["yellow"]
        heuristic_values_green = self.BaseHeuristic.evaluate_state(state)["green"]

        # Combine game-related information into a single array (for game_info input)
        game_info = np.concatenate([piece_info, player_info, turn_info, move_info, game_status])

        # Return the inputs as a dictionary
        return {
            "board_latent": board_latent,
            "game_info": game_info,
            "heuristic_red": heuristic_values_red,
            "heuristic_blue": heuristic_values_blue,
            "heuristic_yellow": heuristic_values_yellow,
            "heuristic_green": heuristic_values_green
        } 

    def flatten_board(self, state):
        """Flatten the 8x8 board with piece types and colors."""
        board_rep = np.zeros((8, 8, 2))  # One-hot for piece types and colors
        for (x, y), piece in state.pieces.items():
            piece_type = self.piece_type_to_index(piece.piece_type)
            piece_color = self.color_to_index(piece.color)
            board_rep[x][y] = [piece_type, piece_color]
        return board_rep

    def create_random_board_states(self, num_states):
        """Generate random board states by playing random games."""
        random_states = []

        while len(random_states) < num_states:
            # Initialize a new game state
            state = State()

            # Simulate the game with random moves until it ends or we have enough states
            i = 0
            while not state.check_end_conditions():
                i += 1
                current_turn = state.turn_order[state.turn_index]
                current_player = state.players[current_turn]
                
                if not current_player.active:
                    state.turn_index = (state.turn_index + 1) % len(state.turn_order)
                    continue

                # Select a random move for other players, or predict for AI's turn
                if current_turn == self.color:
                    # Use the model to predict the best move
                    _, _, best_action = self.choose_best_action(state)
                    random_move = best_action
                else:
                    # Get possible moves for the current player
                    possible_moves = state.get_all_possible_moves(current_turn)

                    # Choose a random move, but retry if surrender is chosen
                    random_move = random.choice(possible_moves)
                    if random_move[0] == 'surrender':
                        random_move = random.choice(possible_moves)

                # Execute the chosen move
                if random_move[0] == 'surrender':
                    state.surrender(current_turn)
                else:
                    state.move_piece(*random_move)

                # Append the current state to the list
                if i > 16 or (i > random.randint(0, 32)):
                    random_states.append(self.flatten_board(state))

                # Break if we have enough states
                if len(random_states) >= num_states:
                    break

                # Update to the next player's turn
                state.turn_index = (state.turn_index + 1) % len(state.turn_order)

        return random_states


    def piece_type_to_index(self, piece_type):
        piece_dict = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'K': 5}
        return piece_dict.get(piece_type, 0)

    def color_to_index(self, color):
        color_dict = {'red': 1, 'blue': 2, 'yellow': 3, 'green': 4}
        return color_dict.get(color, 0)

    def get_piece_info(self, state):
        """Return a feature vector for piece counts and king status for each player."""
        piece_info = []
        for color in state.turn_order:
            count = sum(1 for piece in state.pieces.values() if piece.color == color)
            king_status = any(piece.piece_type == 'K' and piece.color == color for piece in state.pieces.values())
            piece_info.extend([count, int(king_status)])
        return np.array(piece_info)

    def get_player_info(self, state):
        """Return player-related info such as active status, score, turn order, and current player."""
        player_info = []
        for color in state.turn_order:
            player = state.players[color]
            player_info.extend([int(player.active), player.score])
        player_info.append(state.turn_index)  # Append current player's turn
        return np.array(player_info)

    def get_turn_info(self, state):
        """Return turn-related information such as turn index and remaining active players."""
        remaining_players = sum(player.active for player in state.players.values())
        return np.array([state.turn_index, remaining_players])

    def get_move_info(self, state):
        """Return move possibilities for each piece for the current player."""
        moves = state.get_all_possible_moves(self.color)
        return np.array([len(moves)])  # Only the number of legal moves

    def get_game_status(self, state):
        """Return game status such as check conditions, king capture, and surrender status."""
        king_in_check = any(self.BaseHeuristic.king_safety(state, color) < 0 for color in state.turn_order)
        king_capture = any(not state.players[color].active for color in state.turn_order)
        surrender_status = any(player.active == False for player in state.players.values())
        return np.array([int(king_in_check), int(king_capture), int(surrender_status)])
    
    def get_all_neighbors(self, state):
        """Generate all possible neighbor states from the current board state."""
        possible_actions = state.get_all_possible_moves(self.color)
        neighbor_states = []

        for action in possible_actions:
            # Copy the current state to avoid modifying it
            new_state = State(state)

            # Execute the action to get a new neighbor state
            if action[0] == 'surrender':
                new_state.surrender(self.color)
            else:
                new_state.move_piece(*action)

            # Append the new state to the list of neighbor states
            neighbor_states.append((self.get_state_input(new_state), action))  # Store both the new state and the action taken

        return neighbor_states

    def choose_best_action_using_tournament(self, state):
        """Choose the best action using a tournament-style comparison of neighbor states."""
        # Get all possible neighbor states
        neighbors = self.get_all_neighbors(state)
        lossing_states = []
        # If there's only one neighbor, no need for comparison
        if len(neighbors) == 1:
            return lossing_states, neighbors[0][0], neighbors[0][1]  # Return the result for the only neighbor

        # Tournament-style comparison: Compare states two at a time
        while len(neighbors) > 1:
            neighbors = random.sample(neighbors, len(neighbors))  # Shuffle the neighbors for random comparison
            next_round = []
            for i in range(0, len(neighbors), 2):
                if i + 1 < len(neighbors):
                    # Get the two states for comparison
                    state_input_1, action_1 = neighbors[i]
                    state_input_2, action_2 = neighbors[i + 1]

                    # Use the Siamese model to predict which state is better
                    inputs = [
                        state_input_1["board_latent"], state_input_1["game_info"], state_input_1["heuristic_red"],
                        state_input_1["heuristic_blue"], state_input_1["heuristic_yellow"], state_input_1["heuristic_green"],
                        state_input_2["board_latent"], state_input_2["game_info"], state_input_2["heuristic_red"],
                        state_input_2["heuristic_blue"], state_input_2["heuristic_yellow"], state_input_2["heuristic_green"]
                    ]

                    # Ensure inputs are wrapped as individual numpy arrays
                    inputs = [np.array([input_val]) for input_val in inputs]

                    # Perform prediction
                    comparison = self.model.predict(inputs)

                    # Choose the better state (higher probability in the output)
                    if np.argmax(comparison) == 0:
                        next_round.append((state_input_1, action_1))  # State 1 is better
                        lossing_states.append(state_input_2)
                    else:
                        next_round.append((state_input_2, action_2))  # State 2 is better
                        lossing_states.append(state_input_1)
                else:
                    # If there's an odd number of neighbors, move the last one to the next round directly
                    next_round.append(neighbors[i])

            # Prepare for the next round of comparisons
            neighbors = next_round

        # The final remaining neighbor is the best one
        best_state, best_action = neighbors[0]
        return lossing_states, best_state, best_action

    def choose_best_action(self, state):
        """Choose the best action using the neural network model."""
        return self.choose_best_action_using_tournament(state)
