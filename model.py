import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Flatten
from keras.callbacks import LearningRateScheduler, EarlyStopping
from chess_game import State
from heuristics import HeuristicAI

TF_ENABLE_ONEDNN_OPTS=0

# Build a model for board latent features
def build_board_model(input_size):
    """Builds a model that processes board latent features."""
    input_layer = Input(shape=(input_size,))
    dense_1 = Dense(400, activation='relu')(input_layer)
    dense_2 = Dense(200, activation='relu')(dense_1)
    dense_3 = Dense(100, activation='relu')(dense_2)
    return Model(inputs=input_layer, outputs=dense_3)

# Build a model for game-related information like piece info, player info, etc.
def build_game_info_model(info_size):
    """Builds a model that processes game-related features (piece_info, player_info, etc.)."""
    input_layer = Input(shape=(info_size,))
    dense_1 = Dense(200, activation='relu')(input_layer)
    dense_2 = Dense(100, activation='relu')(dense_1)
    return Model(inputs=input_layer, outputs=dense_2)

# Build a model for heuristic values
def build_heuristic_model(heuristic_size):
    """Builds a model that processes heuristic-related features."""
    input_layer = Input(shape=(heuristic_size,))
    dense_1 = Dense(200, activation='relu')(input_layer)
    dense_2 = Dense(100, activation='relu')(dense_1)
    return Model(inputs=input_layer, outputs=dense_2)

# Learning rate scheduler
def scheduler(epoch, lr):
    return lr * 0.99 if lr > 5e-5 else 5e-5  # Decay learning rate by 0.98 with a lower bound

# Build the Siamese model with BaseHeuristic integration
def build_siamese_model(board_input_size, game_info_size, heuristic_size):
    """Builds the Siamese model that compares two states using latent board features, game info, and heuristic values."""
    
    # Input layers for two board states
    board_input_1 = Input(shape=(board_input_size,))
    board_input_2 = Input(shape=(board_input_size,))
    
    # Input layers for game information of two board states
    game_info_input_1 = Input(shape=(game_info_size,))
    game_info_input_2 = Input(shape=(game_info_size,))
    
    # Input layers for heuristic values for all 4 players (1st state)
    heuristic_input_1_red = Input(shape=(heuristic_size,))
    heuristic_input_1_blue = Input(shape=(heuristic_size,))
    heuristic_input_1_yellow = Input(shape=(heuristic_size,))
    heuristic_input_1_green = Input(shape=(heuristic_size,))
    
    # Input layers for heuristic values for all 4 players (2nd state)
    heuristic_input_2_red = Input(shape=(heuristic_size,))
    heuristic_input_2_blue = Input(shape=(heuristic_size,))
    heuristic_input_2_yellow = Input(shape=(heuristic_size,))
    heuristic_input_2_green = Input(shape=(heuristic_size,))
    
    # Create models for board, game info, and heuristic features
    board_model = build_board_model(board_input_size)
    game_info_model = build_game_info_model(game_info_size)
    heuristic_model = build_heuristic_model(heuristic_size)
    
    # Process both board states with the shared models
    processed_board_1 = board_model(board_input_1)
    processed_board_2 = board_model(board_input_2)
    
    processed_game_info_1 = game_info_model(game_info_input_1)
    processed_game_info_2 = game_info_model(game_info_input_2)
    
    # Process heuristic values for all 4 players (state 1)
    processed_heuristic_1_red = heuristic_model(heuristic_input_1_red)
    processed_heuristic_1_blue = heuristic_model(heuristic_input_1_blue)
    processed_heuristic_1_yellow = heuristic_model(heuristic_input_1_yellow)
    processed_heuristic_1_green = heuristic_model(heuristic_input_1_green)
    
    # Process heuristic values for all 4 players (state 2)
    processed_heuristic_2_red = heuristic_model(heuristic_input_2_red)
    processed_heuristic_2_blue = heuristic_model(heuristic_input_2_blue)
    processed_heuristic_2_yellow = heuristic_model(heuristic_input_2_yellow)
    processed_heuristic_2_green = heuristic_model(heuristic_input_2_green)
    
    # Combine all heuristic features for both states
    combined_heuristic_1 = Concatenate()([processed_heuristic_1_red, processed_heuristic_1_blue, processed_heuristic_1_yellow, processed_heuristic_1_green])
    combined_heuristic_2 = Concatenate()([processed_heuristic_2_red, processed_heuristic_2_blue, processed_heuristic_2_yellow, processed_heuristic_2_green])
    
    # Combine all features for both states
    combined_1 = Concatenate()([processed_board_1, processed_game_info_1, combined_heuristic_1])
    combined_2 = Concatenate()([processed_board_2, processed_game_info_2, combined_heuristic_2])
    
    # Calculate the difference between the two combined states
    diff = Concatenate()([combined_1, combined_2])
    
    # Dense layers to process the difference and determine which state is better
    dense_1 = Dense(400, activation='relu')(diff)
    dense_2 = Dense(200, activation='relu')(dense_1)
    
    # Output layer: a softmax predicting which of the two states is better
    output = Dense(2, activation='softmax')(dense_2)
    
    # Here is where you put the model definition
    model = Model([
        board_input_1, game_info_input_1, heuristic_input_1_red, heuristic_input_1_blue, heuristic_input_1_yellow, heuristic_input_1_green,
        board_input_2, game_info_input_2, heuristic_input_2_red, heuristic_input_2_blue, heuristic_input_2_yellow, heuristic_input_2_green
    ], output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def new_track_game_history(ai_agents):
    """Track the game history for each agent to record the states they encountered."""
    for color in ai_agents:
        histories[color] = []        

def update_game_history(color, state, possible_states):
    """Update the history of each agent with the state they visited and the alternatives."""
    histories[color].append((state, possible_states))

def train_model(ai_agents, num_games=1000):
    """Simulate games and train the AI using HeuristicAI."""

    for game in range(num_games):
        state = State()
        print(f"Game {game+1}/{num_games} started.")
        
        # Track game history
        new_track_game_history(ai_agents)

        while not state.check_end_conditions():
            current_turn = state.turn_order[state.turn_index]
            current_player = state.players[current_turn]
            if not current_player.active:
                state.turn_index = (state.turn_index + 1) % len(state.turn_order)
                continue

            ai_agent = ai_agents[current_turn]
            lossing_states, state_chosen, action = ai_agent.choose_best_action(state)
            
            # Execute the chosen action in the game
            if action[0] == 'surrender':
                state.surrender(current_turn)
            else:
                state.move_piece(*action)
            
            # Update the turn to the next player
            state.turn_index = (state.turn_index + 1) % len(state.turn_order)
            
            # Update the history with the chosen state and the alternatives
            update_game_history(current_turn, state_chosen, lossing_states)

        # Assign rewards based on the outcome of the game
        result = calculate_rewards(state, ai_agents)
        # Backpropagate rewards using game history
        backpropagate_rewards(ai_agents, result)

        print(f"Game {game+1} ended. Scores: {state.get_scores()}")
        if game % 100 == 0:
            for color, ai_agent in ai_agents.items():
                ai_agent.model.save(f"model_{color}_{game}.h5")
    for color, ai_agent in ai_agents.items():
        ai_agent.model.save(f"model_{color}_final.h5")

def calculate_rewards(state, ai_agents):
    """Assign rewards to the AI agents based on game results."""
    max_score = max(player.score for player in state.players.values())
    result = [0, 0, 0, 0]  # Change this to a list
    i = 0
    for color, ai_agent in ai_agents.items():
        if state.players[color].score == max_score:
            result[i] = 1  # Reward for winning
        else:
            result[i] = -0.2  # Penalty for losing
        i += 1
    return result

def backpropagate_rewards(ai_agents, result):
    """Backpropagate the final rewards for each agent using the last game state."""
    early_stopping = EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(scheduler)
    play_order = list(ai_agents.keys())
    for i, color in enumerate(play_order):
        ai_agent = ai_agents[color]
        reward = result[i]
        for state_input, alternative_states in histories[color]:
            for alt_state_input in alternative_states:
                inputs = [
                    state_input["board_latent"], state_input["game_info"], state_input["heuristic_red"],
                    state_input["heuristic_blue"], state_input["heuristic_yellow"], state_input["heuristic_green"],
                    alt_state_input["board_latent"], alt_state_input["game_info"], alt_state_input["heuristic_red"],
                    alt_state_input["heuristic_blue"], alt_state_input["heuristic_yellow"], alt_state_input["heuristic_green"]
                ]
                inputs = [np.array([inp]) for inp in inputs]
                
                # Reward is [1, 0] for the winning state, and [0, 1] for the alternative state
                reward_array = np.array([[reward, 0]])
                ai_agent.model.fit(inputs, reward_array, epochs=1, verbose=0, callbacks=[early_stopping, lr_scheduler])
                
            reward *= 0.95  # Discounted reward for previous states

# Define input sizes based on feature representation
latent_size = 100  # Adjusted size from encoder
game_info_size = 23  # Example size for game-related info like piece counts, player info
heuristic_size = 6  # Example size for heuristic values (based on material, mobility, king safety, etc.)

# Build the Siamese model
supervised_model = build_siamese_model(latent_size, game_info_size, heuristic_size)

# Initialize AI agents
ai_agents = {
    'red': HeuristicAI('red', supervised_model, 64),
    'blue': HeuristicAI('blue', supervised_model, 64),
    'yellow': HeuristicAI('yellow', supervised_model, 64),
    'green': HeuristicAI('green', supervised_model, 64)
}
histories = {color: [] for color in ai_agents}

# Train the model using the AI agents
train_model(ai_agents, num_games=1000)
