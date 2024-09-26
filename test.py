from chess_game import State
from heuristics import HeuristicAI
import random
from keras.models import load_model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


ai_agents = {
    'red': HeuristicAI('red', load_model('model_red_60_3.h5'), 64),
    'blue': HeuristicAI('blue', load_model('model_red_60_3.h5'), 64),
    'yellow': HeuristicAI('yellow', load_model('model_red_60_3.h5'), 64),
    'green': HeuristicAI('green', load_model('model_red_60_3.h5'), 64)
}

def test_model(ai_agent, color, num_games=1000):
    """Simulate games and train the AI using HeuristicAI."""

    for game in range(num_games):
        state = State()
        print(f"Game {game+1}/{num_games} started.")
        sum = 0

        while not state.check_end_conditions():
            current_turn = state.turn_order[state.turn_index]
            current_player = state.players[current_turn]
            if not current_player.active:
                state.turn_index = (state.turn_index + 1) % len(state.turn_order)
                continue
            if current_player.color == color:
                # If the player is an AI, choose the best action
                flossing_states, state_chosen, action = ai_agent.choose_best_action(state)
            else:
                # If the player is GreedyOpponent, let them choose an action
                action = choose_greedy_action(state, current_player.color)
            # Execute the chosen action in the game
            if action[0] == 'surrender':
                state.surrender(current_turn)
            else:
                state.move_piece(*action)
            
            # Update the turn to the next player
            state.turn_index = (state.turn_index + 1) % len(state.turn_order)
        # Assign rewards based on the outcome of the game
        max_score = max(player.score for player in state.players.values())
        if state.players[color].score == max_score:
            sum += 1

    print ("Testing complete.")
    print (f"Win rate: {sum}/{num_games}")
    return sum

def choose_greedy_action(state, color):
    """Choose the best action for GreedyOpponent."""
    actions = state.get_all_possible_actions(color)
    #shuffle actions
    random.shuffle(actions)
    best_action = None
    best_score = -1
    for action in actions:
        state_copy = state.copy()
        state_copy.move_piece(*action)
        score = state_copy.get_scores()[color]
        if score > best_score:
            best_score = score
            best_action = action
    return best_action

performance = [test_model(ai_agents['red'], 'red', 1000), test_model(ai_agents['blue'], 'blue', 1000), test_model(ai_agents['yellow'], 'yellow', 1000), test_model(ai_agents['green'], 'green', 1000)]
print("-------------------------------------------------------------------------------------------------")   
print("Performance: ", performance)
print("Average performance: ", sum(performance)/4)
print("-------------------------------------------------------------------------------------------------")
