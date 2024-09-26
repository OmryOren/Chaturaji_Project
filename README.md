# Chaturaji AI Project

This project implements an AI for the Chaturaji game, a chess variant with four players. The AI uses heuristic-based models and deep learning to train agents capable of playing the game. The project includes four main files:

- `test.py`: The main file to run and train the AI models.
- `chess_game.py`: The core game logic for Chaturaji, including player management, board setup, and move execution.
- `heuristics.py`: Contains heuristic functions that evaluate the game state and help the AI make better decisions.
- `model.py`: Defines the neural network architecture and training process for the AI agents.

## Project Structure

├── test.py # Main entry point for training and running the AI 
├── chess_game.py # Core game logic and state management 
├── heuristics.py # Heuristic functions for AI decision-making 
├── model.py # Neural network architecture and training logic 
├── README.md # Project description and setup instructions


## Requirements

This project uses the following dependencies:

- Python 3.x
- TensorFlow (for the neural network)
- NumPy (for numerical operations)

To install the necessary dependencies, you can use:

```bash
pip install tensorflow numpy

## Files Description

1. test.py
This is the main file that simulates games and trains the AI models. It contains the function train_model which runs multiple games and trains the AI over epochs. The AI uses a combination of neural networks and heuristic functions to evaluate game states and make decisions.

usage: python test.py

2. chess_game.py
This file defines the State class, which manages the game board, player turns, and game logic. It also includes functions for moving pieces, checking end conditions, and managing player actions (e.g., surrendering, capturing pieces).

3. heuristics.py
This file defines the heuristic evaluation functions used by the AI to assess the value of different board states. These heuristics are used by the HeuristicAI to help the AI make more informed decisions.

4. model.py
This file defines the neural network model used by the AI agents to evaluate the game states and decide on the best moves. The build_siamese_model function creates a neural network that compares two board states and predicts which is better. It also includes training logic and backpropagation using rewards.\

## How to Run the Project

Train the Model: To train the AI models, simply run the test.py file. The training process will simulate a specified number of games, save models at regular intervals, and store them in your local system or Google Drive (if using Colab).

using: python test.py

Configure Hyperparameters: You can adjust the number of epochs, games per epoch, or other model parameters in test.py or model.py to suit your needs.

Game Simulation: The game logic in chess_game.py simulates the Chaturaji gameplay, while heuristics.py and model.py handle AI decision-making and learning.

License
This project is for educational purposes. Please feel free to use, modify, and distribute it under the terms of your preferred open-source license.

