# Tic Tac Toe Bolt

**Tic Tac Toe Bolt** is an implementation of "Infinite Tic Tac Toe" as a Gymnasium environment, featuring an AlphaZero-style AI agent.

## Game Rules
- **Grid**: 3x3.
- **Players**: 2 (X and O).
- **Infinite Mechanic**: Each player can have at most **3 marks** on the board. When a player places a 4th mark, their **oldest mark disappears**.
- **Win Condition**: 3 marks in a row, column, or diagonal.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd tic_tac_toe_bolt
    ```

2.  **Create a virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install .
    ```

## Usage

### Play Against AI
You can play against the AI (trained or untrained) using the play script.

```bash
# Play against an untrained agent (random/initial weights)
python3 src/tic_tac_toe_bolt/play.py

# Play against a trained model
python3 src/tic_tac_toe_bolt/play.py --model current_policy_50.pth

# Let AI start first
python3 src/tic_tac_toe_bolt/play.py --ai_starts
```

### Train the AI
To train the agent using AlphaZero (Self-Play + MCTS + Neural Network):

```bash
python3 src/tic_tac_toe_bolt/train.py
```
This will generate `.pth` model checkpoints (e.g., `current_policy_50.pth`).

### Run Random Agent
To verify the environment with a random agent:

```bash
python3 tests/random_agent.py
```

### Run Tests
To verify the game mechanics and model:

```bash
# Verify Infinite Mechanic
python3 tests/test_mechanics.py

# Verify Neural Network Forward Pass
python3 tests/test_model.py
```

## Project Structure
- `src/tic_tac_toe_bolt/`:
    - `env.py`: Gymnasium environment implementation.
    - `model.py`: PyTorch Policy/Value Neural Network.
    - `mcts.py`: Monte Carlo Tree Search implementation.
    - `train.py`: AlphaZero training loop.
    - `play.py`: Script to play against the AI.
- `tests/`: Verification scripts.
