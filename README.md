# Tic Tac Toe Bolt

**Tic Tac Toe Bolt** is a reinforcement learning project that implements "Infinite Tic Tac Toe" (where a player can have at most 3 marks, and the oldest disappears on the 4th move) as a Gymnasium environment. It features an AlphaZero-style AI agent trained via self-play and Monte Carlo Tree Search (MCTS).

## Project Overview

*   **Domain**: Infinite Tic Tac Toe (3x3 grid, max 3 marks per player).
*   **Core Logic**: Implemented as a custom Gymnasium environment (`TicTacToeBolt-v0`).
*   **AI Architecture**: AlphaZero (ResNet/CNN backbone with dual Policy/Value heads + MCTS).
*   **Performance**: Includes a C++ extension (`_mcts_cpp`) for high-performance MCTS during inference/training, supporting both CPU and CUDA.

## Directory Structure

*   `src/tic_tac_toe_bolt/`: Main source code.
    *   `env.py`: Gymnasium environment implementation.
    *   `model.py`: PyTorch Policy/Value Neural Network (CNN).
    *   `mcts.py`: Pure Python MCTS implementation.
    *   `mcts_cpp/`: C++ source for the optimized MCTS extension.
    *   `train.py`: Main training loop (Self-Play -> Optimize -> Evaluate).
    *   `play.py`: CLI script for human vs. AI play.
*   `tests/`: Unit and integration tests.
    *   `test_mechanics.py`: Verifies game rules (infinite mechanic, win conditions).
    *   `test_model.py`: Verifies NN architecture and forward pass.
    *   `test_mcts_cpp.py`: Verifies C++ MCTS bindings.
    *   `performance_benchmark.py`: Benchmarks Python vs C++ MCTS performance.
*   `play_match.py`: Script to pit two models against each other.

## Development & Usage

### 1. Installation

The project uses `setuptools` and builds a C++ extension.

```bash
# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install (compiles C++ extension)
pip install .
```

### 2. Training the AI

Start the AlphaZero training loop:

```bash
python3 src/tic_tac_toe_bolt/train.py
```
*   Generates checkpoints (e.g., `current_policy_50.pth`).
*   The training script automatically detects if CUDA is available and utilizes the C++ MCTS extension on the appropriate device.

### 3. Playing

**Human vs. AI:**
```bash
# Play against random/untrained
python3 src/tic_tac_toe_bolt/play.py

# Play against a specific model
python3 src/tic_tac_toe_bolt/play.py --model current_policy_50.pth --ai_starts
```

**Model vs. Model:**
```bash
python3 play_match.py --model1 current_policy_50.pth --model2 current_policy_100.pth --n_games 100
```

### 4. Testing

Run verification scripts to ensure game logic and model integrity:

```bash
# Test Game Logic
python3 tests/test_mechanics.py

# Test C++ Integration
python3 tests/test_mcts_cpp.py

# Benchmark Performance (Python vs C++ MCTS)
python3 tests/performance_benchmark.py
```

## Implementation Details

*   **State Representation**: The board is represented as a 3x3x3 tensor (Channel 0: Player Marks, Channel 1: Opponent Marks, Channel 2: All 1s/0s for turn or bias).
*   **C++ Extension**: The `_mcts_cpp` module is built from `src/tic_tac_toe_bolt/mcts_cpp/mcts.cpp` using `torch.utils.cpp_extension`. It interacts with the Python environment by converting the Python board state into a C++ representation.
    *   **Device Support**: The C++ MCTS class is device-aware. It accepts a device string (e.g., "cuda" or "cpu") during initialization and ensures the TorchScript model and input tensors are on the correct device for inference.
*   **Model Loading**: The `play_match.py` and `play.py` scripts gracefully handle missing models by falling back to random/untrained behavior, but warning the user.
