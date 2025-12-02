# Board Game AI

A reinforcement learning project implementing AlphaZero-style AI agents for multiple board games. Each game is implemented as a Gymnasium environment with both Python and C++ MCTS backends for high-performance training and inference.

## Supported Games

### Tic Tac Toe Bolt
"Infinite Tic Tac Toe" where each player can have at most 3 marks on the board. When a player places a 4th mark, their oldest mark disappears.
*   **Board**: 3x3 grid
*   **Environment**: `TicTacToeBolt-v0`
*   **Rendering**: ASCII, PyGame (human, rgb_array)

### Cublino Contra
A dice-based strategy game where players move dice across a 7x7 board. Dice roll when moved, changing their top face value. Players can capture opponent dice through surrounding battles.
*   **Board**: 7x7 grid with 7 dice per player
*   **Environment**: `CublinoContra-v0`
*   **Rendering**: ASCII, 3D OpenGL visualization
*   **Features**: Dice rolling mechanics, battle system, 3-fold repetition draw detection, state history for neural network input

## Project Overview

*   **AI Architecture**: AlphaZero (ResNet/CNN backbone with dual Policy/Value heads + MCTS).
*   **Performance**: Includes C++ extensions (`_mcts_cpp`) for high-performance MCTS during inference/training, supporting both CPU and CUDA.
*   **Training**: Parallel self-play with configurable workers, checkpointing, and evaluation against previous best models.

## Directory Structure

*   `src/tic_tac_toe_bolt/`: Tic Tac Toe Bolt source code.
    *   `env.py`: Gymnasium environment implementation.
    *   `model.py`: PyTorch Policy/Value Neural Network (CNN).
    *   `mcts.py`: Pure Python MCTS implementation.
    *   `mcts_cpp/`: C++ source for the optimized MCTS extension.
    *   `train.py`: Main training loop (Self-Play -> Optimize -> Evaluate).
    *   `play.py`: CLI script for human vs. AI play (supports difficulty levels).
    *   `play_match.py`: Script to pit two models against each other.
*   `src/cublino_contra/`: Cublino Contra source code.
    *   `env.py`: Gymnasium environment with dice mechanics and battle resolution.
    *   `model.py`: PyTorch Policy/Value Neural Network (ResNet with 4 residual blocks).
    *   `mcts.py`: Pure Python MCTS implementation.
    *   `mcts_cpp/`: C++ source for the optimized MCTS extension.
    *   `train.py`: Training loop with parallel self-play workers.
    *   `play.py`: 3D OpenGL visualization for human vs. AI play (supports game replay).
*   `tests/`: Unit and integration tests.
    *   `test_mechanics.py`: Tic Tac Toe game rules verification.
    *   `test_model.py`: NN architecture verification.
    *   `test_mcts_cpp.py`: Tic Tac Toe C++ MCTS bindings verification.
    *   `test_cublino.py`: Cublino Contra game rules verification.
    *   `test_cublino_ai.py`: Cublino Contra AI integration tests.
    *   `test_cublino_mcts_cpp.py`: Cublino Contra C++ MCTS bindings verification.
    *   `performance_benchmark.py`: Benchmarks Python vs C++ MCTS performance.
*   `extract_model.py`: Utility to extract model state dictionaries from checkpoints.
*   `run_train.sh`: Helper script to run Cublino Contra training with correct library paths.

## Development & Usage

### 1. Installation

This project is licensed under the GPL-2.0 License.

The project uses `setuptools` and builds C++ extensions for both games.

```bash
# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install (compiles C++ extensions)
pip install .
```

### 2. Training

#### Tic Tac Toe Bolt
```bash
# Standard training
python3 src/tic_tac_toe_bolt/train.py

# Train with Force Asymmetry (10% random opponent moves during eval)
python3 src/tic_tac_toe_bolt/train.py --force_asymmetry 0.1
```

#### Cublino Contra
```bash
# Standard training (use the helper script for correct library paths)
./run_train.sh

# Or run directly
python3 src/cublino_contra/train.py

# Resume from checkpoint
python3 src/cublino_contra/train.py --resume checkpoint_cublino.pth

# Dry run for testing
python3 src/cublino_contra/train.py --dry_run
```

*   Generates checkpoints (e.g., `current_policy_50.pth` or `current_policy_cublino_50.pth`).
*   Training scripts automatically detect if CUDA is available and utilize the C++ MCTS extension on the appropriate device.

### 3. Playing

#### Tic Tac Toe Bolt - Human vs. AI (CLI)
```bash
# Play against random/untrained
python3 src/tic_tac_toe_bolt/play.py

# Play against a specific model with difficulty adjustment (1-20, default 20)
python3 src/tic_tac_toe_bolt/play.py --model current_policy_50.pth --ai_starts --difficulty 10
```

#### Tic Tac Toe Bolt - Model vs. Model
```bash
python3 src/tic_tac_toe_bolt/play_match.py --model1 current_policy_50.pth --model2 current_policy_100.pth --n_games 100
```

#### Cublino Contra - Human vs. AI (3D OpenGL)
```bash
# Play against untrained model
python3 src/cublino_contra/play.py

# Play against a specific model
python3 src/cublino_contra/play.py --model current_policy_cublino_50.pth --difficulty 10

# AI starts first
python3 src/cublino_contra/play.py --model current_policy_cublino_50.pth --ai_starts

# Replay a logged game
python3 src/cublino_contra/play.py --replay game_log_12345.json
```

**Cublino Contra 3D Controls:**
*   **Left Mouse**: Select your dice and target squares
*   **Right Mouse Drag**: Rotate camera
*   **W/S**: Zoom in/out
*   **A/D**: Rotate camera left/right
*   **Q/E**: Tilt camera up/down
*   **Numpad 1-9**: Camera preset positions
*   **Any key**: Exit after game ends

### 4. Model Extraction

Extract the model state dictionary from a full training checkpoint:

```bash
python3 extract_model.py checkpoint.pth --output model.pth
```

### 5. Testing

Run verification scripts to ensure game logic and model integrity:

```bash
# Test Tic Tac Toe Game Logic
python3 tests/test_mechanics.py

# Test Tic Tac Toe C++ Integration
python3 tests/test_mcts_cpp.py

# Test Cublino Contra Game Logic
python3 tests/test_cublino.py

# Test Cublino Contra AI
python3 tests/test_cublino_ai.py

# Test Cublino Contra C++ Integration
python3 tests/test_cublino_mcts_cpp.py

# Benchmark Performance (Python vs C++ MCTS)
python3 tests/performance_benchmark.py
```

## Implementation Details

### Tic Tac Toe Bolt
*   **State Representation**: The board is represented as a 3x3x3 tensor (Channel 0: Player Marks, Channel 1: Opponent Marks, Channel 2: All 1s for bias).
*   **Action Space**: 9 discrete actions (one per board cell).

### Cublino Contra
*   **State Representation**: The board is a 7x7x12 tensor (4 stacked historical states × 3 channels). Each state has:
    *   Channel 0: Player occupancy (1 for P1, -1 for P2, 0 for empty)
    *   Channel 1: Dice top value (1-6)
    *   Channel 2: Dice south face value (1-6)
*   **Action Space**: 196 discrete actions (7×7 grid positions × 4 directions).
*   **Dice Mechanics**: Dice roll when moved, with faces calculated using cross-product orientation logic.
*   **Battle Resolution**: When a die becomes surrounded by 2+ opponent dice, battles are resolved by comparing the sum of top values.

### C++ Extensions
*   Both games include optimized C++ MCTS modules built from source files in their respective `mcts_cpp/` directories using `torch.utils.cpp_extension`.
*   **Device Support**: The C++ MCTS classes are device-aware, accepting a device string (e.g., "cuda" or "cpu") during initialization.
*   **Model Loading**: Play scripts gracefully handle missing models by falling back to random/untrained behavior with appropriate warnings.
