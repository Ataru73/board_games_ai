import time
import torch
import gymnasium as gym
import numpy as np
from tic_tac_toe_bolt.model import PolicyValueNet
from tic_tac_toe_bolt.mcts import MCTS, MCTS_CPP

def benchmark_mcts():
    # Setup
    env = gym.make("TicTacToeBolt-v0")
    env.reset()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = PolicyValueNet().to(device)
    model.eval()
    
    # Save model for C++ MCTS
    model_path = "temp_benchmark_model.pt"
    script_model = torch.jit.script(model)
    script_model.save(model_path)
    
    n_playout = 400
    c_puct = 5
    
    # Python MCTS wrapper to match signature if needed, or just use class directly
    # The Python MCTS class takes a policy_value_fn
    def policy_value_fn(env):
        board = env.unwrapped.board
        legal_positions = []
        for i in range(9):
            if board[i // 3, i % 3] == 0:
                legal_positions.append(i)
        
        current_player = env.unwrapped.current_player
        canonical_board = board * current_player
        
        input_board = np.zeros((1, 3, 3, 3))
        input_board[0, 0, :, :] = (canonical_board == 1)
        input_board[0, 1, :, :] = (canonical_board == -1)
        input_board[0, 2, :, :] = 1.0
        
        input_tensor = torch.FloatTensor(input_board).to(device)
        
        with torch.no_grad():
            log_act_probs, value = model(input_tensor)
            act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
            
        return zip(legal_positions, act_probs[legal_positions]), value.item()

    mcts_py = MCTS(policy_value_fn, c_puct=c_puct, n_playout=n_playout)
    
    try:
        mcts_cpp = MCTS_CPP(model_path, c_puct=c_puct, n_playout=n_playout, device=device)
    except Exception as e:
        print(f"Failed to load C++ MCTS: {e}")
        return

    num_iterations = 20
    print(f"\nRunning benchmark with {n_playout} playouts per move, over {num_iterations} iterations...")

    # Warmup
    print("Warming up...")
    mcts_py.get_move_probs(env, temp=1.0)
    mcts_cpp.get_move_probs(env, temp=1.0)
    
    # Benchmark Python
    print("Benchmarking Python MCTS...")
    start_time = time.time()
    for _ in range(num_iterations):
        mcts_py.get_move_probs(env, temp=1.0)
        # We don't update root to keep the tree small/consistent for benchmark or just reset?
        # If we don't update, the tree grows? No, MCTS starts from root each time get_move_probs is called 
        # unless we explicitly move the root. 
        # Wait, MCTS implementation provided usually rebuilds or keeps?
        # The provided Python `MCTS` class: `self._root = TreeNode(None, 1.0)` is in `__init__`.
        # `get_move_probs` calls `_playout` `n_playout` times.
        # It keeps the tree state.
        # To strictly measure speed of `get_move_probs` from scratch, we should probably reset the tree or just accept it grows.
        # For fairness, let's just let it run. It mimics real play.
    py_time = time.time() - start_time
    
    # Benchmark C++
    print("Benchmarking C++ MCTS...")
    start_time = time.time()
    for _ in range(num_iterations):
        mcts_cpp.get_move_probs(env, temp=1.0)
    cpp_time = time.time() - start_time
    
    print("\nResults:")
    print(f"Python MCTS Total Time: {py_time:.4f}s")
    print(f"Python MCTS Avg Time:   {py_time/num_iterations:.4f}s")
    print(f"C++ MCTS Total Time:    {cpp_time:.4f}s")
    print(f"C++ MCTS Avg Time:      {cpp_time/num_iterations:.4f}s")
    print(f"Speedup:                {py_time/cpp_time:.2f}x")

if __name__ == "__main__":
    benchmark_mcts()
