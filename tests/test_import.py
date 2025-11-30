
import sys
import os
sys.path.append(os.getcwd())
try:
    from src.cublino_contra import _mcts_cpp
    print("Import successful")
    print(dir(_mcts_cpp))
except ImportError as e:
    print(f"Import failed: {e}")
