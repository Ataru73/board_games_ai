import gymnasium as gym
import tic_tac_toe_bolt
import numpy as np

def test_infinite_mechanic():
    env = gym.make("TicTacToeBolt-v0")
    env.reset()
    
    # Player 1 moves
    moves_p1 = [(0, 0), (0, 1), (0, 2)]
    # Player 2 moves
    moves_p2 = [(1, 0), (1, 1), (1, 2)]
    
    # Interleave moves to avoid winning immediately if possible, 
    # but for simplicity let's just place them such that no one wins yet.
    # P1: (0,0), P2: (1,0), P1: (0,1), P2: (1,1), P1: (0,2) -> P1 wins?
    # Wait, (0,0), (0,1), (0,2) is a win for P1.
    # Let's use a non-winning sequence.
    
    # P1: (0,0)
    # P2: (1,0)
    # P1: (0,1)
    # P2: (1,1)
    # P1: (2,2)
    # P2: (2,0)
    
    # Board:
    # X X .
    # O O .
    # O . X
    
    actions = [
        0, # P1 (0,0)
        3, # P2 (1,0)
        1, # P1 (0,1)
        4, # P2 (1,1)
        8, # P1 (2,2)
        6, # P2 (2,0)
    ]
    
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print("Game ended unexpectedly during setup")
            return

    print("State before 4th move for P1:")
    print(env.unwrapped.board)
    
    # Now P1 places 4th mark at (2,1) -> action 7
    # Should remove oldest P1 mark at (0,0)
    obs, reward, terminated, truncated, info = env.step(7)
    
    print("State after 4th move for P1 (should have removed (0,0)):")
    print(env.unwrapped.board)
    
    assert env.unwrapped.board[0, 0] == 0, "Oldest mark (0,0) was not removed!"
    assert env.unwrapped.board[2, 1] == 1, "New mark (2,1) was not placed!"
    
    print("Infinite mechanic verified successfully!")

if __name__ == "__main__":
    test_infinite_mechanic()
