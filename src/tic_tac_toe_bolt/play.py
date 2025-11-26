import gymnasium as gym
import torch
import numpy as np
import pygame
import argparse
import sys

from tic_tac_toe_bolt.model import PolicyValueNet
from tic_tac_toe_bolt.mcts import MCTS
import tic_tac_toe_bolt # Register env

class HumanPlayer:
    def __init__(self):
        self.player = None
    
    def set_player_ind(self, p):
        self.player = p

    def get_action(self, env):
        # Wait for mouse click
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    # Map to grid
                    # Window size is 512
                    cell_size = 512 / 3
                    col = int(x // cell_size)
                    row = int(y // cell_size)
                    action = row * 3 + col
                    
                    # Check if valid (not occupied)
                    if env.unwrapped.board[row, col] == 0:
                        return action
            
            env.render()

class MCTSPlayer:
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, env):
        sensible_moves = [i for i in range(9) if env.unwrapped.board[i//3, i%3] == 0]
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, temp=1e-3)
            # Choose action with highest prob
            move = acts[np.argmax(probs)]
            self.mcts.update_with_move(-1) # Reset MCTS tree for next move? 
            # Actually, we should keep the tree if we want to reuse it, but we need to update it with the move made.
            # But here we are re-creating MCTS or resetting it?
            # In AlphaZero, we usually reuse the tree.
            # But for simplicity here, let's just reset or create new one.
            # My MCTS implementation has update_with_move.
            # But here I am calling get_move_probs which runs playouts.
            # If I want to reuse, I should update with the opponent's move too.
            # For now, let's just run fresh MCTS each time for simplicity.
            return move
        else:
            print("WARNING: No sensible moves found!")
            return -1

def run_game(model_path=None, human_starts=True):
    env = gym.make("TicTacToeBolt-v0", render_mode="human")
    env.reset()
    env.render() # Init pygame
    
    # Load model
    if model_path:
        policy_value_net = PolicyValueNet()
        policy_value_net.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        policy_value_net = PolicyValueNet()
        print("Using untrained model")
        
    def policy_value_fn(env):
        board = env.unwrapped.board
        current_player = env.unwrapped.current_player
        canonical_board = board * current_player
        
        input_board = np.zeros((1, 3, 3, 3))
        input_board[0, 0, :, :] = (canonical_board == 1)
        input_board[0, 1, :, :] = (canonical_board == -1)
        input_board[0, 2, :, :] = 1.0
        
        input_tensor = torch.FloatTensor(input_board)
        
        legal_positions = []
        for i in range(9):
            if board[i // 3, i % 3] == 0:
                legal_positions.append(i)

        with torch.no_grad():
            log_act_probs, value = policy_value_net(input_tensor)
            act_probs = np.exp(log_act_probs.numpy().flatten())
            
        return zip(legal_positions, act_probs[legal_positions]), value.item()

    # Players
    human = HumanPlayer()
    ai = MCTSPlayer(policy_value_fn, c_puct=5, n_playout=400)
    
    players = {1: human, -1: ai}
    if not human_starts:
        players = {1: ai, -1: human}
        
    obs, info = env.reset()
    done = False
    
    while not done:
        current_player_idx = env.unwrapped.current_player
        player = players[current_player_idx]
        
        if isinstance(player, HumanPlayer):
            print("Your turn!")
        else:
            print("AI is thinking...")
        
        action = player.get_action(env)
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated:
            if isinstance(player, HumanPlayer):
                print("You win!")
            else:
                print("AI wins!")
            done = True
            pygame.time.wait(3000)
        elif truncated:
            print("Draw!")
            done = True
            pygame.time.wait(3000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model .pth file")
    parser.add_argument("--ai_starts", action="store_true", help="If set, AI starts first")
    args = parser.parse_args()
    
    run_game(model_path=args.model, human_starts=not args.ai_starts)
