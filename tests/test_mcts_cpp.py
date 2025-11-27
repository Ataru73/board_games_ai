import unittest
import torch
import numpy as np
import gymnasium as gym
import os
from tic_tac_toe_bolt.mcts import MCTS_CPP, MCTS
from tic_tac_toe_bolt.model import PolicyValueNet
import tic_tac_toe_bolt

class TestMCTSCpp(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("TicTacToeBolt-v0")
        self.model = PolicyValueNet()
        self.model_path = "test_model.pt"
        
        # Save model as TorchScript
        script_model = torch.jit.script(self.model)
        script_model.save(self.model_path)
        
    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_mcts_cpp_initialization(self):
        mcts = MCTS_CPP(self.model_path, c_puct=5, n_playout=50)
        self.assertIsNotNone(mcts)

    def test_get_move_probs(self):
        mcts = MCTS_CPP(self.model_path, c_puct=5, n_playout=50)
        self.env.reset()
        
        acts, probs = mcts.get_move_probs(self.env, temp=1.0)
        
        self.assertEqual(len(acts), 9) # All moves valid initially
        self.assertEqual(len(probs), 9)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        
    def test_update_with_move(self):
        mcts = MCTS_CPP(self.model_path, c_puct=5, n_playout=50)
        self.env.reset()
        
        # Play a move
        mcts.update_with_move(4) # Center
        
        # Check if it runs without error
        acts, probs = mcts.get_move_probs(self.env, temp=1.0)
        self.assertTrue(len(acts) > 0)

if __name__ == "__main__":
    unittest.main()
