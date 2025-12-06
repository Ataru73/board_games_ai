import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Adjust the path to import from src.cublino_contra
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..' )))

from src.cublino_contra.train import run_self_play_worker
from src.cublino_contra.env import CublinoContraEnv

class TestRandomInitialization(unittest.TestCase):

    @patch('src.cublino_contra.train.MCTS_CPP')
    @patch('src.cublino_contra.train.MCTS')
    @patch('src.cublino_contra.train.PolicyValueNet')
    @patch('src.cublino_contra.train.torch')
    @patch('src.cublino_contra.train.CublinoContraEnv') # Patch the actual env for controlled behavior
    def test_random_initialization_no_errors(self, MockCublinoContraEnv, MockTorch, MockPolicyValueNet, MockMCTS, MockMCTS_CPP):
        # Configure mocks
        mock_env_instance = MockCublinoContraEnv.return_value
        mock_env_instance.reset.return_value = (None, {})
        mock_env_instance.get_legal_actions.return_value = [0, 1, 2, 3, 4, 5, 6, 7, 8] # Always return legal actions
        
        # Configure step to terminate eventually to avoid infinite loop
        def mock_step_sequence(*args, **kwargs):
            mock_step_sequence.count += 1
            if mock_step_sequence.count >= 20: # Terminate after some moves
                return (None, 1, True, False, {"winner": 1})
            return (None, 0, False, False, {})
        mock_step_sequence.count = 0
        mock_env_instance.step.side_effect = mock_step_sequence
        
        mock_env_instance.current_player = 1 # Mock current player
        mock_env_instance.board = np.zeros((7,7,3), dtype=np.int8) # Mock board
        
        # Mock MCTS to return dummy probs
        mock_mcts_instance = MockMCTS_CPP.return_value # Or MockMCTS.return_value if C++ fails
        mock_mcts_instance.get_move_probs.return_value = ([0], [1.0]) # A single action with 100% prob
        mock_mcts_instance.update_with_move.return_value = None

        # Test parameters
        model_path = "dummy_model.pth"
        c_puct = 5
        n_playout = 100
        device_str = "cpu"
        temp = 1.0
        num_games_to_play_per_worker = 2 # Reduced to 2 for speed, was 5
        draw_reward = 0.0

        # Run the worker
        all_play_data, worker_game_stats, game_log_data = run_self_play_worker(
            model_path, c_puct, n_playout, device_str, temp, num_games_to_play_per_worker, draw_reward
        )

        # Assertions
        self.assertGreater(len(all_play_data), 0) # Should have generated some game data
        self.assertGreaterEqual(mock_env_instance.reset.call_count, num_games_to_play_per_worker) # At least one reset per game

    @patch('src.cublino_contra.train.MCTS_CPP')
    @patch('src.cublino_contra.train.MCTS')
    @patch('src.cublino_contra.train.PolicyValueNet')
    @patch('src.cublino_contra.train.torch')
    @patch('src.cublino_contra.train.CublinoContraEnv')
    def test_random_initialization_with_initial_error(self, MockCublinoContraEnv, MockTorch, MockPolicyValueNet, MockMCTS, MockMCTS_CPP):
        mock_env_instance = MockCublinoContraEnv.return_value
        mock_env_instance.reset.return_value = (None, {})
        mock_env_instance.get_legal_actions.return_value = [0] # Always one legal action
        
        # Simulate an immediate error on the first step of a random move sequence
        # This should trigger a game restart attempt
        def mock_step_with_error(action):
            mock_step_with_error.call_count += 1
            if mock_step_with_error.call_count == 1: # First random move for the current attempt -> ERROR
                return (None, -1, False, True, {"error": "Invalid Move", "reason": "Test Error"})
            # Subsequent calls (retry random move, then main loop) -> OK then Terminate
            if mock_step_with_error.call_count > 10:
                return (None, 1, True, False, {"winner": 1})
            return (None, 0, False, False, {})
            
        mock_step_with_error.call_count = 0
        mock_env_instance.step.side_effect = mock_step_with_error
        mock_env_instance.current_player = 1
        mock_env_instance.board = np.zeros((7,7,3), dtype=np.int8)

        mock_mcts_instance = MockMCTS_CPP.return_value
        mock_mcts_instance.get_move_probs.return_value = ([0], [1.0])
        mock_mcts_instance.update_with_move.return_value = None

        model_path = "dummy_model.pth"
        c_puct = 5
        n_playout = 100
        device_str = "cpu"
        temp = 1.0
        num_games_to_play_per_worker = 1 # Only one game needed for this test
        draw_reward = 0.0
        
        # Patch np.random.randint to always return a fixed number of random moves (e.g., 1)
        with patch('numpy.random.randint', return_value=1) as mock_randint:
            all_play_data, worker_game_stats, game_log_data = run_self_play_worker(
                model_path, c_puct, n_playout, device_str, temp, num_games_to_play_per_worker, draw_reward
            )
        
        # We expect:
        # 1. First reset.
        # 2. Random move attempt 1 -> Error.
        # 3. Second reset.
        # 4. Random move attempt 2 -> Success.
        # 5. Main loop -> Success -> Termination.
        
        self.assertGreater(mock_env_instance.reset.call_count, 1)
        
        # The final state should not have any errors
        self.assertGreater(len(all_play_data), 0)

    @patch('src.cublino_contra.train.MCTS_CPP')
    @patch('src.cublino_contra.train.MCTS')
    @patch('src.cublino_contra.train.PolicyValueNet')
    @patch('src.cublino_contra.train.torch')
    @patch('src.cublino_contra.train.CublinoContraEnv')
    def test_random_initialization_with_termination(self, MockCublinoContraEnv, MockTorch, MockPolicyValueNet, MockMCTS, MockMCTS_CPP):
        mock_env_instance = MockCublinoContraEnv.return_value
        mock_env_instance.reset.return_value = (None, {})
        mock_env_instance.get_legal_actions.return_value = [0] # Always one legal action
        
        # Simulate an immediate termination on the first step of a random move sequence
        def mock_step_with_termination(action):
            mock_step_with_termination.call_count += 1
            if mock_step_with_termination.call_count == 1: 
                # First attempt: Terminate during random moves
                return (None, 1, True, False, {"winner": 1}) 
            # Second attempt: Success random move
            # Main loop: Terminate eventually
            if mock_step_with_termination.call_count > 5:
                return (None, 1, True, False, {"winner": 1})
            return (None, 0, False, False, {})
            
        mock_step_with_termination.call_count = 0
        mock_env_instance.step.side_effect = mock_step_with_termination
        mock_env_instance.current_player = 1
        mock_env_instance.board = np.zeros((7,7,3), dtype=np.int8)

        mock_mcts_instance = MockMCTS_CPP.return_value
        mock_mcts_instance.get_move_probs.return_value = ([0], [1.0])
        mock_mcts_instance.update_with_move.return_value = None

        model_path = "dummy_model.pth"
        c_puct = 5
        n_playout = 100
        device_str = "cpu"
        temp = 1.0
        num_games_to_play_per_worker = 1 # Only one game needed for this test
        draw_reward = 0.0
        
        with patch('numpy.random.randint', return_value=1) as mock_randint:
            all_play_data, worker_game_stats, game_log_data = run_self_play_worker(
                model_path, c_puct, n_playout, device_str, temp, num_games_to_play_per_worker, draw_reward
            )
        
        # Expectation:
        # 1. Reset 1.
        # 2. Step 1 -> Terminate -> Abort.
        # 3. Reset 2.
        # 4. Step 2 -> Success.
        # 5. Main loop -> ... -> Terminate.
        
        self.assertEqual(mock_env_instance.reset.call_count, 2) 
        self.assertEqual(len(all_play_data), 1) # One valid game collected
        self.assertEqual(worker_game_stats[1], 1) # Player 1 should have won

if __name__ == '__main__':
    unittest.main()