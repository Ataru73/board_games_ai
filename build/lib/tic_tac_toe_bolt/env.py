import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class TicTacToeBoltEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.window_size = 512  # The size of the PyGame window
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.int8)
        self.action_space = spaces.Discrete(9)
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Game state
        self.board = None
        self.current_player = 1 # 1 or -1
        self.player_moves = {1: [], -1: []} # Track moves for "Infinite" mechanic

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1
        self.player_moves = {1: [], -1: []}
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map action (0-8) to (row, col)
        row = action // 3
        col = action % 3

        # Check if move is valid
        if self.board[row, col] != 0:
            # Invalid move: return same state, no reward (or negative reward if desired), not done
            # For this implementation, let's give a small negative reward for invalid move and continue
            # Or we can just ignore it. Let's return a large negative reward and end game? 
            # Or just ignore. Standard gym practice varies. 
            # Let's treat it as an invalid move that doesn't change state but gives penalty?
            # User didn't specify invalid move handling. 
            # Let's assume the agent should learn not to do it.
            # We will return a large negative reward and terminate to speed up learning valid moves?
            # Or just -10 and continue. Let's do -10 and continue.
            return self._get_obs(), -10, False, False, self._get_info()

        # "Infinite" mechanic: Check if player has 3 marks
        if len(self.player_moves[self.current_player]) == 3:
            # Remove oldest mark
            old_r, old_c = self.player_moves[self.current_player].pop(0)
            self.board[old_r, old_c] = 0

        # Place new mark
        self.board[row, col] = self.current_player
        self.player_moves[self.current_player].append((row, col))

        # Check for win
        if self._check_win(self.current_player):
            reward = 1
            terminated = True
            winner = self.current_player
        else:
            reward = 0
            terminated = False
            winner = 0
        
        # Switch player
        self.current_player *= -1

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_obs(self):
        return self.board.copy()

    def _get_info(self):
        return {"current_player": self.current_player}

    def _check_win(self, player):
        # Check rows, cols, diagonals
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        
        pix_square_size = (self.window_size / 3)

        # Draw gridlines
        for x in range(self.window_size + 1):
            if x % int(pix_square_size) == 0:
                pygame.draw.line(
                    canvas,
                    0,
                    (0, x),
                    (self.window_size, x),
                    width=3,
                )
                pygame.draw.line(
                    canvas,
                    0,
                    (x, 0),
                    (x, self.window_size),
                    width=3,
                )

        # Draw marks
        for r in range(3):
            for c in range(3):
                if self.board[r, c] == 1: # X
                    pygame.draw.line(
                        canvas,
                        (255, 0, 0),
                        (c * pix_square_size + 20, r * pix_square_size + 20),
                        ((c + 1) * pix_square_size - 20, (r + 1) * pix_square_size - 20),
                        width=8,
                    )
                    pygame.draw.line(
                        canvas,
                        (255, 0, 0),
                        ((c + 1) * pix_square_size - 20, r * pix_square_size + 20),
                        (c * pix_square_size + 20, (r + 1) * pix_square_size - 20),
                        width=8,
                    )
                elif self.board[r, c] == -1: # O
                    pygame.draw.circle(
                        canvas,
                        (0, 0, 255),
                        (int((c + 0.5) * pix_square_size), int((r + 0.5) * pix_square_size)),
                        int(pix_square_size / 3),
                        width=8,
                    )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
