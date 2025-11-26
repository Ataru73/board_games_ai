import gymnasium as gym
import tic_tac_toe_bolt
import time

def run_random_agent():
    env = gym.make("TicTacToeBolt-v0", render_mode="human")
    observation, info = env.reset()
    
    for _ in range(20):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            observation, info = env.reset()
            
    env.close()

if __name__ == "__main__":
    run_random_agent()
