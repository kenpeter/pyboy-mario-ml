import gymnasium as gym  # Replace OpenAI Gym with Gymnasium
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from stable_baselines3 import PPO
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Categorical

class MarioEnv(gym.Env):
    def __init__(self, rom_path, render=True):  # Default to UI
        super().__init__()
        self.rom_path = rom_path
        self.render_enabled = render
        self.pyboy = PyBoy(rom_path, window="SDL2" if render else "null")
        self.action_space = gym.spaces.Discrete(5)  # 5 actions: idle, right, jump, right+jump, left
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        self._start_game()
        self.prev_coins = self.pyboy.memory[0xFFFA]
        self.prev_x = self.pyboy.memory[0xC202]
        self.prev_lives = self.pyboy.memory[0xDA15]
        self.steps = 0

    def _start_game(self):
        """Press Start to begin the game."""
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        for _ in range(60):
            self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
        for i in range(600):
            self.pyboy.tick()
            if i == 300:
                self._start_game()
            if self.pyboy.memory[0xDA15] > 0:
                break

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        self.pyboy.stop(save=False)
        self.pyboy = PyBoy(self.rom_path, window="SDL2" if self.render_enabled else "null")
        self._start_game()
        self.prev_coins = self.pyboy.memory[0xFFFA]
        self.prev_x = self.pyboy.memory[0xC202]
        self.prev_lives = self.pyboy.memory[0xDA15]
        self.steps = 0
        observation = np.array(self.pyboy.screen.image)[:, :, :3]
        info = {"steps": self.steps}  # Additional info
        return observation, info  # Gymnasium requires returning observation and info

    def step(self, action):
        """Execute one step with the given action."""
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        if action == 0:  # Idle
            for _ in range(4):
                self.pyboy.tick()
        elif action == 1:  # Move right
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            for _ in range(4):
                self.pyboy.tick()
        elif action == 2:  # Jump
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            for _ in range(10):
                self.pyboy.tick()
        elif action == 3:  # Move right and jump
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            for _ in range(10):
                self.pyboy.tick()
        elif action == 4:  # Move left
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
            for _ in range(4):
                self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)

        self.steps += 1
        observation = np.array(self.pyboy.screen.image)[:, :, :3]
        reward = self._get_reward()
        terminated = self.pyboy.memory[0xDA15] == 0  # Gymnasium uses "terminated" instead of "done"
        truncated = False  # Gymnasium requires a "truncated" flag (e.g., for time limits)
        info = {"steps": self.steps}  # Additional info
        self.prev_coins = self.pyboy.memory[0xFFFA]
        self.prev_x = self.pyboy.memory[0xC202]
        self.prev_lives = self.pyboy.memory[0xDA15]
        return observation, reward, terminated, truncated, info  # Gymnasium requires 5 return values

    def _get_reward(self):
        """Calculate the reward based on current state."""
        x, y = self.pyboy.memory[0xC202], self.pyboy.memory[0xC201]
        coins, lives = self.pyboy.memory[0xFFFA], self.pyboy.memory[0xDA15]
        progress_reward = (x - self.prev_x) * 2  # Higher reward for progress
        coin_reward = (coins - self.prev_coins) * 50
        survival_reward = 5
        death_penalty = -100 if lives < self.prev_lives else 0
        jump_reward = 10 if y < 80 else 0
        stall_penalty = -1 if x == self.prev_x and y >= 100 else 0
        return progress_reward + coin_reward + survival_reward + death_penalty + jump_reward + stall_penalty

    def render(self):
        pass  # SDL2 handles rendering automatically

    def close(self):
        """Clean up the environment."""
        self.pyboy.stop()

def play(model_path="mario_ppo_model_improved.zip"):
    """Load the trained model and play the game."""
    env = MarioEnv('SuperMarioLand.gb', render=True)
    model = PPO.load(model_path)
    print("Model loaded successfully!")  # Debug print
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(2000):
        action, _ = model.predict(obs)
        print(f"Action taken: {action}")  # Debug print
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()  # Shows the game UI
        if terminated or truncated:
            print(f"Episode ended. Total Reward: {total_reward}, Steps: {info['steps']}")
            obs, _ = env.reset()
            total_reward = 0
    env.close()

if __name__ == "__main__":
    play()