import gym
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from stable_baselines3 import PPO

class MarioEnv(gym.Env):
    def __init__(self, rom_path, render=True):  # Default to UI
        super().__init__()
        self.rom_path = rom_path
        self.render_enabled = render
        self.pyboy = PyBoy(rom_path, window="SDL2" if render else "null")
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        self._start_game()
        self.prev_coins = self.pyboy.memory[0xFFFA]
        self.prev_x = self.pyboy.memory[0xC202]
        self.prev_lives = self.pyboy.memory[0xDA15]
        self.steps = 0

    def _start_game(self):
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        for _ in range(60): self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
        for i in range(600):
            self.pyboy.tick()
            if i == 300: self._start_game()
            if self.pyboy.memory[0xDA15] > 0: break

    def reset(self):
        self.pyboy.stop(save=False)
        self.pyboy = PyBoy(self.rom_path, window="SDL2" if self.render_enabled else "null")
        self._start_game()
        self.prev_coins = self.pyboy.memory[0xFFFA]
        self.prev_x = self.pyboy.memory[0xC202]
        self.prev_lives = self.pyboy.memory[0xDA15]
        self.steps = 0
        return np.array(self.pyboy.screen.image)[:, :, :3]

    def step(self, action):
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        if action == 0:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            for _ in range(4): self.pyboy.tick()
        elif action == 1:
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            for _ in range(10): self.pyboy.tick()
        elif action == 2:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            for _ in range(10): self.pyboy.tick()
        elif action == 3:
            for _ in range(4): self.pyboy.tick()
        self.steps += 1
        obs = np.array(self.pyboy.screen.image)[:, :, :3]
        reward = self._get_reward()
        done = self.pyboy.memory[0xDA15] == 0
        self.prev_coins = self.pyboy.memory[0xFFFA]
        self.prev_x = self.pyboy.memory[0xC202]
        self.prev_lives = self.pyboy.memory[0xDA15]
        return obs, reward, done, {"steps": self.steps}

    def _get_reward(self):
        x, y = self.pyboy.memory[0xC202], self.pyboy.memory[0xC201]
        coins, lives = self.pyboy.memory[0xFFFA], self.pyboy.memory[0xDA15]
        return (x - self.prev_x) * 1 + (coins - self.prev_coins) * 50 + 1 + (-10 if lives < self.prev_lives else 0) + (5 if y < 80 else 0) + (-2 if x == self.prev_x and y >= 100 else 0)

    def render(self):
        pass  # SDL2 handles rendering automatically

    def close(self):
        self.pyboy.stop()

def play(model_path="mario_ppo_model_improved.zip"):
    env = MarioEnv('SuperMarioLand.gb', render=True)
    model = PPO.load(model_path)
    obs = env.reset()
    total_reward = 0
    for _ in range(2000):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()  # Shows the game UI
        if done:
            print(f"Episode ended. Total Reward: {total_reward}, Steps: {info['steps']}")
            obs = env.reset()
            total_reward = 0
    env.close()

if __name__ == "__main__":
    play()