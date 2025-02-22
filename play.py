from pyboy import PyBoy
from pyboy.utils import WindowEvent
import time
import numpy as np
import gym
from stable_baselines3 import PPO

# Memory addresses for Super Mario Land
MARIO_X_POS = 0xC202
MARIO_Y_POS = 0xC201
LIVES = 0xDA15
COINS = 0xDA1D

# Helper functions
def get_mario_position(pyboy):
    return pyboy.memory[MARIO_X_POS], pyboy.memory[MARIO_Y_POS]

def get_lives(pyboy):
    return pyboy.memory[LIVES]

def get_coins(pyboy):
    return pyboy.memory[COINS]

def move_right(pyboy):
    pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)

def stop_moving(pyboy):
    pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)

def jump(pyboy):
    pyboy.send_input(WindowEvent.PRESS_BUTTON_A)

def stop_jumping(pyboy):
    pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)

# Simple AI logic
def simple_ai(pyboy):
    mario_x, mario_y = get_mario_position(pyboy)
    move_right(pyboy)
    
    # Jump if Mario is on the ground
    if mario_y >= 100:  # Adjust this threshold based on testing
        jump(pyboy)
        for _ in range(10):  # Hold jump for 10 frames
            pyboy.tick()
        stop_jumping(pyboy)

# Custom Gym environment
class MarioEnv(gym.Env):
    def __init__(self, rom_path, render=False):
        super(MarioEnv, self).__init__()
        self.rom_path = rom_path
        self.render = render
        self.pyboy = PyBoy(rom_path, window="SDL2" if render else "null")
        self.action_space = gym.spaces.Discrete(4)  # right, jump, right+jump, nothing
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)

    def reset(self):
        self.pyboy.stop(save=False)
        self.pyboy = PyBoy(self.rom_path, window="SDL2" if self.render else "null")
        return self._get_observation()

    def step(self, action):
        # Reset inputs
        stop_moving(self.pyboy)
        stop_jumping(self.pyboy)

        # Apply action
        if action == 0:  # Right
            move_right(self.pyboy)
            for _ in range(4):
                self.pyboy.tick()
        elif action == 1:  # Jump
            jump(self.pyboy)
            for _ in range(10):  # Hold jump for 10 frames
                self.pyboy.tick()
            stop_jumping(self.pyboy)
        elif action == 2:  # Right + Jump
            move_right(self.pyboy)
            jump(self.pyboy)
            for _ in range(10):  # Hold jump for 10 frames
                self.pyboy.tick()
            stop_jumping(self.pyboy)
        elif action == 3:  # Do nothing
            for _ in range(4):
                self.pyboy.tick()

        observation = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        return observation, reward, done, {}

    def _get_observation(self):
        return self.pyboy.botsupport_manager().screen().screen_ndarray()

    def _get_reward(self):
        mario_x, _ = get_mario_position(self.pyboy)
        return mario_x

    def _is_done(self):
        return get_lives(self.pyboy) == 0

    def render(self, mode='human'):
        if self.render and mode == 'human':
            self.pyboy.botsupport_manager().screen().screen_ndarray()
        elif mode == 'rgb_array':
            return self._get_observation()

    def close(self):
        self.pyboy.stop()

# Main function for simple AI
def main():
    pyboy = PyBoy('SuperMarioLand.gb', window="SDL2")
    target_fps = 60
    frame_time = 1 / target_fps
    last_time = time.time()

    try:
        while True:
            if not pyboy.tick():
                break

            simple_ai(pyboy)

            # Dynamic timing
            current_time = time.time()
            elapsed = current_time - last_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last_time = time.time()

            # Print state
            mario_x, mario_y = get_mario_position(pyboy)
            lives = get_lives(pyboy)
            coins = get_coins(pyboy)
            print(f"Mario Position: ({mario_x}, {mario_y}), Lives: {lives}, Coins: {coins}")

    finally:
        pyboy.stop()

# Train RL agent
def train_rl_agent():
    env = MarioEnv('SuperMarioLand.gb', render=False)
    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save(r"C:\Users\figo2\work\pyboy\mario_ppo_model")
    print("Model saved as mario_ppo_model.zip")

    # Test with rendering
    env = MarioEnv('SuperMarioLand.gb', render=True)
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()
    # train_rl_agent()