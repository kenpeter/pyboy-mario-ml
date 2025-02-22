# Import PyBoy for Game Boy emulation
from pyboy import PyBoy
# Import WindowEvent for sending controller inputs
from pyboy.utils import WindowEvent
# Import time for frame rate control
import time
# Import NumPy for array operations (screen data)
import numpy as np
# Import Gym for RL environment framework
import gym
# Import PPO from Stable Baselines3 for RL training
from stable_baselines3 import PPO

# Memory addresses for Super Mario Land (game-specific RAM locations)
MARIO_X_POS = 0xC202  # Mario's X position (horizontal)
MARIO_Y_POS = 0xC201  # Mario's Y position (vertical, lower is higher)
LIVES = 0xDA15        # Number of lives remaining
COINS = 0xDA1D        # Number of coins collected

# Helper function to get Mario's position from memory
def get_mario_position(pyboy):
    return pyboy.memory[MARIO_X_POS], pyboy.memory[MARIO_Y_POS]

# Helper function to get Mario's lives from memory
def get_lives(pyboy):
    return pyboy.memory[LIVES]

# Helper function to get Mario's coin count from memory
def get_coins(pyboy):
    return pyboy.memory[COINS]

# Helper function to press the right arrow key
def move_right(pyboy):
    pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)

# Helper function to release the right arrow key
def stop_moving(pyboy):
    pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)

# Helper function to press the A button (jump)
def jump(pyboy):
    pyboy.send_input(WindowEvent.PRESS_BUTTON_A)

# Helper function to release the A button
def stop_jumping(pyboy):
    pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)

# Improved simple AI logic for baseline behavior
def simple_ai(pyboy):
    mario_x, mario_y = get_mario_position(pyboy)
    move_right(pyboy)  # Always move right to progress
    if mario_y >= 100:  # Jump if on ground or near it
        jump(pyboy)
        for _ in range(10):  # Hold jump for 10 frames (higher jump)
            pyboy.tick()
        stop_jumping(pyboy)

# Custom Gym environment for Super Mario Land
class MarioEnv(gym.Env):
    def __init__(self, rom_path, render=False):
        super(MarioEnv, self).__init__()
        self.rom_path = rom_path
        self.render = render
        self.pyboy = PyBoy(rom_path, window=("SDL2" if render else "null"))
        self.action_space = gym.spaces.Discrete(4)  # Actions: right, jump, right+jump, nothing
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        self.prev_coins = 0
        self.prev_x = 0
        self.prev_lives = 2  # Initial lives assumption
        self.steps = 0  # Track steps per episode

    def reset(self):
        self.pyboy.stop(save=False)
        self.pyboy = PyBoy(self.rom_path, window=("SDL2" if self.render else "null"))
        self.pyboy.tick()  # Initialize screen buffer
        self.prev_coins = get_coins(self.pyboy)
        self.prev_x = get_mario_position(self.pyboy)[0]
        self.prev_lives = get_lives(self.pyboy)
        self.steps = 0
        return self._get_observation()

    def step(self, action):
        stop_moving(self.pyboy)
        stop_jumping(self.pyboy)

        if action == 0:  # Right
            move_right(self.pyboy)
            for _ in range(4):
                self.pyboy.tick()
        elif action == 1:  # Jump
            jump(self.pyboy)
            for _ in range(10):
                self.pyboy.tick()
            stop_jumping(self.pyboy)
        elif action == 2:  # Right + Jump
            move_right(self.pyboy)
            jump(self.pyboy)
            for _ in range(10):
                self.pyboy.tick()
            stop_jumping(self.pyboy)
        elif action == 3:  # Do nothing
            for _ in range(4):
                self.pyboy.tick()

        self.steps += 1
        observation = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        self.prev_coins = get_coins(self.pyboy)
        self.prev_x = get_mario_position(self.pyboy)[0]
        self.prev_lives = get_lives(self.pyboy)
        return observation, reward, done, {"steps": self.steps}

    def _get_observation(self):
        # Get screen as RGB NumPy array (remove alpha channel if present)
        screen_image = np.array(self.pyboy.screen.image)
        return screen_image[:, :, :3] if screen_image.shape[-1] == 4 else screen_image

    def _get_reward(self):
        mario_x, mario_y = get_mario_position(self.pyboy)
        current_coins = get_coins(self.pyboy)
        current_lives = get_lives(self.pyboy)

        # Improved reward structure
        progress_reward = (mario_x - self.prev_x) * 0.5  # More weight to progress
        coin_reward = (current_coins - self.prev_coins) * 20  # Higher coin incentive
        survival_reward = 1  # Reward for staying alive each step
        death_penalty = -10 if current_lives < self.prev_lives else 0  # Less harsh penalty
        jump_reward = 5 if mario_y < 80 else 0  # Encourage jumping for items
        stall_penalty = -2 if mario_x == self.prev_x and mario_y >= 100 else 0  # Milder penalty

        return progress_reward + coin_reward + survival_reward + death_penalty + jump_reward + stall_penalty

    def _is_done(self):
        # Only end episode on true game over (lives = 0)
        return get_lives(self.pyboy) == 0

    def render(self, mode='human'):
        if self.render and mode == 'human':
            pass  # SDL2 handles rendering automatically
        elif mode == 'rgb_array':
            return self._get_observation()

    def close(self):
        self.pyboy.stop()

# Main function for simple AI demo
def main():
    pyboy = PyBoy('SuperMarioLand.gb', window="SDL2")
    target_fps = 60
    frame_time = 1 / target_fps
    last_time = time.time()
    prev_lives = get_lives(pyboy)

    try:
        while True:
            if not pyboy.tick():
                break
            simple_ai(pyboy)
            current_lives = get_lives(pyboy)
            if current_lives < prev_lives:
                print(f"Mario died! Lives: {current_lives}")
            prev_lives = current_lives
            if current_lives == 0:
                print("Game Over!")
                break
            current_time = time.time()
            elapsed = current_time - last_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last_time = time.time()
            mario_x, mario_y = get_mario_position(pyboy)
            lives = get_lives(pyboy)
            coins = get_coins(pyboy)
            print(f"Mario Position: ({mario_x}, {mario_y}), Lives: {lives}, Coins: {coins}")
    finally:
        pyboy.stop()

# Improved RL training function
def train_rl_agent():
    print("=== Starting RL Training ===")
    env = MarioEnv('SuperMarioLand.gb', render=False)
    # Tuned PPO hyperparameters for better learning
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=0.0001,  # Lower for stability
        n_steps=2048,          # Steps per update
        batch_size=64,         # Batch size for training
        n_epochs=10,           # Epochs per update
        gamma=0.99,            # Discount factor
        gae_lambda=0.95,       # GAE for advantage estimation
        clip_range=0.2,        # PPO clip range
        ent_coef=0.01          # Entropy coefficient for exploration
    )
    model.learn(total_timesteps=1000)  # Increase timesteps for better learning
    model.save("mario_ppo_model")
    print("=== Training Complete. Saved 'mario_ppo_model.zip' ===")

    print("=== Starting RL Testing ===")
    env = MarioEnv('SuperMarioLand.gb', render=True)
    obs = env.reset()
    total_reward = 0
    for _ in range(2000):  # Longer test for observation
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        if done:
            print(f"Episode ended. Total Reward: {total_reward}")
            obs = env.reset()
            total_reward = 0
    env.close()
    print("=== Testing Complete ===")

if __name__ == "__main__":
    main()
    train_rl_agent()