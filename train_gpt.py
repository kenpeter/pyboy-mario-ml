from pyboy import PyBoy
from pyboy.utils import WindowEvent
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Categorical

# Memory addresses for Super Mario Land
MARIO_X_POS = 0xC202  # X-coordinate of Mario
MARIO_Y_POS = 0xC201  # Y-coordinate of Mario
LIVES = 0xDA15  # Number of lives remaining
COINS = 0xFFFA  # Coin counter (BCD format)
WORLD_LEVEL = 0xFFB4  # Current world and level (e.g., 0x11 = 1-1)

def get_mario_position(pyboy):
    """Return Mario's current (x, y) position from memory."""
    return pyboy.memory[MARIO_X_POS], pyboy.memory[MARIO_Y_POS]

def get_lives(pyboy):
    """Return the number of lives remaining."""
    return pyboy.memory[LIVES]

def get_coins(pyboy):
    """Return the current coin count."""
    return pyboy.memory[COINS]

def get_world_level(pyboy):
    """Return the current world and level."""
    return pyboy.memory[WORLD_LEVEL]

def move_right(pyboy):
    """Move Mario right."""
    pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)

def stop_moving(pyboy):
    """Stop Mario's right or left movement."""
    pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
    pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)

def move_left(pyboy):
    """Move Mario left."""
    pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)

def stop_left(pyboy):
    """Stop Mario's left movement."""
    pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)

def jump(pyboy):
    """Make Mario jump."""
    pyboy.send_input(WindowEvent.PRESS_BUTTON_A)

def stop_jumping(pyboy):
    """Stop Mario's jump."""
    pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)

def press_start(pyboy):
    """Press and release the Start button to begin the game."""
    pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
    for _ in range(60):
        pyboy.tick()
    pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)

def enable_turbo_mode(pyboy):
    """Enable turbo mode by pressing the space bar."""
    pyboy.send_input(WindowEvent.PRESS_SPEED_UP)
    pyboy.tick()  # Ensure the event is processed

def disable_turbo_mode(pyboy):
    """Disable turbo mode by releasing the space bar."""
    pyboy.send_input(WindowEvent.RELEASE_SPEED_UP)
    pyboy.tick()  # Ensure the event is processed

class MarioEnv(gym.Env):
    def __init__(self, rom_path, render=False):
        """Initialize the Mario environment with ROM path and render option."""
        super(MarioEnv, self).__init__()
        self.rom_path = rom_path
        self.render_enabled = render
        self.pyboy = PyBoy(rom_path, window="SDL2" if render else "null")
        self.action_space = gym.spaces.Discrete(5)  # 5 actions: idle, right, jump, right+jump, left
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        
        # Initialize the game
        press_start(self.pyboy)
        for i in range(600):
            self.pyboy.tick()
            if i == 300:
                press_start(self.pyboy)
            if get_lives(self.pyboy) > 0:
                break

        # Delay turbo mode activation
        if self.render_enabled:
            time.sleep(5)  # Wait 5 seconds before enabling turbo mode
            enable_turbo_mode(self.pyboy)

        self.prev_coins = get_coins(self.pyboy)
        self.prev_x = get_mario_position(self.pyboy)[0]
        self.prev_y = get_mario_position(self.pyboy)[1]
        self.prev_lives = get_lives(self.pyboy)
        self.prev_world = get_world_level(self.pyboy)
        self.steps = 0

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        self.pyboy.stop(save=False)
        self.pyboy = PyBoy(self.rom_path, window="SDL2" if self.render_enabled else "null")
        
        # Initialize the game
        press_start(self.pyboy)
        for i in range(600):
            self.pyboy.tick()
            if i == 300:
                press_start(self.pyboy)
            if get_lives(self.pyboy) > 0:
                break

        # Delay turbo mode activation
        if self.render_enabled:
            time.sleep(5)  # Wait 5 seconds before enabling turbo mode
            enable_turbo_mode(self.pyboy)

        self.prev_coins = get_coins(self.pyboy)
        self.prev_x = get_mario_position(self.pyboy)[0]
        self.prev_y = get_mario_position(self.pyboy)[1]
        self.prev_lives = get_lives(self.pyboy)
        self.prev_world = get_world_level(self.pyboy)
        self.steps = 0
        observation = self._get_observation()
        info = {"steps": self.steps}  # Additional info
        return observation, info  # Gymnasium requires returning observation and info

    def step(self, action):
        """Execute one step with the given action."""
        stop_moving(self.pyboy)
        stop_jumping(self.pyboy)

        if action == 0:  # Idle
            for _ in range(4):
                self.pyboy.tick()
        elif action == 1:  # Move right
            move_right(self.pyboy)
            for _ in range(4):
                self.pyboy.tick()
        elif action == 2:  # Jump
            jump(self.pyboy)
            for _ in range(20):
                self.pyboy.tick()
            stop_jumping(self.pyboy)
        elif action == 3:  # Move right and jump
            move_right(self.pyboy)
            jump(self.pyboy)
            for _ in range(20):
                self.pyboy.tick()
            stop_jumping(self.pyboy)
        elif action == 4:  # Move left
            move_left(self.pyboy)
            for _ in range(4):
                self.pyboy.tick()
            stop_left(self.pyboy)

        self.steps += 1
        observation = self._get_observation()
        reward = self._get_reward()
        terminated = self._is_done()  # Gymnasium uses "terminated" instead of "done"
        truncated = False  # Gymnasium requires a "truncated" flag (e.g., for time limits)
        info = {"steps": self.steps}  # Additional info
        self.prev_coins = get_coins(self.pyboy)
        self.prev_x = get_mario_position(self.pyboy)[0]
        self.prev_y = get_mario_position(self.pyboy)[1]
        self.prev_lives = get_lives(self.pyboy)
        self.prev_world = get_world_level(self.pyboy)
        return observation, reward, terminated, truncated, info  # Gymnasium requires 5 return values

    def _get_observation(self):
        """Get the current screen observation."""
        screen_image = np.array(self.pyboy.screen.image)
        return screen_image[:, :, :3] if screen_image.shape[-1] == 4 else screen_image

    def _get_reward(self):
        """Calculate the reward based on current state."""
        mario_x, mario_y = get_mario_position(self.pyboy)
        current_coins = get_coins(self.pyboy)
        current_lives = get_lives(self.pyboy)
        current_world = get_world_level(self.pyboy)

        # Reward for progress (moving right)
        progress_reward = (mario_x - self.prev_x) * 2  # Reduced from 5 to 2

        # Penalty for moving left
        if mario_x < self.prev_x:
            progress_reward = -1  # Reduced from -5 to -1

        # Reward for collecting coins
        coin_reward = (current_coins - self.prev_coins) * 5  # Reduced from 20 to 5

        # Penalty for losing a life
        death_penalty = -50 if current_lives < self.prev_lives else 0  # Reduced from -500 to -50

        # Reward for surviving
        survival_reward = 0.1  # Reduced from 1 to 0.1

        # Reward for completing the stage
        stage_complete = 100 if current_world > self.prev_world else 0  # Reduced from 500 to 100

        # Reward for jumping (even if no enemy is present)
        jump_reward = 5 if mario_y < self.prev_y else 0  # Reward any jump

        # Total reward (normalized)
        total_reward = (
            progress_reward +
            coin_reward +
            survival_reward +
            death_penalty +
            stage_complete +
            jump_reward
        ) / 100  # Normalize rewards

        return total_reward
        

    def _is_done(self):
        """Check if the episode is done."""
        return get_lives(self.pyboy) == 0

    def render(self, mode='human'):
        """Render the game environment."""
        if self.render_enabled:
            self.pyboy.tick()

    def close(self):
        """Close the environment."""
        self.pyboy.stop(save=False)

# Custom neural network policy class (modified)
class CustomMarioPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMarioPolicy, self).__init__(*args, **kwargs)
        self.latent_pi = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_space.n)
        )
        self.latent_vf = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _build_mlp_extractor(self):
        pass

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        pi = self.latent_pi(features)
        vf = self.latent_vf(features)
        dist = Categorical(logits=pi)
        if deterministic:
            actions = dist.probs.argmax(dim=-1)
        else:
            actions = dist.sample()
        return actions, dist.log_prob(actions), dist.entropy(), vf

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        pi = self.latent_pi(features)
        vf = self.latent_vf(features)
        dist = Categorical(logits=pi)
        return dist.log_prob(actions), dist.entropy(), vf

# Custom feature extractor class
class MarioFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(MarioFeatureExtractor, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 17 * 12, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.cnn(observations / 255.0)  # Normalize pixel values to [0, 1]

# Load the ROM path
rom_path = "SuperMarioLand.gb"

# Create and train the PPO agent
env = MarioEnv(rom_path)
policy_kwargs = dict(
    features_extractor_class=MarioFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=512),
)
model = PPO(CustomMarioPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=100000)

# Test the trained model
obs, info = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        obs, info = env.reset()

env.close()
