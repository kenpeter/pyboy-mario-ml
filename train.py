from pyboy import PyBoy
from pyboy.utils import WindowEvent
import time
import numpy as np
import gym
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

class MarioEnv(gym.Env):
    def __init__(self, rom_path, render=False):
        """Initialize the Mario environment with ROM path and render option."""
        super(MarioEnv, self).__init__()
        self.rom_path = rom_path
        self.render_enabled = render
        self.pyboy = PyBoy(rom_path, window="SDL2" if render else "null")
        self.action_space = gym.spaces.Discrete(5)  # 5 actions: idle, right, jump, right+jump, left
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        press_start(self.pyboy)
        for i in range(600):
            self.pyboy.tick()
            if i == 300:
                press_start(self.pyboy)
            if get_lives(self.pyboy) > 0:
                break
        self.prev_coins = get_coins(self.pyboy)
        self.prev_x = get_mario_position(self.pyboy)[0]
        self.prev_y = get_mario_position(self.pyboy)[1]
        self.prev_lives = get_lives(self.pyboy)
        self.prev_world = get_world_level(self.pyboy)
        self.steps = 0

    def reset(self):
        """Reset the environment to the initial state."""
        self.pyboy.stop(save=False)
        self.pyboy = PyBoy(self.rom_path, window="SDL2" if self.render_enabled else "null")
        press_start(self.pyboy)
        for i in range(600):
            self.pyboy.tick()
            if i == 300:
                press_start(self.pyboy)
            if get_lives(self.pyboy) > 0:
                break
        self.prev_coins = get_coins(self.pyboy)
        self.prev_x = get_mario_position(self.pyboy)[0]
        self.prev_y = get_mario_position(self.pyboy)[1]
        self.prev_lives = get_lives(self.pyboy)
        self.prev_world = get_world_level(self.pyboy)
        self.steps = 0
        return self._get_observation()

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
        done = self._is_done()
        self.prev_coins = get_coins(self.pyboy)
        self.prev_x = get_mario_position(self.pyboy)[0]
        self.prev_y = get_mario_position(self.pyboy)[1]
        self.prev_lives = get_lives(self.pyboy)
        self.prev_world = get_world_level(self.pyboy)
        return observation, reward, done, {"steps": self.steps}

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
        progress_reward = abs(mario_x - self.prev_x) * 1  # Reward for any movement
        coin_reward = (current_coins - self.prev_coins) * 50
        survival_reward = 20
        death_penalty = -100 if current_lives < self.prev_lives else 0
        jump_reward = 20 if mario_y < 80 and self.prev_y >= 100 else 0
        stall_penalty = -1 if mario_x == self.prev_x and mario_y >= 100 else 0
        goomba_proximity = -10 if 110 <= mario_x <= 130 and mario_y >= 100 else 0
        stage_complete = 2000 if current_world > self.prev_world else 0
        return (progress_reward + coin_reward + survival_reward + death_penalty + 
                jump_reward + stall_penalty + goomba_proximity + stage_complete)

    def _is_done(self):
        """Check if the episode is done."""
        return get_lives(self.pyboy) == 0

    def render(self, mode='human'):
        """Render the game environment."""
        if self.render_enabled and mode == 'human':
            pass
        elif mode == 'rgb_array':
            return self._get_observation()

    def close(self):
        """Clean up the environment."""
        self.pyboy.stop()

# Custom Transformer Feature Extractor
class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, feature_dim: int = 128):
        """Initialize Transformer feature extractor."""
        super(TransformerExtractor, self).__init__(observation_space, feature_dim)
        # Adjust for VecTransposeImage [C, H, W] format
        self.image_channels, self.image_height, self.image_width = observation_space.shape  # [C, H, W]
        self.patch_size = 16  # Size of each patch
        self.num_patches = (self.image_height // self.patch_size) * (self.image_width // self.patch_size)
        self.flatten_dim = self.image_channels * self.patch_size * self.patch_size  # 3 * 16 * 16 = 768

        self.embedding = nn.Linear(self.flatten_dim, feature_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True),
            num_layers=1
        )
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches, feature_dim))

    def forward(self, observations):
        """Forward pass of the Transformer extractor."""
        batch_size = observations.shape[0]
        # Handle transposed input from VecTransposeImage [B, C, H, W]
        if observations.dim() == 4 and observations.shape[1] == 3:  # [B, C, H, W]
            patches = observations.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
            patches = patches.contiguous().view(batch_size, -1, self.flatten_dim)  # [B, num_patches, flatten_dim]
        else:
            raise ValueError(f"Unexpected observation shape: {observations.shape}")
        x = self.embedding(patches) + self.positional_encoding[:, :self.num_patches, :]
        x = self.transformer(x)
        return x.mean(dim=1)  # Global average pooling

# Custom Transformer Policy
class TransformerPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        """Initialize Transformer-based policy."""
        super(TransformerPolicy, self).__init__(*args, **kwargs, features_extractor_class=TransformerExtractor,
                                               features_extractor_kwargs={'feature_dim': 128})

    def _build(self, lr_schedule):
        """Build the actor and critic networks with learning rate schedule."""
        if lr_schedule is not None:
            self.lr_schedule = lr_schedule  # Store learning rate schedule if provided

        # Build the actor and critic networks
        self.actor = nn.Sequential(
            nn.Linear(128, 64),  # Reduce dimensions
            nn.ReLU(),
            nn.Linear(64, self.action_space.n)  # Output for 5 actions
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Value output
        )

        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))

    def forward(self, obs, deterministic=False):
        """Forward pass for policy and value."""
        features = self.extract_features(obs)
        logits = self.actor(features)  # Actor output as logits
        value = self.critic(features)
        dist = Categorical(logits=logits)  # Create a categorical distribution
        if deterministic:
            actions = dist.probs.argmax(dim=-1)  # Choose the action with the highest probability
        else:
            actions = dist.sample()  # Sample an action from the distribution
        log_probs = dist.log_prob(actions)  # Log probabilities of the sampled actions
        return actions, value, log_probs  # Return actions, values, and log_probs

    def evaluate_actions(self, obs, actions):
        """Evaluate actions for training."""
        features = self.extract_features(obs)
        values = self.critic(features)
        logits = self.actor(features)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_probs, entropy

    def predict_values(self, obs):
        """Predict the value for a given observation."""
        features = self.extract_features(obs)
        return self.critic(features)

def train_rl_agent(headless=True):
    """Train the RL agent with Transformer policy."""
    env = MarioEnv('SuperMarioLand.gb', render=not headless)
    model = PPO(
        TransformerPolicy,  # Use custom Transformer policy directly
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05
    )
    model.learn(total_timesteps=100000)  # Reduced to 100,000 steps (~20-25 minutes)
    model.save("mario_ppo_model_improved")
    print("=== Training Complete. Saved 'mario_ppo_model_improved.zip' ===")

    env = MarioEnv('SuperMarioLand.gb', render=True)
    obs = env.reset()
    total_reward = 0
    for _ in range(5000):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        if done:
            print(f"Episode ended. Total Reward: {total_reward}, Steps: {info['steps']}")
            obs = env.reset()
            total_reward = 0
    env.close()

'''
def main():
    """Run a simple AI demo before training."""
    pyboy = PyBoy('SuperMarioLand.gb', window="SDL2")
    target_fps = 60
    frame_time = 1 / target_fps
    last_time = time.time()
    for i in range(600):
        pyboy.tick()
        if i == 300:
            press_start(pyboy)
        if get_lives(pyboy) > 0:
            break
    prev_lives = get_lives(pyboy)

    try:
        for _ in range(1000):
            if not pyboy.tick():
                break
            mario_x, mario_y = get_mario_position(pyboy)
            if 120 <= mario_x <= 130:  # Near Goomba-pipe section
                stop_moving(pyboy)
                move_left(pyboy)  # Move left to avoid Goombas
                for _ in range(12):  # Move left for 12 ticks
                    pyboy.tick()
                stop_left(pyboy)
                jump(pyboy)  # Jump over the pipe and Goombas
                for _ in range(25):
                    pyboy.tick()
                stop_jumping(pyboy)
                move_right(pyboy)
            else:
                move_right(pyboy)
                if mario_y >= 100 or (mario_x % 20 == 0):
                    jump(pyboy)
                    for _ in range(10):
                        pyboy.tick()
                    stop_jumping(pyboy)
            current_lives = get_lives(pyboy)
            if current_lives < prev_lives:
                pass
            prev_lives = current_lives
            if current_lives == 0:
                break
            current_time = time.time()
            elapsed = current_time - last_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last_time = current_time
    finally:
        pyboy.stop()
'''

if __name__ == "__main__":
    # Comment out the main function as requested
    # main()
    train_rl_agent(headless=True)