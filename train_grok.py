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
        """Calculate the reward based on current state with refined incentives."""
        mario_x, mario_y = get_mario_position(self.pyboy)
        current_coins = get_coins(self.pyboy)
        current_lives = get_lives(self.pyboy)
        current_world = get_world_level(self.pyboy)

        # Reward for progressing right (stronger incentive)
        progress_reward = (mario_x - self.prev_x) * 3 if mario_x > self.prev_x else 0

        # Penalty for moving left or stalling
        movement_penalty = -0.5 if mario_x <= self.prev_x else 0

        # Reward for collecting coins (moderate incentive)
        coin_reward = (current_coins - self.prev_coins) * 5

        # Penalty for losing a life (significant but not overwhelming)
        death_penalty = -50 if current_lives < self.prev_lives else 0

        # Reward for surviving
        survival_reward = 0.1

        # Reward for completing the stage
        stage_complete = 100 if current_world > self.prev_world else 0

        # Reward for jumping (even if no enemy is present)
        jump_reward = 5 if mario_y < self.prev_y else 0

        # Total reward (normalized)
        total_reward = (
            progress_reward +
            movement_penalty +
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
        if self.render_enabled and mode == 'human':
            pass
        elif mode == 'rgb_array':
            return self._get_observation()

    def close(self):
        """Clean up the environment."""
        if self.render_enabled:
            disable_turbo_mode(self.pyboy)
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
        """Initialize Transformer-based policy with actor and critic networks."""
        super(TransformerPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=TransformerExtractor,
            features_extractor_kwargs={'feature_dim': 128}
        )
        # Define actor and critic networks
        self.actor = nn.Linear(128, self.action_space.n)  # Action logits for 5 actions
        self.critic = nn.Linear(128, 1)  # Value estimate

    def forward(self, obs, deterministic=False):
        """Forward pass for policy and value, returning actions, values, and log_probs."""
        features = self.extract_features(obs)
        logits = self.actor(features)
        values = self.critic(features)
        dist = Categorical(logits=logits)
        actions = dist.mode() if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, values, log_probs  # Return three values as required by PPO

    def predict(self, obs, deterministic=False):
        """Predict an action given an observation."""
        with torch.no_grad():
            # Ensure obs is a tensor and add batch dimension if needed
            obs_tensor = torch.as_tensor(obs, device=self.device)
            if obs_tensor.dim() == 3:  # Add batch dimension for single observation
                obs_tensor = obs_tensor.unsqueeze(0)
            actions, values, _ = self.forward(obs_tensor, deterministic)
            # Squeeze to remove batch dimension and convert to NumPy
            actions = actions.squeeze().cpu().numpy()
            values = values.squeeze().cpu().numpy()
            return actions, values

def train_rl_agent(headless=True):
    """Train the RL agent with Transformer policy."""
    env = MarioEnv('SuperMarioLand.gb', render=not headless)
    model = PPO(
        TransformerPolicy,  # Use custom Transformer policy
        env,
        verbose=1,
        learning_rate=0.0001,  # Reduced learning rate
        n_steps=2048,
        batch_size=128,  # Increased batch size
        n_epochs=10,  # Increased epochs
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.1,  # Increased entropy coefficient
    )
    model.learn(total_timesteps=1_000_000)  # Train for 1 million steps
    model.save("grok")
    print("=== Training Complete. Saved 'grok.zip' ===")

    env = MarioEnv('SuperMarioLand.gb', render=True)
    obs, _ = env.reset()  # Gymnasium requires unpacking observation and info
    total_reward = 0
    for _ in range(5000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if terminated or truncated:
            print(f"Episode ended. Total Reward: {total_reward}, Steps: {info['steps']}")
            obs, _ = env.reset()
            total_reward = 0
    env.close()

def play(model_path="grok.zip"):
    """Load the trained model and play the game with debugging to address movement issues."""
    try:
        env = MarioEnv('SuperMarioLand.gb', render=True)
        model = PPO.load(model_path)
        print("Model loaded successfully!")
        print(f"Loaded model policy class: {model.policy.__class__.__name__}")  # Debug policy class
        obs, _ = env.reset()
        total_reward = 0
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Track action distribution
        prev_x = get_mario_position(env.pyboy)[0]  # Track previous x-position
        for step in range(5000):
            # Debug the raw prediction output
            raw_action, _ = model.predict(obs, deterministic=True)
            print(f"Step {step}, Raw prediction: {raw_action}, Type: {type(raw_action)}")
            # Extract action based on type
            if isinstance(raw_action, Categorical):
                action = raw_action.mode().item()
            elif isinstance(raw_action, np.ndarray):
                action = raw_action.item() if raw_action.size == 1 else raw_action[0]
            elif isinstance(raw_action, torch.Tensor):
                action = raw_action.item() if raw_action.dim() == 0 else raw_action[0].item()
            else:
                raise ValueError(f"Unexpected action type: {type(raw_action)}")
            action_counts[action] += 1
            print(f"Step {step}, Action predicted: {action}")
            # Log Mario's position
            current_x, current_y = get_mario_position(env.pyboy)
            delta_x = current_x - prev_x
            print(f"Step {step}, Mario position (x, y): ({current_x}, {current_y}), Delta x: {delta_x}")
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            prev_x = current_x  # Update previous x-position
            time.sleep(0.01)  # Prevent quick shutdown
            if terminated or truncated:
                print(f"Episode ended at step {step}. Total Reward: {total_reward}, Steps: {info['steps']}")
                print(f"Action distribution: {action_counts}")
                obs, _ = env.reset()
                total_reward = 0
                action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                prev_x = get_mario_position(env.pyboy)[0]  # Reset prev_x
        env.close()
    except Exception as e:
        print(f"Error during play: {e}")

if __name__ == "__main__":
    # Train the RL agent or play
    train_rl_agent(headless=True)  # Uncomment to train
    #play(model_path="grok.zip")  # Play with the new trained model