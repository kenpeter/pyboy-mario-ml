from pyboy import PyBoy
from pyboy.utils import WindowEvent
import time
import numpy as np
import gymnasium as gym
try:
    from gymnasium.wrappers.frame_stack import FrameStack
except ImportError:
    import logging
    logging.error("FrameStack not found in gymnasium.wrappers.frame_stack. Please check gymnasium version.")
    raise

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Categorical
import signal
import sys
import os
import logging
import argparse

def setup_logging(debug=False):
    logging_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Logging level set to: {logging.getLevelName(logging_level)}")
    return logger

MARIO_X_POS = 0xC202
MARIO_Y_POS = 0xC201
LIVES = 0xDA15
COINS = 0xFFFA
WORLD_LEVEL = 0xFFB4

def get_mario_position(pyboy): return pyboy.memory[MARIO_X_POS], pyboy.memory[MARIO_Y_POS]
def get_lives(pyboy): return pyboy.memory[LIVES]
def get_coins(pyboy): return pyboy.memory[COINS]
def get_world_level(pyboy): return pyboy.memory[WORLD_LEVEL]
def move_right(pyboy): pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
def stop_moving(pyboy):
    pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
    pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
def move_left(pyboy): pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
def stop_left(pyboy): pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
def jump(pyboy): pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
def stop_jumping(pyboy): pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
def press_start(pyboy):
    pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
    for _ in range(30): pyboy.tick()
    pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)

class MarioEnv(gym.Env):
    def __init__(self, rom_path, render=False):
        super(MarioEnv, self).__init__()
        self.rom_path = rom_path
        self.render_enabled = render
        self.pyboy = PyBoy(rom_path, window="SDL2" if render else "null", sound=False)
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        self.frame_skip = 4
        
        press_start(self.pyboy)
        for i in range(300):
            self.pyboy.tick()
            if i == 150: press_start(self.pyboy)
            if get_lives(self.pyboy) > 0: break
        self.initial_lives = get_lives(self.pyboy)

        self.prev_coins = get_coins(self.pyboy)
        self.prev_x = get_mario_position(self.pyboy)[0]
        self.prev_y = get_mario_position(self.pyboy)[1]
        self.prev_lives = get_lives(self.pyboy)
        self.prev_world = get_world_level(self.pyboy)
        self.steps = 0
        self.visited_x = set()
        self.total_reward = 0.0
        self.last_action = None

    def reset(self, seed=None, options=None):
        self.pyboy.stop(save=False)
        self.pyboy = PyBoy(self.rom_path, window="SDL2" if self.render_enabled else "null", sound=False)
        press_start(self.pyboy)
        for i in range(300):
            self.pyboy.tick()
            if i == 150: press_start(self.pyboy)
            lives = get_lives(self.pyboy)
            if lives > 0:
                logger.info(f"Game started with {lives} lives at tick {i}")
                self.initial_lives = lives
                break
        else:
            logger.error("Failed to start game with lives > 0 after 300 ticks")
            raise RuntimeError("Could not initialize game with lives")
        
        for _ in range(15): self.pyboy.tick()
        
        self.prev_coins = get_coins(self.pyboy)
        self.prev_x = get_mario_position(self.pyboy)[0]
        self.prev_y = get_mario_position(self.pyboy)[1]
        self.prev_lives = get_lives(self.pyboy)
        self.prev_world = get_world_level(self.pyboy)
        self.steps = 0
        self.visited_x = {self.prev_x}
        self.total_reward = 0.0
        self.last_action = None
        observation = self._get_observation()
        info = {"steps": self.steps, "total_reward": self.total_reward}
        logger.info(f"Reset: Lives={self.prev_lives}, X={self.prev_x}, Y={self.prev_y}")
        return observation, info

    def step(self, action):
        stop_moving(self.pyboy)
        stop_jumping(self.pyboy)
        reward = 0
        for _ in range(self.frame_skip):
            if action == 0:
                self.pyboy.tick()
            elif action == 1:
                move_right(self.pyboy)
                self.pyboy.tick()
            elif action == 2:
                jump(self.pyboy)
                self.pyboy.tick()
                stop_jumping(self.pyboy)
            elif action == 3:
                move_right(self.pyboy)
                jump(self.pyboy)
                self.pyboy.tick()
                stop_jumping(self.pyboy)
            elif action == 4:
                move_left(self.pyboy)
                self.pyboy.tick()
                stop_left(self.pyboy)
            elif action == 5:
                move_left(self.pyboy)
                jump(self.pyboy)
                self.pyboy.tick()
                stop_jumping(self.pyboy)
                stop_left(self.pyboy)
            reward += self._get_reward()
            if self._is_done():
                break
        
        self.steps += 1
        observation = self._get_observation()
        self.total_reward += reward
        terminated = self._is_done()
        current_x = get_mario_position(self.pyboy)[0]
        current_y = get_mario_position(self.pyboy)[1]
        
        self.last_action = action
        truncated = False
        info = {"steps": self.steps, "total_reward": self.total_reward}
        if terminated:
            logger.info(f"Episode ended: action={action}, reward={reward}, total_reward={self.total_reward}, lives={get_lives(self.pyboy)}, x={self.prev_x}")
        if self.render_enabled:
            self.pyboy.tick()
        self.prev_coins = get_coins(self.pyboy)
        self.prev_x = current_x
        self.prev_y = current_y
        self.prev_lives = get_lives(self.pyboy)
        self.prev_world = get_world_level(self.pyboy)
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        screen_image = np.array(self.pyboy.screen.ndarray)
        return screen_image[:, :, :3] if screen_image.shape[-1] == 4 else screen_image

    def _get_reward(self):
        mario_x, mario_y = get_mario_position(self.pyboy)
        current_coins = get_coins(self.pyboy)
        current_lives = get_lives(self.pyboy)
        current_world = get_world_level(self.pyboy)
        progress_reward = (mario_x - self.prev_x) * 2.0 if mario_x > self.prev_x else 0
        movement_penalty = -0.01 if mario_x <= self.prev_x else 0
        coin_reward = (current_coins - self.prev_coins) * 5.0
        death_penalty = -50.0 if current_lives < self.prev_lives else 0
        survival_reward = 0.05
        stage_complete = 50.0 if current_world > self.prev_world else 0
        total_reward = progress_reward + movement_penalty + coin_reward + survival_reward + death_penalty + stage_complete
        return np.clip(total_reward, -10.0, 10.0)

    def _is_done(self):
        return get_lives(self.pyboy) == 0

    def render(self, mode='human'):
        if self.render_enabled and mode == 'human':
            pass
        elif mode == 'rgb_array':
            return self._get_observation()

    def close(self):
        logger.debug("Closing MarioEnv...")
        if hasattr(self, 'pyboy') and self.pyboy is not None:
            self.pyboy.stop(save=False)  # Ensure no save on close to avoid conflicts
            self.pyboy = None  # Clear reference
        logger.debug("MarioEnv closed.")

    def save_state(self, path):
        try:
            with open(path, 'wb') as f:
                self.pyboy.save_state(f)
            logger.info(f"Saved state to {path}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self, path):
        try:
            with open(path, 'rb') as f:
                self.pyboy.load_state(f)
            logger.info(f"Loaded state from {path}")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

class MambaExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, feature_dim: int = 128):
        super(MambaExtractor, self).__init__(observation_space, feature_dim)
        self.stack_size, self.image_height, self.image_width, self.image_channels = observation_space.shape
        self.patch_size = 32
        self.num_patches_per_frame = (self.image_height // self.patch_size) * (self.image_width // self.patch_size)
        self.flatten_dim = self.image_channels * self.patch_size * self.patch_size
        self.embedding = nn.Linear(self.flatten_dim, feature_dim)
        self.ssm_dim = feature_dim
        self.A = nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.01)
        self.C = nn.Linear(feature_dim, feature_dim)

    def forward(self, observations):
        batch_size = observations.shape[0]
        if observations.dim() != 5:
            raise ValueError(f"Unexpected shape: {observations.shape}, expected [B, {self.stack_size}, {self.image_height}, {self.image_width}, {self.image_channels}]")
        
        stack_size = observations.shape[1]
        patches = observations.view(batch_size, stack_size, self.image_height, self.image_width, self.image_channels)
        patches = patches.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, stack_size * self.num_patches_per_frame, self.flatten_dim)
        
        x = self.embedding(patches)
        seq_len = x.shape[1]
        state = torch.zeros(batch_size, self.ssm_dim, device=x.device)
        outputs = []
        for t in range(min(seq_len, 20)):
            state = torch.tanh(torch.matmul(state, self.A) + torch.matmul(x[:, t, :], self.B))
            outputs.append(self.C(state))
        x = torch.stack(outputs, dim=1).mean(dim=1)
        return x

class MambaPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(MambaPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=MambaExtractor,
            features_extractor_kwargs={'feature_dim': 128}
        )
        self.actor = nn.Linear(128, self.action_space.n)
        self.critic = nn.Linear(128, 1)
        logger.debug(f"Initialized MambaPolicy: actor={self.actor}, critic={self.critic}")

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        logger.debug(f"Features shape: {features.shape}")
        logits = self.actor(features)
        logger.debug(f"Logits shape: {logits.shape}")
        values = self.critic(features)
        logger.debug(f"Values shape: {values.shape}")
        dist = Categorical(logits=logits)
        logger.debug(f"Distribution created: {dist}")
        if not isinstance(dist, Categorical):
            logger.error(f"dist is not a Categorical object: {type(dist)}")
            raise TypeError("Distribution is not a Categorical object")
        actions = dist.mode() if deterministic else dist.sample()
        logger.debug(f"Actions: {actions}")
        log_probs = dist.log_prob(actions)
        return actions, values, log_probs

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observation, device=self.device)
            if obs_tensor.dim() == 4:  # Add batch dimension if missing
                obs_tensor = obs_tensor.unsqueeze(0)
            logger.debug(f"Observation tensor shape: {obs_tensor.shape}, device: {obs_tensor.device}")
            logger.debug(f"Critic before predict: {self.critic}")
            actions, values, _ = self.forward(obs_tensor, deterministic)
            actions = actions.squeeze().cpu().numpy()  # Convert to numpy for Gym
            return actions, None

should_exit = False

def signal_handler(sig, frame, env, model):
    global should_exit
    if should_exit:  # Prevent multiple calls
        return
    logger.info("Ctrl+C detected! Saving state...")
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore further SIGINT signals
    if env:
        base_env = env.envs[0].env if isinstance(env, DummyVecEnv) else env
        base_env.save_state("mario_state.sav")
        logger.debug("Closing vectorized environment...")
        env.close()
    if model:
        logger.debug("Saving model...")
        model.save("grok_mamba")
    logger.info("State and model saved. Preparing to exit...")
    should_exit = True
    time.sleep(0.5)  # Brief delay to allow cleanup
    sys.exit(0)  # Force exit after saving

class AutosaveCallback(BaseCallback):
    def __init__(self, env, model, total_timesteps, initial_timesteps=0, interval=4096, verbose=0):
        super(AutosaveCallback, self).__init__(verbose)
        self.env = env
        self.model = model
        self.total_timesteps = total_timesteps
        self.initial_timesteps = initial_timesteps
        self.interval = interval
        self.last_save = initial_timesteps

    def _on_step(self):
        return not should_exit

    def _on_rollout_end(self):
        current_session_timesteps = self.num_timesteps
        total_timesteps_so_far = current_session_timesteps + self.initial_timesteps
        percentage_complete = (total_timesteps_so_far / self.total_timesteps) * 100
        logger.info(f"Callback: Total timesteps = {total_timesteps_so_far}/{self.total_timesteps}, Progress = {percentage_complete:.2f}%")
        if total_timesteps_so_far - self.last_save >= self.interval:
            logger.info(f"Autosaving at timestep {total_timesteps_so_far} ({percentage_complete:.2f}%)...")
            self.env.save_state("mario_state_autosave.sav")
            self.model.save("grok_mamba_autosave")
            self.last_save = total_timesteps_so_far

def train_rl_agent(render=False, resume=False, use_cuda=False, model_path="grok_mamba.zip"):
    global should_exit
    should_exit = False

    base_env = MarioEnv('SuperMarioLand.gb', render=render)
    base_env = FrameStack(base_env, num_stack=4)
    logger.info(f"Base observation space: {base_env.observation_space}")
    env = Monitor(base_env)
    env = DummyVecEnv([lambda: base_env])
    logger.info(f"Wrapped observation space: {env.observation_space}")
    total_training_timesteps = 1_000_000  # Adjusted to 1 million as recommended
    
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    if use_cuda and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")

    if resume and os.path.exists(model_path):
        model = PPO.load(model_path, env=env, device=device, custom_objects={"policy_class": MambaPolicy})
        logger.debug(f"Loaded model policy: actor={model.policy.actor}, critic={model.policy.critic}")
        initial_timesteps = model.num_timesteps
        remaining_timesteps = total_training_timesteps - initial_timesteps
        logger.info(f"Resuming from {initial_timesteps} timesteps, {remaining_timesteps} remaining")
    else:
        model = PPO(
            MambaPolicy,
            env,
            verbose=2,
            learning_rate=0.0001,
            n_steps=2048,
            batch_size=256,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=2.0,
            vf_coef=0.5,
            device=device
        )
        initial_timesteps = 0
        remaining_timesteps = total_training_timesteps
        logger.info("Starting fresh training")

    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, env, model))
    logger.info("Signal handler registered")

    logger.info(f"Starting training for {remaining_timesteps} timesteps (Total: {total_training_timesteps})...")
    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=AutosaveCallback(base_env, model, total_training_timesteps, initial_timesteps),
            progress_bar=True,
            reset_num_timesteps=False
        )
        total_timesteps_so_far = model.num_timesteps
        percentage_complete = (total_timesteps_so_far / total_training_timesteps) * 100
        model.save("grok_mamba")
        logger.info(f"Training complete. Saved 'grok_mamba.zip'. Total timesteps: {total_timesteps_so_far}, Progress: {percentage_complete:.2f}%")
    except KeyboardInterrupt:
        total_timesteps_so_far = model.num_timesteps
        percentage_complete = (total_timesteps_so_far / total_training_timesteps) * 100
        logger.info(f"Interrupted! Saving at {percentage_complete:.2f}% (Timesteps: {total_timesteps_so_far}/{total_training_timesteps})...")
        base_env.save_state("mario_state.sav")
        model.save("grok_mamba")
        env.close()
        logger.info("State and model saved. Exiting cleanly...")
        sys.exit(0)
    finally:
        logger.debug("Ensuring environment is closed in finally block...")
        if 'env' in locals():
            env.close()

    if not should_exit:
        play_env = MarioEnv('SuperMarioLand.gb', render=True)
        play_env = FrameStack(play_env, num_stack=4)
        play_env = Monitor(play_env)
        play_vec_env = DummyVecEnv([lambda: play_env])
        obs = play_vec_env.reset()
        total_reward = 0
        for step in range(5000):
            if should_exit:  # Check for exit condition
                logger.info("Exiting play loop due to Ctrl+C...")
                break
            action, _ = model.predict(obs)
            obs, rewards, dones, infos = play_vec_env.step([action])
            total_reward += rewards[0]
            play_vec_env.render()
            if dones[0]:
                logger.info(f"Episode ended. Reward: {total_reward}, Steps: {infos[0]['steps']}")
                obs = play_vec_env.reset()
                total_reward = 0
        play_vec_env.close()

def play(model_path="grok_mamba.zip", state_path="mario_state.sav", use_cuda=False):
    global should_exit
    should_exit = False

    try:
        base_env = MarioEnv('SuperMarioLand.gb', render=True)
        base_env = FrameStack(base_env, num_stack=4)
        env = Monitor(base_env)
        env = DummyVecEnv([lambda: base_env])
        
        device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        if use_cuda and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available")
        
        logger.info(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=env, device=device, custom_objects={"policy_class": MambaPolicy})
        logger.debug(f"Loaded model policy: actor={model.policy.actor}, critic={model.policy.critic}")
        logger.info("Model loaded!")
        
        signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, env, model))
        logger.info("Signal handler registered")

        obs = env.reset()
        total_reward = 0
        for step in range(5000):
            if should_exit:
                logger.info("Exiting play loop due to Ctrl+C...")
                break
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step([action])
            total_reward += rewards[0]
            env.render()
            if dones[0]:
                logger.info(f"Episode ended. Reward: {total_reward}, Steps: {infos[0]['steps']}")
                obs = env.reset()
                total_reward = 0
        env.close()
    except KeyboardInterrupt:
        logger.info("Interrupted in play! Saving state...")
        base_env.save_state("mario_state.sav")
        model.save("grok_mamba")
        logger.info("State and model saved. Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in play: {e}")
        base_env.close()
        raise
    finally:
        base_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play Super Mario Land with RL")
    parser.add_argument('--render', action='store_true', help="Enable game UI")
    parser.add_argument('--play', action='store_true', help="Play trained model")
    parser.add_argument('--resume', action='store_true', help="Resume training")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    parser.add_argument('--cuda', action='store_true', help="Use CUDA")
    parser.add_argument('--model_path', type=str, default="grok_mamba.zip", help="Path to the model file to load")
    args = parser.parse_args()

    logger = setup_logging(debug=args.debug)

    try:
        if args.play:
            play(model_path=args.model_path, use_cuda=args.cuda)
        else:
            train_rl_agent(render=args.render, resume=args.resume, use_cuda=args.cuda, model_path=args.model_path)
    except KeyboardInterrupt:
        logger.info("Interrupted in main! State should be saved.")
        sys.exit(0)