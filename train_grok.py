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
import gc

class TrainingInterrupt(Exception):
    pass

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
        self.action_space = gym.spaces.Discrete(5)  # Reduced from 6 to 5, removing "do nothing"
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        self.frame_skip = 4
        self.closed = False
        
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
        self.action_counts = {i: 0 for i in range(5)}  # Updated to 5 actions

    def reset(self, seed=None, options=None):
        if not self.closed and hasattr(self, 'pyboy'):
            self.pyboy.stop(save=False)
        self.pyboy = PyBoy(self.rom_path, window="SDL2" if self.render_enabled else "null", sound=False)
        self.closed = False
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
        self.action_counts = {i: 0 for i in range(5)}  # Updated to 5 actions
        observation = self._get_observation()
        info = {"steps": self.steps, "total_reward": self.total_reward}
        logger.info(f"Reset: Lives={self.prev_lives}, X={self.prev_x}, Y={self.prev_y}")
        return observation, info

    def step(self, action):
        if self.closed:
            raise RuntimeError("Environment is closed")
        stop_moving(self.pyboy)
        stop_jumping(self.pyboy)
        reward = 0
        action_scalar = action[0] if isinstance(action, np.ndarray) else action  # Extract scalar from array
        for _ in range(self.frame_skip):
            if action_scalar == 0:
                move_right(self.pyboy)
                self.pyboy.tick()
            elif action_scalar == 1:
                jump(self.pyboy)
                self.pyboy.tick()
                stop_jumping(self.pyboy)
            elif action_scalar == 2:
                move_right(self.pyboy)
                jump(self.pyboy)
                self.pyboy.tick()
                stop_jumping(self.pyboy)
            elif action_scalar == 3:
                move_left(self.pyboy)
                self.pyboy.tick()
                stop_left(self.pyboy)
            elif action_scalar == 4:
                move_left(self.pyboy)
                jump(self.pyboy)
                self.pyboy.tick()
                stop_jumping(self.pyboy)
                stop_left(self.pyboy)
            reward += self._get_reward(action_scalar)
            if self._is_done():
                break
        
        self.steps += 1
        self.action_counts[action_scalar] += 1
        observation = self._get_observation()
        self.total_reward += reward
        terminated = self._is_done()
        current_x = get_mario_position(self.pyboy)[0]
        current_y = get_mario_position(self.pyboy)[1]
        
        self.last_action = action_scalar
        truncated = False
        info = {"steps": self.steps, "total_reward": self.total_reward, "action_counts": self.action_counts}
        if terminated:
            logger.info(f"Episode ended: action={action_scalar}, reward={reward}, total_reward={self.total_reward}, lives={get_lives(self.pyboy)}, x={self.prev_x}, action_counts={self.action_counts}")
        if self.render_enabled:
            self.pyboy.tick()
        self.prev_coins = get_coins(self.pyboy)
        self.prev_x = current_x
        self.prev_y = current_y
        self.prev_lives = get_lives(self.pyboy)
        self.prev_world = get_world_level(self.pyboy)
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        if self.closed:
            raise RuntimeError("Cannot get observation from closed environment")
        screen_image = np.array(self.pyboy.screen.ndarray)
        return screen_image[:, :, :3] if screen_image.shape[-1] == 4 else screen_image

    def _get_reward(self, action):
        mario_x, mario_y = get_mario_position(self.pyboy)
        current_coins = get_coins(self.pyboy)
        current_lives = get_lives(self.pyboy)
        current_world = get_world_level(self.pyboy)
        progress_reward = (mario_x - self.prev_x) * 10.0 if mario_x > self.prev_x else 0
        movement_penalty = -0.5 if mario_x <= self.prev_x else 0
        left_penalty = -1.0 if action in [3, 4] else 0  # Adjusted for new action indices
        coin_reward = (current_coins - self.prev_coins) * 5.0
        death_penalty = -50.0 if current_lives < self.prev_lives else 0
        survival_reward = 0.01
        stage_complete = 50.0 if current_world > self.prev_world else 0
        total_reward = progress_reward + movement_penalty + left_penalty + coin_reward + survival_reward + death_penalty + stage_complete
        return np.clip(total_reward, -10.0, 10.0)

    def _is_done(self):
        return get_lives(self.pyboy) == 0

    def render(self, mode='human'):
        if self.closed:
            raise RuntimeError("Cannot render closed environment")
        if self.render_enabled and mode == 'human':
            pass
        elif mode == 'rgb_array':
            return self._get_observation()

    def close(self):
        logger.debug("Closing MarioEnv...")
        if not self.closed and hasattr(self, 'pyboy') and self.pyboy is not None:
            self.pyboy.stop(save=False)
            self.pyboy = None
            self.closed = True
        logger.debug("MarioEnv closed.")

    def save_state(self, path):
        if self.closed:
            logger.warning("Cannot save state of closed environment")
            return
        try:
            with open(path, 'wb') as f:
                self.pyboy.save_state(f)
            logger.info(f"Saved state to {path}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self, path):
        if self.closed:
            logger.warning("Cannot load state into closed environment")
            return
        try:
            with open(path, 'rb') as f:
                self.pyboy.load_state(f)
            logger.info(f"Loaded state from {path}")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

class MambaExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, feature_dim: int = 256):
        super(MambaExtractor, self).__init__(observation_space, feature_dim)
        self.stack_size, self.image_height, self.image_width, self.image_channels = observation_space.shape
        input_dim = self.stack_size * self.image_height * self.image_width * self.image_channels
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, feature_dim)
        logger.debug(f"MambaExtractor initialized: input_dim={input_dim}, feature_dim={feature_dim}")

    def forward(self, observations):
        logger.debug(f"Input observations shape: {observations.shape}, dtype: {observations.dtype}")
        x = observations.float() / 255.0
        logger.debug(f"After conversion shape: {x.shape}, dtype: {x.dtype}")
        x = self.flatten(x)
        logger.debug(f"Flattened shape: {x.shape}")
        features = self.linear(x)
        logger.debug(f"Output features shape: {features.shape}")
        return features

class MambaPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(MambaPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=MambaExtractor,
            features_extractor_kwargs={'feature_dim': 256},
            net_arch=dict(pi=[256], vf=[256])
        )
        self.actor_net = nn.Linear(256, self.action_space.n)
        self.value_net = nn.Linear(256, 1)
        logger.debug(f"Initialized MambaPolicy: actor_net={self.actor_net}, value_net={self.value_net}")

    def _get_features(self, obs):
        features = self.features_extractor(obs)
        logger.debug(f"Extracted features shape: {features.shape}")
        return features

    def forward(self, obs, deterministic=False):
        features = self._get_features(obs)
        logits = self.actor_net(features)
        values = self.value_net(features)
        distribution = Categorical(logits=logits)
        actions = distribution.mode() if deterministic else distribution.sample()
        log_probs = distribution.log_prob(actions)
        return actions, values, log_probs

    def evaluate_actions(self, obs, actions):
        features = self._get_features(obs)
        logits = self.actor_net(features)
        values = self.value_net(features)
        distribution = Categorical(logits=logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

should_exit = False

def signal_handler(sig, frame, env, model):
    global should_exit
    if should_exit:
        logger.debug("Signal handler called again, ignoring...")
        return
    should_exit = True
    logger.info("Ctrl+C detected! Saving state...")
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        if env:
            monitor_env = env.envs[0]
            frame_stack_env = monitor_env.env
            mario_env = frame_stack_env.unwrapped
            mario_env.save_state("mario_state.sav")
            logger.debug("Closing vectorized environment...")
            env.close()
        if model:
            logger.debug("Saving model...")
            model.save("grok_mamba")
        logger.info("State and model saved. Exiting gracefully...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.2)
        raise TrainingInterrupt("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error in signal handler: {e}")
        raise

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
        if should_exit:
            logger.debug("Stopping training loop due to should_exit")
            return False
        return True

    def _on_rollout_end(self):
        if should_exit:
            return
        current_session_timesteps = self.num_timesteps
        total_timesteps_so_far = current_session_timesteps + self.initial_timesteps
        percentage_complete = (total_timesteps_so_far / self.total_timesteps) * 100
        logger.info(f"Callback: Total timesteps = {total_timesteps_so_far}/{self.total_timesteps}, Progress = {percentage_complete:.2f}%")
        if total_timesteps_so_far - self.last_save >= self.interval:
            logger.info(f"Autosaving at timestep {total_timesteps_so_far} ({percentage_complete:.2f}%)...")
            monitor_env = self.env.envs[0]
            frame_stack_env = monitor_env.env
            mario_env = frame_stack_env.unwrapped
            mario_env.save_state("mario_state_autosave.sav")
            self.model.save("grok_mamba_autosave")
            self.last_save = total_timesteps_so_far

def train_rl_agent(render=False, resume=False, use_cuda=False, model_path="grok_mamba.zip"):
    global should_exit
    should_exit = False

    base_env = MarioEnv('SuperMarioLand.gb', render=render)
    base_env = FrameStack(base_env, num_stack=4)
    logger.info(f"Base observation space: {base_env.observation_space}")
    env = Monitor(base_env)
    env = DummyVecEnv([lambda: env])
    logger.info(f"Wrapped observation space: {env.observation_space}")
    total_training_timesteps = 500_000
    
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    if use_cuda and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")

    if resume and os.path.exists(model_path):
        model = PPO.load(model_path, env=env, device=device, custom_objects={"policy_class": MambaPolicy})
        logger.debug(f"Loaded model policy: actor_net={model.policy.actor_net}, value_net={model.policy.value_net}")
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
            ent_coef=10.0,
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
            callback=AutosaveCallback(env, model, total_training_timesteps, initial_timesteps),
            progress_bar=True,
            reset_num_timesteps=False
        )
        total_timesteps_so_far = model.num_timesteps
        percentage_complete = (total_timesteps_so_far / total_training_timesteps) * 100
        model.save("grok_mamba")
        logger.info(f"Training complete. Saved 'grok_mamba.zip'. Total timesteps: {total_timesteps_so_far}, Progress: {percentage_complete:.2f}%")
    except TrainingInterrupt:
        logger.info("Training interrupted by signal handler")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in training, cleanup handled by signal handler")
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        raise
    finally:
        logger.debug("Ensuring environment is closed in finally block...")
        if 'env' in locals():
            try:
                env.close()
            except Exception as e:
                logger.error(f"Error closing environment: {e}")
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error clearing CUDA cache: {e}")
        gc.collect()
        time.sleep(0.2)

    if not should_exit and render:
        play_env = MarioEnv('SuperMarioLand.gb', render=True)
        play_env = FrameStack(play_env, num_stack=4)
        play_env = Monitor(play_env)
        play_vec_env = DummyVecEnv([lambda: play_env])
        obs = play_vec_env.reset()
        total_reward = 0
        for step in range(5000):
            if should_exit:
                logger.info("Exiting play loop due to Ctrl+C...")
                break
            action, _ = model.predict(obs)
            logger.debug(f"Post-training action: {action}, type: {type(action)}")
            obs, rewards, dones, infos = play_vec_env.step(action)
            total_reward += rewards[0]
            play_vec_env.render()
            if dones[0]:
                logger.info(f"Episode ended. Reward: {total_reward}, Steps: {infos[0]['steps']}, Action Counts: {infos[0]['action_counts']}")
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
        env = DummyVecEnv([lambda: env])
        
        device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        if use_cuda and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available")
        
        logger.info(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=env, device=device, custom_objects={"policy_class": MambaPolicy})
        logger.debug(f"Loaded model policy: actor_net={model.policy.actor_net}, value_net={model.policy.value_net}")
        logger.debug(f"Post-load self.forward type: {type(model.policy.forward)}")
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
            logger.debug(f"Play action: {action}, type: {type(action)}")
            obs, rewards, dones, infos = env.step(action)
            total_reward += rewards[0]
            env.render()
            if dones[0]:
                logger.info(f"Episode ended. Reward: {total_reward}, Steps: {infos[0]['steps']}, Action Counts: {infos[0]['action_counts']}")
                obs = env.reset()
                total_reward = 0
        env.close()
    except TrainingInterrupt:
        logger.info("Play interrupted by signal handler")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in play, cleanup handled by signal handler")
    except Exception as e:
        logger.error(f"Error in play: {e}")
        raise
    finally:
        if 'env' in locals():
            try:
                env.close()
            except Exception as e:
                logger.error(f"Error closing environment: {e}")
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error clearing CUDA cache: {e}")
        gc.collect()
        time.sleep(0.2)

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
    except TrainingInterrupt:
        logger.info("Main interrupted by signal handler")
    except KeyboardInterrupt:
        logger.info("Interrupted in main! Cleanup handled by signal handler")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)