from pyboy import PyBoy
from pyboy.utils import WindowEvent
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    for _ in range(60): pyboy.tick()
    pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
def enable_turbo_mode(pyboy):
    pyboy.send_input(WindowEvent.PRESS_SPEED_UP)
    pyboy.tick()
def disable_turbo_mode(pyboy):
    pyboy.send_input(WindowEvent.RELEASE_SPEED_UP)
    pyboy.tick()

class MarioEnv(gym.Env):
    def __init__(self, rom_path, render=False, state_path=None):
        super(MarioEnv, self).__init__()
        self.rom_path = rom_path
        self.render_enabled = render
        self.state_path = state_path
        self.pyboy = PyBoy(rom_path, window="SDL2" if render else "null")
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        
        if state_path and os.path.exists(state_path):
            self.load_state(state_path)
        else:
            press_start(self.pyboy)
            for i in range(600):
                self.pyboy.tick()
                if i == 300: press_start(self.pyboy)
                if get_lives(self.pyboy) > 0: break

        if self.render_enabled:
            time.sleep(5)
            enable_turbo_mode(self.pyboy)
        
        self.prev_coins = get_coins(self.pyboy)
        self.prev_x = get_mario_position(self.pyboy)[0]
        self.prev_y = get_mario_position(self.pyboy)[1]
        self.prev_lives = get_lives(self.pyboy)
        self.prev_world = get_world_level(self.pyboy)
        self.steps = 0

    def reset(self, seed=None, options=None):
        if self.state_path and os.path.exists(self.state_path):
            self.load_state(self.state_path)
        else:
            self.pyboy.stop(save=False)
            self.pyboy = PyBoy(self.rom_path, window="SDL2" if self.render_enabled else "null")
            press_start(self.pyboy)
            for i in range(600):
                self.pyboy.tick()
                if i == 300: press_start(self.pyboy)
                lives = get_lives(self.pyboy)
                if lives > 0:
                    logger.info(f"Game started with {lives} lives at tick {i}")
                    break
            else:
                logger.error("Failed to start game with lives > 0 after 600 ticks")
                raise RuntimeError("Could not initialize game with lives")
            if self.render_enabled:
                time.sleep(5)
                enable_turbo_mode(self.pyboy)
        
        self.prev_coins = get_coins(self.pyboy)
        self.prev_x = get_mario_position(self.pyboy)[0]
        self.prev_y = get_mario_position(self.pyboy)[1]
        self.prev_lives = get_lives(self.pyboy)
        self.prev_world = get_world_level(self.pyboy)
        self.steps = 0
        observation = self._get_observation()
        info = {"steps": self.steps}
        logger.info(f"Reset: Lives={self.prev_lives}, X={self.prev_x}, Y={self.prev_y}")
        return observation, info

    def step(self, action):
        stop_moving(self.pyboy)
        stop_jumping(self.pyboy)
        if action == 0:
            for _ in range(4): self.pyboy.tick()
        elif action == 1:
            move_right(self.pyboy)
            for _ in range(4): self.pyboy.tick()
        elif action == 2:
            jump(self.pyboy)
            for _ in range(20): self.pyboy.tick()
            stop_jumping(self.pyboy)
        elif action == 3:
            move_right(self.pyboy)
            jump(self.pyboy)
            for _ in range(20): self.pyboy.tick()
            stop_jumping(self.pyboy)
        elif action == 4:
            move_left(self.pyboy)
            for _ in range(4): self.pyboy.tick()
            stop_left(self.pyboy)
        self.steps += 1
        observation = self._get_observation()
        reward = self._get_reward()
        terminated = self._is_done()
        truncated = False
        info = {"steps": self.steps}
        logger.debug(f"Step {self.steps}: action={action}, reward={reward}, terminated={terminated}, truncated={truncated}, lives={get_lives(self.pyboy)}")
        self.prev_coins = get_coins(self.pyboy)
        self.prev_x = get_mario_position(self.pyboy)[0]
        self.prev_y = get_mario_position(self.pyboy)[1]
        self.prev_lives = get_lives(self.pyboy)
        self.prev_world = get_world_level(self.pyboy)
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        screen_image = np.array(self.pyboy.screen.image)
        logger.debug(f"Screen image shape: {screen_image.shape}")
        return screen_image[:, :, :3] if screen_image.shape[-1] == 4 else screen_image

    def _get_reward(self):
        mario_x, mario_y = get_mario_position(self.pyboy)
        current_coins = get_coins(self.pyboy)
        current_lives = get_lives(self.pyboy)
        current_world = get_world_level(self.pyboy)
        progress_reward = (mario_x - self.prev_x) * 3 if mario_x > self.prev_x else 0
        movement_penalty = -0.5 if mario_x <= self.prev_x else 0
        coin_reward = (current_coins - self.prev_coins) * 5
        death_penalty = -50 if current_lives < self.prev_lives else 0
        survival_reward = 0.1
        stage_complete = 100 if current_world > self.prev_world else 0
        jump_reward = 5 if mario_y < self.prev_y else 0
        total_reward = (progress_reward + movement_penalty + coin_reward + survival_reward +
                        death_penalty + stage_complete + jump_reward) / 100
        return total_reward

    def _is_done(self): return get_lives(self.pyboy) == 0

    def render(self, mode='human'):
        if self.render_enabled and mode == 'human': pass
        elif mode == 'rgb_array': return self._get_observation()

    def close(self):
        if self.render_enabled: disable_turbo_mode(self.pyboy)
        self.pyboy.stop()

    def save_state(self, path):
        try:
            with open(path, 'wb') as f:
                self.pyboy.save_state(f)
            logger.info(f"Saved emulator state to {path}")
        except Exception as e:
            logger.error(f"Failed to save state to {path}: {e}")

    def load_state(self, path):
        try:
            with open(path, 'rb') as f:
                self.pyboy.load_state(f)
            logger.info(f"Loaded emulator state from {path}")
        except Exception as e:
            logger.error(f"Failed to load state from {path}: {e}")

class MambaExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, feature_dim: int = 128):
        super(MambaExtractor, self).__init__(observation_space, feature_dim)
        self.image_channels, self.image_height, self.image_width = observation_space.shape  # [C, H, W]
        self.patch_size = 16
        if self.image_height < self.patch_size or self.image_width < self.patch_size:
            raise ValueError(f"Observation height ({self.image_height}) or width ({self.image_width}) smaller than patch_size ({self.patch_size})")
        self.num_patches = (self.image_height // self.patch_size) * (self.image_width // self.patch_size)
        self.flatten_dim = self.image_channels * self.patch_size * self.patch_size
        self.embedding = nn.Linear(self.flatten_dim, feature_dim)
        self.ssm_dim = feature_dim
        self.A = nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.01)
        self.C = nn.Linear(feature_dim, feature_dim)

    def forward(self, observations):
        batch_size = observations.shape[0]
        logger.debug(f"Observations shape: {observations.shape}")
        if observations.dim() != 4 or observations.shape[1] != self.image_channels:
            raise ValueError(f"Unexpected observation shape: {observations.shape}, expected [B, {self.image_channels}, H, W]")
        
        if observations.shape[2] < self.patch_size or observations.shape[3] < self.patch_size:
            raise ValueError(f"Observation height ({observations.shape[2]}) or width ({observations.shape[3]}) smaller than patch_size ({self.patch_size})")
        
        patches = observations.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.flatten_dim)
        
        x = self.embedding(patches)
        seq_len = x.shape[1]
        state = torch.zeros(batch_size, self.ssm_dim, device=x.device)  # Reset state per forward pass
        outputs = []
        for t in range(seq_len):
            state = torch.tanh(torch.matmul(state, self.A) + torch.matmul(x[:, t, :], self.B))
            output = self.C(state)
            outputs.append(output)
        x = torch.stack(outputs, dim=1)
        return x.mean(dim=1)

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

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        logits = self.actor(features)
        values = self.critic(features)
        dist = Categorical(logits=logits)
        actions = dist.mode() if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, values, log_probs

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observation, device=self.device)
            if obs_tensor.dim() == 3: obs_tensor = obs_tensor.unsqueeze(0)
            actions, values, _ = self.forward(obs_tensor, deterministic)
            actions = actions.squeeze().cpu().numpy()
            return actions, None

def signal_handler(sig, frame, env, model):
    logger.info("Ctrl+C detected via signal handler! Saving state...")
    if env:
        env.save_state("mario_state.sav")
    if model:
        model.save("grok_mamba")
    logger.info("State and model saved via signal handler. Exiting...")
    sys.exit(0)

class AutosaveCallback(BaseCallback):
    def __init__(self, env, model, interval=8192, verbose=0):
        super(AutosaveCallback, self).__init__(verbose)
        self.env = env
        self.model = model
        self.interval = interval
        self.last_save = 0

    def _on_step(self):
        return True  # Don’t interfere with step-by-step execution

    def _on_rollout_end(self):
        current_timesteps = self.num_timesteps
        logger.info(f"Callback: Current timesteps = {current_timesteps}")
        if current_timesteps - self.last_save >= self.interval:
            logger.info(f"Autosaving at timestep {current_timesteps}...")
            self.env.save_state("mario_state_autosave.sav")
            self.model.save("grok_mamba_autosave")
            self.last_save = current_timesteps

def train_rl_agent(headless=True, resume=False):
    state_path = "mario_state.sav" if resume and os.path.exists("mario_state.sav") else None
    base_env = MarioEnv('SuperMarioLand.gb', render=not headless, state_path=state_path)
    logger.info(f"Base observation space: {base_env.observation_space}")
    env = Monitor(base_env)
    env = DummyVecEnv([lambda: base_env])
    env = VecTransposeImage(env)
    logger.info(f"Wrapped observation space: {env.observation_space}")
    model_path = "grok_mamba.zip"
    model = PPO.load(model_path, env=env) if resume and os.path.exists(model_path) else PPO(
        MambaPolicy,
        env,
        verbose=2,
        learning_rate=0.0001,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, base_env, model))
    logger.info("Signal handler registered for SIGINT")

    logger.info("Starting training...")
    try:
        model.learn(total_timesteps=1_000_000, callback=AutosaveCallback(base_env, model))
        model.save("grok_mamba")
        logger.info(f"=== Training Complete. Saved 'grok_mamba.zip'. Total timesteps: {model.num_timesteps} ===")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught! Saving state...")
        base_env.save_state("mario_state.sav")
        model.save("grok_mamba")
        logger.info("State and model saved via try-except. Exiting...")
        sys.exit(0)

    play_env = MarioEnv('SuperMarioLand.gb', render=True, state_path=state_path)
    play_env = Monitor(play_env)
    play_env = DummyVecEnv([lambda: play_env])
    play_env = VecTransposeImage(play_env)
    obs = play_env.reset()
    total_reward = 0
    for _ in range(5000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = play_env.step([action])
        total_reward += reward[0]
        play_env.render()
        if terminated[0] or truncated[0]:
            logger.info(f"Episode ended. Total Reward: {total_reward}, Steps: {info[0]['steps']}")
            obs = play_env.reset()
            total_reward = 0
    play_env.close()

def play(model_path="grok_mamba.zip", state_path="mario_state.sav"):
    try:
        env = MarioEnv('SuperMarioLand.gb', render=True, state_path=state_path if os.path.exists(state_path) else None)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecTransposeImage(env)
        model = PPO.load(model_path)
        logger.info("Model loaded successfully!")
        logger.info(f"Loaded model policy class: {model.policy.__class__.__name__}")
        
        signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, env, model))
        logger.info("Signal handler registered for SIGINT in play")

        obs = env.reset()
        total_reward = 0
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        prev_x = get_mario_position(env.envs[0].env.pyboy)[0]
        for step in range(5000):
            raw_action, _ = model.predict(obs, deterministic=True)
            logger.info(f"Step {step}, Raw prediction: {raw_action}, Type: {type(raw_action)}")
            if isinstance(raw_action, np.ndarray):
                action = raw_action.item() if raw_action.size == 1 else raw_action[0]
            elif isinstance(raw_action, torch.Tensor):
                action = raw_action.item() if raw_action.dim() == 0 else raw_action[0].item()
            else:
                raise ValueError(f"Unexpected action type: {type(raw_action)}")
            action_counts[action] += 1
            logger.info(f"Step {step}, Action predicted: {action}")
            current_x, current_y = get_mario_position(env.envs[0].env.pyboy)
            delta_x = current_x - prev_x
            logger.info(f"Step {step}, Mario position (x, y): ({current_x}, {current_y}), Delta x: {delta_x}")
            obs, reward, terminated, truncated, info = env.step([action])
            total_reward += reward[0]
            env.render()
            prev_x = current_x
            time.sleep(0.01)
            if terminated[0] or truncated[0]:
                logger.info(f"Episode ended at step {step}. Total Reward: {total_reward}, Steps: {info[0]['steps']}")
                logger.info(f"Action distribution: {action_counts}")
                obs = env.reset()
                total_reward = 0
                action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                prev_x = get_mario_position(env.envs[0].env.pyboy)[0]
        env.close()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in play! Saving state...")
        env.envs[0].env.save_state("mario_state.sav")
        model.save("grok_mamba")
        logger.info("State and model saved in play. Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during play: {e}")

if __name__ == "__main__":
    try:
        train_rl_agent(headless=True, resume=False)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in main! State should have been saved by train_rl_agent.")
        sys.exit(0)