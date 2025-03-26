import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import logging
import time
import os
import signal
import sys
import argparse
import gc

# Logging setup
def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Logging level set to: {logging.getLevelName(level)}")
    return logger

# Memory addresses
MARIO_X_POS = 0xC202
MARIO_Y_POS = 0xC201
LIVES = 0xDA15
COINS = 0xFFFA
WORLD_LEVEL = 0xFFB4

# PyBoy controls
def get_mario_position(pyboy): return pyboy.memory[MARIO_X_POS], pyboy.memory[MARIO_Y_POS]
def get_lives(pyboy): return pyboy.memory[LIVES]
def move_right(pyboy): pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
def stop_moving(pyboy): 
    pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
    pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
def jump(pyboy): pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
def stop_jumping(pyboy): pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
def press_start(pyboy):
    pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
    for _ in range(30): pyboy.tick()
    pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)

class MarioEnv(gym.Env):
    def __init__(self, rom_path, render=False):
        super().__init__()
        self.rom_path = rom_path
        self.render_enabled = render
        self.pyboy = PyBoy(rom_path, window="SDL2" if render else "null", sound=False)
        self.action_space = gym.spaces.Discrete(3)  # 0=Right, 1=Jump, 2=Right+Jump
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        self.frame_skip = 4
        self.closed = False
        
        press_start(self.pyboy)
        for i in range(300):
            self.pyboy.tick()
            if i == 150: press_start(self.pyboy)
            if get_lives(self.pyboy) > 0: break
        self.initial_lives = get_lives(self.pyboy)

        self.prev_x = get_mario_position(self.pyboy)[0]
        self.prev_y = get_mario_position(self.pyboy)[1]
        self.prev_lives = get_lives(self.pyboy)
        self.steps = 0
        self.total_reward = 0.0
        self.stall_counter = 0
        self.jump_frames = 0
        self.epsilon = 1.0

    def reset(self, seed=None, options=None):
        if not self.closed:
            self.pyboy.stop(save=False)
        self.pyboy = PyBoy(self.rom_path, window="SDL2" if self.render_enabled else "null", sound=False)
        press_start(self.pyboy)
        for i in range(300):
            self.pyboy.tick()
            if i == 150: press_start(self.pyboy)
            if get_lives(self.pyboy) > 0: break
        
        self.prev_x = get_mario_position(self.pyboy)[0]
        self.prev_y = get_mario_position(self.pyboy)[1]
        self.prev_lives = get_lives(self.pyboy)
        self.steps = 0
        self.total_reward = 0.0
        self.stall_counter = 0
        self.jump_frames = 0
        self.epsilon = max(0.1, self.epsilon - 0.000045)
        return self._get_observation(), {"steps": self.steps}

    def step(self, action):
        stop_moving(self.pyboy)
        if self.jump_frames == 0: stop_jumping(self.pyboy)
        
        if np.random.random() < self.epsilon:
            action = np.random.randint(3)
            logger.debug(f"Epsilon-greedy: Forced action={action}")

        if self.stall_counter > 3:
            action = 2
            self.jump_frames = 20
            logger.debug("Stall detected, forcing action=2")

        reward = 0
        for frame in range(self.frame_skip):
            if action == 0:
                move_right(self.pyboy)
                self.pyboy.tick()
            elif action == 1:
                if frame == 0: self.jump_frames = 20
                if self.jump_frames > 0:
                    jump(self.pyboy)
                    self.jump_frames -= 1
                self.pyboy.tick()
            elif action == 2:
                move_right(self.pyboy)
                if frame == 0: self.jump_frames = 20
                if self.jump_frames > 0:
                    jump(self.pyboy)
                    self.jump_frames -= 1
                self.pyboy.tick()
            mario_x, mario_y = get_mario_position(self.pyboy)
            reward += self._get_reward(action, mario_x, mario_y)
            if self._is_done(): break
        
        if self.jump_frames == 0: stop_jumping(self.pyboy)
        stop_moving(self.pyboy)
        
        self.steps += 1
        observation = self._get_observation()
        self.total_reward += reward
        terminated = self._is_done()
        current_x = get_mario_position(self.pyboy)[0]
        current_y = get_mario_position(self.pyboy)[1]
        
        self.prev_x = current_x
        self.prev_y = current_y
        self.prev_lives = get_lives(self.pyboy)
        info = {"steps": self.steps, "x_pos": current_x}
        return observation, reward, terminated, False, info

    def _get_observation(self):
        screen = np.array(self.pyboy.screen.ndarray)[..., :3]
        return screen

    def _get_reward(self, action, mario_x, mario_y):
        progress_reward = 10.0 if mario_x > self.prev_x else 0
        jump_reward = 50.0 if mario_y < self.prev_y and action == 2 else 0
        death_penalty = -100.0 if get_lives(self.pyboy) < self.prev_lives else 0
        
        if mario_x == self.prev_x:
            self.stall_counter += 1
        else:
            self.stall_counter = 0
        stall_penalty = -20.0 if self.stall_counter > 3 else 0
        
        total_reward = progress_reward + jump_reward + death_penalty + stall_penalty
        logger.debug(f"Reward: progress={progress_reward}, jump={jump_reward}, death={death_penalty}, stall={stall_penalty}, total={total_reward}")
        return total_reward

    def _is_done(self):
        return get_lives(self.pyboy) == 0

    def close(self):
        if not self.closed:
            self.pyboy.stop(save=False)
            self.closed = True

    def save_state(self, path):
        if not self.closed:
            with open(path, 'wb') as f:
                self.pyboy.save_state(f)
            logger.info(f"Saved state to {path}")

    def load_state(self, path):
        if os.path.exists(path) and not self.closed:
            with open(path, 'rb') as f:
                self.pyboy.load_state(f)
            logger.info(f"Loaded state from {path}")
            self.prev_x = get_mario_position(self.pyboy)[0]
            self.prev_y = get_mario_position(self.pyboy)[1]
            self.prev_lives = get_lives(self.pyboy)
            return True
        return False

class CNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(3136, features_dim)
        self._features_dim = features_dim

    def forward(self, x):
        x = x.float() / 255.0
        return self.fc(self.cnn(x))

should_exit = False

def signal_handler(sig, frame, env, model, model_path):
    global should_exit
    if should_exit:
        logger.debug("Signal handler called again, ignoring...")
        return
    should_exit = True
    logger.info("Ctrl+C detected! Saving state and model...")
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        if env:
            monitor_env = env.envs[0]
            frame_stack_env = monitor_env.env
            mario_env = frame_stack_env.unwrapped
            mario_env.save_state("mario_state.sav")
            env.close()
        if model:
            model.save(model_path)
        logger.info(f"Saved model to {model_path} and state to mario_state.sav")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during save: {e}")
        sys.exit(1)

def train(render=False, model_path="mario_ppo.zip", use_cuda=False, resume=False):
    global logger, should_exit
    base_env = MarioEnv('SuperMarioLand.gb', render=render)
    base_env = GrayScaleObservation(base_env, keep_dim=False)
    base_env = ResizeObservation(base_env, (84, 84))
    base_env = FrameStack(base_env, num_stack=4)
    env = Monitor(base_env)
    env = DummyVecEnv([lambda: env])
    
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    total_timesteps = 500_000
    completed_timesteps = 0

    if resume and os.path.exists(model_path):
        model = PPO.load(model_path, env=env, device=device)
        logger.info(f"Resumed model from {model_path}")
        if base_env.unwrapped.load_state("mario_state.sav"):
            completed_timesteps = model.num_timesteps  # Approximate from model
            logger.info(f"Resuming from {completed_timesteps} timesteps")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs={"features_extractor_class": CNNExtractor, "features_extractor_kwargs": {"features_dim": 256}},
            learning_rate=2.5e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=5,
            ent_coef=0.1,
            device=device
        )
        logger.info("Starting fresh training")

    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, env, model, model_path))
    logger.info("Starting training with progress bar...")
    try:
        remaining_timesteps = total_timesteps - completed_timesteps
        model.learn(total_timesteps=remaining_timesteps, progress_bar=True, reset_num_timesteps=False)
        model.save(model_path)
        logger.info(f"Training complete. Saved model to {model_path}")
    except KeyboardInterrupt:
        logger.info("Training interrupted by Ctrl+C, handled by signal handler")
    finally:
        env.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def play(model_path="mario_ppo.zip", use_cuda=False):
    global logger
    env = MarioEnv('SuperMarioLand.gb', render=True)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, num_stack=4)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model = PPO.load(model_path, env=env, device=device)
    obs = env.reset()
    total_reward = 0
    for step in range(5000):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        env.render()
        logger.debug(f"Step {step}: X={info[0]['x_pos']}, Action={action}, Reward={reward}")
        if done:
            logger.info(f"Episode ended. Total Reward: {total_reward}")
            obs = env.reset()
            total_reward = 0
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play Mario RL agent")
    parser.add_argument('--model_path', type=str, default="mario_ppo.zip", help="Path to save/load model")
    parser.add_argument('--render', action='store_true', help="Render the game during training/play")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    parser.add_argument('--play', action='store_true', help="Play mode instead of training")
    parser.add_argument('--cuda', action='store_true', help="Use CUDA if available")
    parser.add_argument('--resume', action='store_true', help="Resume training from saved model and state")
    args = parser.parse_args()

    logger = setup_logging(args.debug)

    if args.play:
        play(model_path=args.model_path, use_cuda=args.cuda)
    else:
        train(render=args.render, model_path=args.model_path, use_cuda=args.cuda, resume=args.resume)