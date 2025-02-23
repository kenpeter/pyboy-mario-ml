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
COINS = 0xFFFA  # Correct coin counter (BCD)

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

def press_start(pyboy):
    pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
    for _ in range(60):
        pyboy.tick()
    pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)

class MarioEnv(gym.Env):
    def __init__(self, rom_path, render=False):
        super(MarioEnv, self).__init__()
        self.rom_path = rom_path
        self.render_enabled = render
        self.pyboy = PyBoy(rom_path, window="SDL2" if render else "null")
        self.action_space = gym.spaces.Discrete(4)
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
        self.prev_lives = get_lives(self.pyboy)
        self.steps = 0

    def reset(self):
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
        self.prev_lives = get_lives(self.pyboy)
        self.steps = 0
        return self._get_observation()

    def step(self, action):
        stop_moving(self.pyboy)
        stop_jumping(self.pyboy)

        if action == 0:
            move_right(self.pyboy)
            for _ in range(4):
                self.pyboy.tick()
        elif action == 1:
            jump(self.pyboy)
            for _ in range(10):
                self.pyboy.tick()
            stop_jumping(self.pyboy)
        elif action == 2:
            move_right(self.pyboy)
            jump(self.pyboy)
            for _ in range(10):
                self.pyboy.tick()
            stop_jumping(self.pyboy)
        elif action == 3:
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
        screen_image = np.array(self.pyboy.screen.image)
        return screen_image[:, :, :3] if screen_image.shape[-1] == 4 else screen_image

    def _get_reward(self):
        mario_x, mario_y = get_mario_position(self.pyboy)
        current_coins = get_coins(self.pyboy)
        current_lives = get_lives(self.pyboy)
        progress_reward = (mario_x - self.prev_x) * 0.5
        coin_reward = (current_coins - self.prev_coins) * 20
        survival_reward = 1
        death_penalty = -10 if current_lives < self.prev_lives else 0
        jump_reward = 5 if mario_y < 80 else 0
        stall_penalty = -2 if mario_x == self.prev_x and mario_y >= 100 else 0
        return progress_reward + coin_reward + survival_reward + death_penalty + jump_reward + stall_penalty

    def _is_done(self):
        return get_lives(self.pyboy) == 0

    def render(self, mode='human'):
        if self.render_enabled and mode == 'human':
            pass
        elif mode == 'rgb_array':
            return self._get_observation()

    def close(self):
        self.pyboy.stop()

def main():
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
            move_right(pyboy)
            if mario_y >= 100 or (mario_x % 20 == 0):
                jump(pyboy)
                for _ in range(10):
                    pyboy.tick()
                stop_jumping(pyboy)
            current_lives = get_lives(pyboy)
            if current_lives < prev_lives:
                pass  # Removed debug
            prev_lives = current_lives
            if current_lives == 0:
                break  # Removed debug
            current_time = time.time()
            elapsed = current_time - last_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last_time = time.time()
    finally:
        pyboy.stop()

def train_rl_agent(headless=True):
    env = MarioEnv('SuperMarioLand.gb', render=not headless)  # headless=True means no UI, False means UI
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=0.0001,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )
    model.learn(total_timesteps=1000)
    model.save("mario_ppo_model")
    print("=== Training Complete. Saved 'mario_ppo_model.zip' ===")

    env = MarioEnv('SuperMarioLand.gb', render=True)  # Testing always with UI
    obs = env.reset()
    total_reward = 0
    for _ in range(2000):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        if done:
            print(f"Episode ended. Total Reward: {total_reward}, Steps: {info['steps']}")
            obs = env.reset()
            total_reward = 0
    env.close()

if __name__ == "__main__":
    main()
    # Switch between headless (True) and UI (False) mode here
    train_rl_agent(headless=True)  # Default to headless mode