# Import PyBoy library to emulate the Game Boy and run the Super Mario Land ROM
from pyboy import PyBoy
# Import WindowEvent from PyBoy utils to send input events like button presses
from pyboy.utils import WindowEvent
# Import time module for controlling frame rate and timing in the simulation
import time
# Import NumPy for handling screen data as multidimensional arrays (e.g., RGB images)
import numpy as np
# Import Gym to create a reinforcement learning environment compatible with RL frameworks
import gym
# Import PPO (Proximal Policy Optimization) from Stable Baselines3 for RL training
from stable_baselines3 import PPO

# Define memory addresses specific to Super Mario Land for accessing game state
# Address for Mario's horizontal (X) position on the screen
MARIO_X_POS = 0xC202
# Address for Mario's vertical (Y) position (lower values mean higher on screen)
MARIO_Y_POS = 0xC201
# Address for Mario's remaining lives
LIVES = 0xDA15
# Address for the number of coins Mario has collected
COINS = 0xDA1D

# Function to retrieve Mario's current X and Y position from memory
def get_mario_position(pyboy):
    # Return a tuple of (X, Y) coordinates by reading specified memory addresses
    return pyboy.memory[MARIO_X_POS], pyboy.memory[MARIO_Y_POS]

# Function to get Mario's current number of lives
def get_lives(pyboy):
    # Read and return the value at the LIVES memory address
    return pyboy.memory[LIVES]

# Function to get Mario's current coin count
def get_coins(pyboy):
    # Read and return the value at the COINS memory address
    return pyboy.memory[COINS]

# Function to simulate pressing the right arrow key
def move_right(pyboy):
    # Send the PRESS_ARROW_RIGHT event to move Mario right
    pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)

# Function to simulate releasing the right arrow key
def stop_moving(pyboy):
    # Send the RELEASE_ARROW_RIGHT event to stop horizontal movement
    pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)

# Function to simulate pressing the A button for jumping
def jump(pyboy):
    # Send the PRESS_BUTTON_A event to make Mario jump
    pyboy.send_input(WindowEvent.PRESS_BUTTON_A)

# Function to simulate releasing the A button
def stop_jumping(pyboy):
    # Send the RELEASE_BUTTON_A event to end Mario's jump
    pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)

# Simple AI function to control Mario with rule-based logic
def simple_ai(pyboy):
    # Get Mario's current X and Y coordinates
    mario_x, mario_y = get_mario_position(pyboy)
    # Get the current number of coins for tracking purposes
    current_coins = get_coins(pyboy)
    
    # Move Mario right continuously to progress through the level
    move_right(pyboy)
    
    # Check if Mario is on the ground (Y >= 100 is a heuristic for ground level)
    if mario_y >= 100:
        # Start jumping to reach coins or question marks above
        jump(pyboy)
        # Loop to hold the jump for 10 frames, ensuring a high jump
        for _ in range(10):
            # Advance the emulator by one frame per iteration
            pyboy.tick()
        # Release the jump button after 10 frames
        stop_jumping(pyboy)
    
    # Store Mario's X position before advancing a frame
    prev_x = mario_x
    # Advance the emulator by one frame to check movement
    pyboy.tick()
    # Get Mario's new X position after the frame
    new_x = pyboy.memory[MARIO_X_POS]
    # Check if X progress is slow (less than 2 units) and Mario is on the ground
    if new_x - prev_x < 2 and mario_y >= 100:
        # Jump to avoid a potential enemy blocking Mario’s path
        jump(pyboy)
        # Hold the jump for 10 frames to clear the obstacle
        for _ in range(10):
            # Advance the emulator one frame per iteration
            pyboy.tick()
        # Release the jump button after avoiding the enemy
        stop_jumping(pyboy)

# Define a custom Gym environment class for reinforcement learning
class MarioEnv(gym.Env):
    # Initialize the environment with a ROM path and rendering option
    def __init__(self, rom_path, render=False):
        # Call the parent Gym Env constructor to set up the base environment
        super(MarioEnv, self).__init__()
        # Store the path to the Super Mario Land ROM file
        self.rom_path = rom_path
        # Store whether rendering (visual output) is enabled
        self.render = render
        # Initialize PyBoy with the ROM; use SDL2 for rendering if enabled, else null (headless)
        self.pyboy = PyBoy(rom_path, window="SDL2" if render else "null")
        # Define the action space: 4 discrete actions (right, jump, right+jump, nothing)
        self.action_space = gym.spaces.Discrete(4)
        # Define the observation space: 144x160x3 RGB screen (Game Boy resolution)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        # Initialize previous coin count to track changes
        self.prev_coins = 0
        # Initialize previous X position to track progress
        self.prev_x = 0

    # Reset the environment to start a new episode
    def reset(self):
        # Stop the current PyBoy instance without saving
        self.pyboy.stop(save=False)
        # Reinitialize PyBoy with the ROM, setting rendering based on self.render
        self.pyboy = PyBoy(self.rom_path, window="SDL2" if self.render else "null")
        # Set initial previous coin count for reward calculation
        self.prev_coins = get_coins(self.pyboy)
        # Set initial previous X position for reward calculation
        self.prev_x = get_mario_position(self.pyboy)[0]
        # Return the initial screen observation
        return self._get_observation()

    # Execute an action in the environment and return the new state
    def step(self, action):
        # Release any held movement inputs to reset state
        stop_moving(self.pyboy)
        # Release any held jump inputs to reset state
        stop_jumping(self.pyboy)

        # Action 0: Move right
        if action == 0:
            # Press the right arrow to move Mario
            move_right(self.pyboy)
            # Advance the emulator 4 frames to simulate movement
            for _ in range(4):
                self.pyboy.tick()
        # Action 1: Jump
        elif action == 1:
            # Press the A button to jump
            jump(self.pyboy)
            # Hold jump for 10 frames to reach higher objects
            for _ in range(10):
                self.pyboy.tick()
            # Release the jump button
            stop_jumping(self.pyboy)
        # Action 2: Move right and jump
        elif action == 2:
            # Press the right arrow to move
            move_right(self.pyboy)
            # Press the A button to jump
            jump(self.pyboy)
            # Hold jump for 10 frames while moving
            for _ in range(10):
                self.pyboy.tick()
            # Release the jump button
            stop_jumping(self.pyboy)
        # Action 3: Do nothing
        elif action == 3:
            # Advance 4 frames without input to simulate idling
            for _ in range(4):
                self.pyboy.tick()

        # Get the current screen observation after the action
        observation = self._get_observation()
        # Calculate the reward based on the new state
        reward = self._get_reward()
        # Check if the episode is done (e.g., Mario died)
        done = self._is_done()
        
        # Update previous coin count for next reward calculation
        self.prev_coins = get_coins(self.pyboy)
        # Update previous X position for next reward calculation
        self.prev_x = get_mario_position(self.pyboy)[0]
        
        # Return observation, reward, done flag, and empty info dict (Gym standard)
        return observation, reward, done, {}

    # Get the current screen as an RGB array
    def _get_observation(self):
        # Retrieve the screen data from PyBoy’s screen manager as a NumPy array
        return self.pyboy.botsupport_manager().screen().screen_ndarray()

    # Calculate the reward based on game state changes
    def _get_reward(self):
        # Get Mario’s current position
        mario_x, mario_y = get_mario_position(self.pyboy)
        # Get the current coin count
        current_coins = get_coins(self.pyboy)
        
        # Reward for moving right (scaled down to prioritize other goals)
        progress_reward = (mario_x - self.prev_x) * 0.1
        # Big reward for collecting coins (10 points per coin)
        coin_reward = (current_coins - self.prev_coins) * 10
        # Penalty if Mario loses all lives (game over)
        death_penalty = -50 if get_lives(self.pyboy) == 0 else 0
        # Reward for being high up (Y < 80), encouraging question mark hits
        jump_reward = 2 if mario_y < 80 else 0
        # Penalty for stalling on the ground (possible enemy collision)
        stall_penalty = -5 if mario_x == self.prev_x and mario_y >= 100 else 0
        
        # Combine all reward components into a total reward
        return progress_reward + coin_reward + death_penalty + jump_reward + stall_penalty

    # Check if the episode is finished (Mario has no lives left)
    def _is_done(self):
        # Return True if lives reach 0, False otherwise
        return get_lives(self.pyboy) == 0

    # Render the current screen if enabled
    def render(self, mode='human'):
        # If rendering is enabled and mode is human, display the screen
        if self.render and mode == 'human':
            self.pyboy.botsupport_manager().screen().screen_ndarray()
        # If mode is rgb_array, return the screen array without displaying
        elif mode == 'rgb_array':
            return self._get_observation()

    # Clean up by stopping the emulator
    def close(self):
        # Shut down PyBoy to free resources
        self.pyboy.stop()

# Main function to run the simple AI with visual output
def main():
    # Initialize PyBoy with the ROM and enable rendering
    pyboy = PyBoy('SuperMarioLand.gb', window="SDL2")
    # Set target frame rate to 60 FPS
    target_fps = 60
    # Calculate the time per frame (1/60 seconds)
    frame_time = 1 / target_fps
    # Record the start time for timing control
    last_time = time.time()

    # Run the emulator loop with error handling
    try:
        # Continue looping until the emulator stops (e.g., window closed)
        while True:
            # Advance one frame; break if tick returns False (emulator stopped)
            if not pyboy.tick():
                break
            # Run the simple AI logic for this frame
            simple_ai(pyboy)
            # Get the current time for timing adjustment
            current_time = time.time()
            # Calculate elapsed time since last frame
            elapsed = current_time - last_time
            # Sleep if elapsed time is less than target frame time to maintain 60 FPS
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            # Update last time for the next iteration
            last_time = time.time()
            # Get Mario’s current position, lives, and coins for debugging
            mario_x, mario_y = get_mario_position(pyboy)
            lives = get_lives(pyboy)
            coins = get_coins(pyboy)
            # Print current game state to the console
            print(f"Mario Position: ({mario_x}, {mario_y}), Lives: {lives}, Coins: {coins}")
    # Ensure PyBoy stops even if an error occurs
    finally:
        # Clean up by stopping the emulator
        pyboy.stop()

# Function to train and test the RL agent with visible logs
def train_rl_agent():
    # Print a clear marker to indicate training is starting
    print("=== Starting Reinforcement Learning Training ===")
    # Create a headless environment for training (no rendering for speed)
    env = MarioEnv('SuperMarioLand.gb', render=False)
    # Initialize PPO with CNN policy and verbose logging for training updates
    model = PPO("CnnPolicy", env, verbose=1)
    # Train the model for 200,000 timesteps, showing logs due to verbose=1
    model.learn(total_timesteps=200000)
    # Save the trained model to a file
    model.save("mario_ppo_model")
    # Print a marker to indicate training is complete and model is saved
    print("=== Training Complete. Model Saved as 'mario_ppo_model.zip' ===")

    # Print a marker to indicate testing phase is starting
    print("=== Starting Reinforcement Learning Testing ===")
    # Create a rendered environment for testing
    env = MarioEnv('SuperMarioLand.gb', render=True)
    # Reset to get initial observation
    obs = env.reset()
    # Run 1000 test steps to demonstrate the trained agent
    for _ in range(1000):
        # Predict the next action based on the current observation
        action, _states = model.predict(obs)
        # Execute the action and get new state, reward, and done flag
        # the step is inside training
        obs, rewards, done, info = env.step(action)
        # Render the screen to visualize Mario’s actions
        env.render()
        # Optional: Print position and reward for each test step (remove if too verbose)
        mario_x, mario_y = get_mario_position(env.pyboy)
        print(f"Test - Mario Position: ({mario_x}, {mario_y}), Reward: {rewards}")
        # Reset if episode ends (Mario dies)
        if done:
            obs = env.reset()
    # Close the environment to free resources
    env.close()
    # Print a marker to indicate testing is complete
    print("=== Testing Complete ===")

# Entry point of the script
if __name__ == "__main__":
    # Run the simple AI first
    main()
    # Run the RL training and testing after main() completes
    train_rl_agent()