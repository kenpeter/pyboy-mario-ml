import time
from pyboy import PyBoy

# Initialize PyBoy with the Super Mario Land ROM
pyboy = PyBoy('SuperMarioLand.gb', window="SDL2")

# Wait for the window to open
time.sleep(2)

# Move the window to a specific position (optional)
try:
    import win32gui
    import win32con

    # Find the PyBoy window by its title
    hwnd = win32gui.FindWindow(None, "PyBoy")
    if hwnd:
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, 100, 100, 0, 0, win32con.SWP_NOSIZE)
except ImportError:
    print("win32gui not installed. Skipping window repositioning.")

# Main loop to keep the window open
try:
    while True:
        if not pyboy.tick():  # Advance the emulator by one frame
            break  # Exit if the emulator is closed
        time.sleep(1 / 60)  # Limit the frame rate to ~60 FPS
finally:
    pyboy.stop()  # Clean up when the script exits