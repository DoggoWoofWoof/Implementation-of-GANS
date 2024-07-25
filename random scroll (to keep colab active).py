import pyautogui
import time
import random

def random_scroll():
    try:
        while True:
            # Scroll up or down by a larger random amount
            scroll_amount = random.randint(-50, 50)
            pyautogui.scroll(scroll_amount)
            # Pause for a shorter random interval between scrolls
            time.sleep(random.uniform(0.05, 0.3))
    except KeyboardInterrupt:
        print("Scrolling stopped by user")

# Run the random scroll indefinitely until stopped by the user
random_scroll()
