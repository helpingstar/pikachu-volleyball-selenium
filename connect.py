from selenium import webdriver
import torch
from utils import WebController, Commander


if __name__ == "__main__":
    game_setting = {
        "graphic": "sharp",  # sharp, soft
        "bgm": "off",  # on, off
        "sfx": "off",  # stereo, mono, off
        "speed": "fast",  # speed, medium, fast
        "winning-score": 15,  # 5, 10, 15
        "practice-mode": "off",  # on, off
    }
    driver: webdriver.Chrome = webdriver.Chrome()
    url = "http://127.0.0.1:8080/en/"
    web_controller = WebController(driver, url, game_setting)

    web_controller.initialize()

    print("Select the option and launch the game from the menu")

    ai_side = input(
        "Which player will the AI control? The AI controls must be a human-controllable player. \
                    \n[1] : left(player1)\n[2] : right(player2)\n[3] : both side\n"
    )

    ai_side = int(ai_side)
    weight_path = "weights/cleanrl_ppo_vec_single_152580.pt"
    commander = Commander(driver, ai_side, weight_path)

    state, prev_raw_observation = web_controller.get_state_and_raw_observation()
    with torch.inference_mode():
        while True:
            state, raw_observation = web_controller.get_state_and_raw_observation()

            if state == "round":
                if prev_raw_observation != raw_observation:
                    commander.act(raw_observation)
                    prev_raw_observation = raw_observation
            else:
                commander.all_key_up()
