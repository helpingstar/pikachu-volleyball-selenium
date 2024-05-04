# selenium의 webdriver를 사용하기 위한 import
from selenium import webdriver

# selenium으로 키를 조작하기 위한 import
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

import numpy as np
import torch
from information import ACTION_TO_KEYS, OBS_HIGH, OBS_LOW
from network import Agent


class KeyController:
    def __init__(self, driver: webdriver.Chrome, player: int) -> None:
        self.previous_action = 0
        self.action_to_key = ACTION_TO_KEYS[player - 1]
        self.action_chain = ActionChains(driver)

    def _key_up(self, k: Keys | str):
        self.action_chain.key_up(k).perform()

    def _key_down(self, k: Keys | str):
        self.action_chain.key_down(k).perform()

    def press(self, action):
        # key_up previous keys
        for prev_key in self.action_to_key[self.previous_action]:
            self._key_up(prev_key)

        for key in self.action_to_key[action]:
            self._key_down(key)

        self.previous_action = action


class Commander:
    def __init__(self, driver: webdriver.Chrome, ai_side: int, weight_path: str) -> None:
        self.p1_is_ai = ai_side & 1 > 0
        self.p2_is_ai = ai_side & 2 > 0

        self.p1_controller = KeyController(driver, 1) if self.p1_is_ai else None
        self.p2_controller = KeyController(driver, 2) if self.p2_is_ai else None

        # get agent
        self.agent = Agent()
        self.agent.load_state_dict(torch.load(weight_path))
        self.agent.eval()

    def process_observation(self, raw_observation: str):
        raw_observation_arr = list(map(int, raw_observation.split(",")))
        p1_power_hit_key_is_down_previous = raw_observation_arr[7]
        p2_power_hit_key_is_down_previous = raw_observation_arr[16]
        p1_agent = raw_observation_arr[:7]
        p2_agent = raw_observation_arr[9:16]
        ball_observation = raw_observation_arr[-9:]
        p1_state = raw_observation_arr[8]
        p2_state = raw_observation_arr[17]
        p1_state_arr = [0, 0, 0, 0, 0]
        p2_state_arr = [0, 0, 0, 0, 0]
        if p1_state < 5:
            p1_state_arr[p1_state] = 1
        if p2_state < 5:
            p2_state_arr[p2_state] = 1

        if self.p1_is_ai:
            p1_observation = sum(
                [
                    p1_agent,
                    [p1_power_hit_key_is_down_previous],
                    p1_state_arr,
                    p2_agent,
                    [p2_power_hit_key_is_down_previous],
                    p2_state_arr,
                    ball_observation,
                ],
                [],
            )
            p1_observation = torch.Tensor(p1_observation)
        if self.p2_is_ai:
            p2_observation = sum(
                [
                    p2_agent,
                    [p2_power_hit_key_is_down_previous],
                    p2_state_arr,
                    p1_agent,
                    [p1_power_hit_key_is_down_previous],
                    p1_state_arr,
                    ball_observation,
                ],
                [],
            )
            p2_observation = torch.Tensor(p2_observation)

        if self.p1_is_ai and self.p2_is_ai:
            observation = torch.stack((p1_observation, p2_observation))
        elif self.p1_is_ai:
            observation = p1_observation
        elif self.p2_is_ai:
            observation = p2_observation

        observation = (observation - OBS_LOW) / (OBS_HIGH - OBS_LOW)

        return observation

    def act(self, raw_observation):
        observation = self.process_observation(raw_observation)
        action = self.agent.get_action(observation).to(torch.int)

        if self.p1_is_ai and self.p2_is_ai:
            self.p1_controller.press(action[0].item())
            self.p2_controller.press(action[1].item())
        elif self.p1_is_ai:
            self.p1_controller.press(action.item())
        elif self.p2_is_ai:
            self.p2_controller.press(action.item())


def get_html_text_by_id(driver: webdriver.Chrome, id: str):
    element = driver.find_element(By.ID, id)
    return element.text


if __name__ == "__main__":
    driver: webdriver.Chrome = webdriver.Chrome()
    driver.get("http://127.0.0.1:8080/en/")

    print("Waiting for the Play Button to be activated.")
    about_btn = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "about-btn")))
    about_btn.click()

    print("Wait for the game to start, (wait for the PikachuVolleyball object to be activated).")
    element = WebDriverWait(driver, 10).until(EC.text_to_be_present_in_element((By.ID, "state"), "intro"))

    print("Start the connection.")

    game_element = driver.find_element(By.ID, "game-canvas-container")

    print("Select the option and launch the game from the menu")

    ai_side = input(
        "Which player will the AI control? The AI controls must be a human-controllable player. \
                    \n[1] : left(player1)\n[2] : right(player2)\n[3] : both side\n"
    )

    ai_side = int(ai_side)
    weight_path = "weights/player_1_pika-zoo_5M.pth"
    commander = Commander(driver, ai_side, weight_path)

    prev_state = get_html_text_by_id(driver, "state")
    prev_raw_observation = get_html_text_by_id(driver, "observation")
    with torch.inference_mode():
        while True:
            state = get_html_text_by_id(driver, "state")
            raw_observation = get_html_text_by_id(driver, "observation")

            if (prev_raw_observation != raw_observation) and state == "round":
                commander.act(raw_observation)
                prev_state = state
                prev_raw_observation = raw_observation
