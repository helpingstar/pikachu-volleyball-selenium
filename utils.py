from information import ACTION_TO_KEYS, OBS_HIGH, OBS_LOW
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import torch
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

    def all_key_up(self):
        for prev_key in self.action_to_key[self.previous_action]:
            self._key_up(prev_key)

    def press(self, action):
        # key_up previous keys
        for prev_key in self.action_to_key[self.previous_action]:
            self._key_up(prev_key)

        for key in self.action_to_key[action]:
            self._key_down(key)

        self.previous_action = action


class Commander:
    def __init__(
        self,
        driver: webdriver.Chrome,
        ai_side: int,
        p1_setting: dict[str : str | int],
        p2_setting: dict[str : str | int],
    ) -> None:
        # ai_side
        self.p1_is_ai = ai_side & 1 > 0
        self.p2_is_ai = ai_side & 2 > 0
        # setting
        self.p1_setting = p1_setting
        self.p2_setting = p2_setting

        self.p1_13map = (0, 1, 2, 3, 4, 6, 7, 10, 11, 12, 13, 14, 16)
        self.p2_13map = (0, 1, 2, 4, 3, 7, 6, 10, 12, 11, 13, 15, 17)

        self.p1_controller = KeyController(driver, 1) if self.p1_is_ai else None
        self.p2_controller = KeyController(driver, 2) if self.p2_is_ai else None
        self.prev_p1_score = 0
        self.prev_p2_score = 0
        # get agent
        if self.p1_is_ai:
            self.p1_agent = Agent(self.p1_setting)
            self.p1_agent.load_state_dict(torch.load(self.p1_setting["weight_path"]))
            self.p1_agent.eval()

        if self.p2_is_ai:
            self.p2_agent = Agent(self.p2_setting)
            self.p2_agent.load_state_dict(torch.load(self.p2_setting["weight_path"]))
            self.p2_agent.eval()

    def reset(self):
        self.prev_p1_score = 0
        self.prev_p2_score = 0

    def process_observation(self, raw_observation: str):
        # len(raw_observation_arr) == 29, 9(player1) + 9(player2) + 9(ball) + 2(score)
        raw_observation_arr = list(map(int, raw_observation.split(",")))
        p1_power_hit_key_is_down_previous = raw_observation_arr[7]
        p2_power_hit_key_is_down_previous = raw_observation_arr[16]
        p1_agent = raw_observation_arr[:7]
        p2_agent = raw_observation_arr[9:16]
        ball_observation = raw_observation_arr[18:27]
        p1_state = raw_observation_arr[8]
        p2_state = raw_observation_arr[17]
        p1_state_arr = [0, 0, 0, 0, 0]
        p2_state_arr = [0, 0, 0, 0, 0]
        if p1_state < 5:
            p1_state_arr[p1_state] = 1
        if p2_state < 5:
            p2_state_arr[p2_state] = 1

        p1_observation = None
        p2_observation = None

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
            p1_observation = (p1_observation - OBS_LOW) / (OBS_HIGH - OBS_LOW)
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
            p2_observation = (p2_observation - OBS_LOW) / (OBS_HIGH - OBS_LOW)

        p1_score, p2_score = raw_observation_arr[27], raw_observation_arr[28]
        rewards = [0, 0]
        if p1_score != self.prev_p1_score:
            rewards[0] = 1
            rewards[1] = -1
            self.prev_p1_score = p1_score
        elif p2_score != self.prev_p2_score:
            rewards[0] = -1
            rewards[1] = 1
            self.prev_p2_score = p2_score

        return p1_observation, p2_observation, rewards

    def all_key_up(self):
        if self.p1_controller is not None:
            self.p1_controller.all_key_up()
        if self.p2_controller is not None:
            self.p2_controller.all_key_up()

    def act(self, raw_observation):
        p1_observation, p2_observation, _ = self.process_observation(raw_observation)

        if self.p1_is_ai:
            p1_action = self.p1_agent.get_action(p1_observation).to(torch.int).item()
            if self.p1_setting["n_action"] == 13:
                p1_action = self.p1_13map[p1_action]
        if self.p2_is_ai:
            if self.p2_setting["n_action"] == 13:
                p2_action = self.p1_13map[p2_action]
            p2_action = self.p2_agent.get_action(p2_observation).to(torch.int).item()

        if self.p1_is_ai:
            self.p1_controller.press(p1_action)
        if self.p2_is_ai:
            self.p2_controller.press(p2_action)


class WebController:
    def __init__(self, driver: webdriver.Chrome, url, setting: dict[str, str | int]) -> None:
        self.driver = driver
        self.url = url
        self.setting = setting
        self.setting_submenu_btn_id = {key: f"{key}-submenu-btn" for key in self.setting.keys()}
        self.setting_desired_btn_id = {key: f"{key}-{value}-btn" for key, value in self.setting.items()}
        # exception, Overwrites existing data.
        self.setting_desired_btn_id["speed"] = f"{self.setting['speed']}-speed-btn"

    def _get_url(self):
        self.driver.get(self.url)

    def _click_start_button(self):
        print("Waiting for the Play Button to be activated.")
        about_btn = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.ID, "about-btn")))
        about_btn.click()

    def _wait_for_game_object(self):
        print("Wait for the game to start, (wait for the PikachuVolleyball object to be activated).")
        element = WebDriverWait(self.driver, 10).until(EC.text_to_be_present_in_element((By.ID, "state"), "intro"))

    def _setting(self):
        print("Setting...")
        # wait for dropdown btn
        options_btn = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.ID, "options-dropdown-btn")))
        options_btn.click()
        options_btn.click()
        for key in self.setting.keys():
            # click options_btn
            options_btn.click()
            # click submenu
            submenu_btn = self.driver.find_element(By.ID, self.setting_submenu_btn_id[key])
            submenu_btn.click()
            # click setting
            desired_btn = self.driver.find_element(By.ID, self.setting_desired_btn_id[key])
            desired_btn.click()

        print("Setting finish...")

    def initialize(self):
        self._get_url()
        self._click_start_button()
        self._wait_for_game_object()
        self._setting()

    def get_state_and_raw_observation(self):
        def _get_html_text_by_id(id: str):
            element = self.driver.find_element(By.ID, id)
            return element.text

        state = _get_html_text_by_id("state")
        raw_observation = _get_html_text_by_id("observation")

        return state, raw_observation
