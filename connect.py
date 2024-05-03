# selenium의 webdriver를 사용하기 위한 import
from selenium import webdriver

# selenium으로 키를 조작하기 위한 import
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains


def action_to_key_down(action: int):
    pass


def get_html_text_by_id(driver: webdriver.Chrome, id: str):
    element = driver.find_element(By.ID, id)
    return element.text


def process_observation(observation: str):
    observation_arr = list(map(int, observation.split(",")))
    p1_power_hit_key_is_down_previous = observation_arr[7]
    p2_power_hit_key_is_down_previous = observation_arr[16]
    p1_agent = observation_arr[:7]
    p2_agent = observation_arr[9:16]
    ball_observation = observation_arr[-9:]
    p1_state = observation_arr[8]
    p2_state = observation_arr[17]
    p1_state_arr = [0, 0, 0, 0, 0]
    p2_state_arr = [0, 0, 0, 0, 0]
    if p1_state < 5:
        p1_state_arr[p1_state] = 1
    if p2_state < 5:
        p2_state_arr[p2_state] = 1
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
    return p1_observation, p2_observation


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

    actions = ActionChains(driver)

    prev_state = get_html_text_by_id(driver, "state")
    prev_observation = get_html_text_by_id(driver, "observation")

    count = 0

    while True:
        state = get_html_text_by_id(driver, "state")
        observation = get_html_text_by_id(driver, "observation")

        if prev_observation != observation:
            pass
            # key press and key down

        prev_state = state
        prev_observation = observation
