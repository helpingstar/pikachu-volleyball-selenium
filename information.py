import torch
from selenium.webdriver.common.keys import Keys


# https://github.com/helpingstar/pika-zoo?tab=readme-ov-file#action-space
ACTION_TO_KEYS: tuple[tuple[None | str | Keys]] = (
    (
        (),  # NOOP
        ("z"),  # FIRE
        ("r"),  # UP
        ("g"),  # RIGHT
        ("d"),  # LEFT
        ("v"),  # DOWN
        ("r", "g"),  # UP RIGHT
        ("r", "d"),  # UP LEFT
        ("v", "g"),  # DOWN RIGHT
        ("v", "d"),  # DOWN LEFT
        ("r", "z"),  # UP FIRE
        ("g", "z"),  # RIGHT FIRE
        ("d", "z"),  # LEFT FIRE
        ("v", "z"),  # DOWN FIRE
        ("r", "g", "z"),  # UP RIGHT FIRE
        ("r", "d", "z"),  # UP LEFT FIRE
        ("v", "g", "z"),  # DOWN RIGHT FIRE
        ("v", "d", "z"),  # DOWN LEFT FIRE
    ),
    (
        (),  # NOOP
        (Keys.ENTER),  # FIRE
        (Keys.UP),  # UP
        (Keys.RIGHT),  # RIGHT
        (Keys.LEFT),  # LEFT
        (Keys.DOWN),  # DOWN
        (Keys.UP, Keys.RIGHT),  # UP RIGHT
        (Keys.UP, Keys.LEFT),  # UP LEFT
        (Keys.DOWN, Keys.RIGHT),  # DOWN RIGHT
        (Keys.DOWN, Keys.LEFT),  # DOWN LEFT
        (Keys.UP, Keys.ENTER),  # UP FIRE
        (Keys.RIGHT, Keys.ENTER),  # RIGHT FIRE
        (Keys.LEFT, Keys.ENTER),  # LEFT FIRE
        (Keys.DOWN, Keys.ENTER),  # DOWN FIRE
        (Keys.UP, Keys.RIGHT, Keys.ENTER),  # UP RIGHT FIRE
        (Keys.UP, Keys.LEFT, Keys.ENTER),  # UP LEFT FIRE
        (Keys.DOWN, Keys.RIGHT, Keys.ENTER),  # DOWN RIGHT FIRE
        (Keys.DOWN, Keys.LEFT, Keys.ENTER),  # DOWN LEFT FIRE
    ),
)

OBS_LOW = torch.tensor(
    [
        32,
        108,
        -15,
        -1,
        -2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        32,
        108,
        -15,
        -1,
        -2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        20,
        0,
        0,
        0,
        0,
        0,
        -20,
        -124,
        0,
    ],
    dtype=torch.float32,
)

OBS_HIGH = torch.tensor(
    [
        400,
        244,
        16,
        1,
        3,
        4,
        4,
        1,
        1,
        1,
        1,
        1,
        1,
        400,
        244,
        16,
        1,
        3,
        4,
        4,
        1,
        1,
        1,
        1,
        1,
        1,
        432,
        252,
        432,
        252,
        432,
        252,
        20,
        124,
        1,
    ],
    dtype=torch.float32,
)
