# pikachu-volleyball-selenium

## Start

If the computer's processing speed is slow, it can degrade the agent's performance. Therefore, it is recommended to use a computer with good performance.

### Preparation

```cmd
git clone --recurse-submodules https://github.com/helpingstar/pikachu-volleyball-selenium.git
git submodule update --remote --recursive
```
### submodules

The submodules below are forks of each game or model. These repositories are created for personal research purposes and not for profit. The original sources are publicly available in the respective forked repositories. Since it is not possible to fork the same repository twice, a new repository was created for the DuckLL version.

The original code referenced is as follows. The submodules below are forked versions, and detailed descriptions of these repositories can be found in the original repositories.

* [gorisanson/pikachu-volleyball](https://github.com/gorisanson/pikachu-volleyball)
* [DuckLL/pikachu-volleyball](https://github.com/DuckLL/pikachu-volleyball)

Select the appropriate repository from the submodules below and execute the command listed in the respective section.

#### pikachu-volleyball-basic

* Users can match against a trained agent.
* You can match a trained agent against a basic AI.
* You can match trained agents against each other on the web (I recommend using pika-zoo).

```cmd
cd pikachu-volleyball-basic
npm install
npm run build
npx http-server dist
cd ..
python connect.py
```

#### pikachu-volleyball-duckll-fork

* A trained agent can match against the most powerful rule-based AI model currently available, DuckLL AI.

```cmd
cd pikachu-volleyball-duckll-fork
npm install
npm run build
npx http-server dist
cd ..
python connect.py
```
