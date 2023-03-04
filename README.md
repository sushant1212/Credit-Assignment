# Credit-Assignment

## Environment:
* It is a modification of the [Simple Spread](https://pettingzoo.farama.org/environments/mpe/simple_spread/) environment from PettingZoo
* Usage:
```python
import simple_spread_custom
import numpy as np

env = simple_spread_custom.env(max_cycles=100, render_mode="human")
env.reset()
done = False
for agent in env.agent_iter():
    global_state = env.state()  # if you want to use the global state
    obs, rew, termination, truncation, info = env.last()
    if truncation:
        env.step(None)
    else:
        env.step(np.random.randint(1, 5))  # taking a random action

```