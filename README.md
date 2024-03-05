# S$^2$-Wrapper

This repo contains the Pytorch implementation of S$^2$-Wrapper, a simple mechanism that enables multi-scale feature extraction on any vision model.


## Quickstart

1. Clone this repo and install `s2wrapper` through pip

```bash
# go to the directory of this repo, and
pip install .
```

2. Extract multi-scale feature on any vision model with one line of code

Assume you have a function (could be `model`, `model.forward`, etc.) that takes in BxCxHxW images and outputs BxNxC features.

Taking `model` as an example, extracing regular feature would be 
```python
feature = model(x)
```

Then extract multi-scale features (e.g., scales of 1x and 2x) by
```python
from s2wrapper import forward as multiscale_forward

mutliscale_feature = multiscale_forward(model, x, scales=[1, 2])
```