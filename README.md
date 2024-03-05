# S<sup>2</sup>-Wrapper

This repo contains the Pytorch implementation of S<sup>2</sup>-Wrapper, a simple mechanism that enables multi-scale feature extraction on any vision model.


## Quickstart

**Step 1.** Clone this repo and install `s2wrapper` through pip

```bash
# go to the directory of this repo, and
pip install .
```

**Step 2.** Extract multi-scale feature on **any vision model** with **one line of code**

Assume you have a function (could be `model`, `model.forward`, _etc._) that takes in BxCxHxW images and outputs BxNxC features.

For example, you have `model` (_e.g._, ViT-B) that extracts feature by
```python
feature = model(x)   # e.g., x: 32*3*224*224, feature: 32*196*768
```

Then extract multi-scale features (_e.g._, scales of 1 and 2) by
```python
from s2wrapper import forward as multiscale_forward
mutliscale_feature = multiscale_forward(model, x, scales=[1, 2])   # x: 32*3*224*224, feature: 32*196*1536
```


## Example:  HuggingFace CLIP with S<sup>2</sup>Wrapper

Regular feature extraction using HuggingFace CLIP vision model (copied from [official example](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel.forward.example)):
```python
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output  # pooled CLS states
```
