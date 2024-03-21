# S<sup>2</sup>-Wrapper

This repo contains the Pytorch implementation of S<sup>2</sup>-Wrapper, a simple mechanism that enables multi-scale feature extraction on *any vision model*.

<div align="center">
  <image src="assets/s2_wrapper_2.png" width="840px" />
  <p></p>
</div>

Read our paper about when scaling on image scales is better than scaling on model size.

**When Do We Not Need Larger Vision Models?**<br>
*[Baifeng Shi](https://bfshi.github.io/), [Ziyang Wu](https://robinwu218.github.io/), [Maolin Mao](https://www.linkedin.com/in/maolin-mao-347469220/), [Xin Wang](https://xinw.ai/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/)*<br>
UC Berkeley, Microsoft Research<br>

Paper: [https://arxiv.org/abs/2403.13043](https://arxiv.org/abs/2403.13043)

## Quickstart

**Step 1.** Clone this repo and install `s2wrapper` through pip.

```bash
# go to the directory of this repo, and run 
pip install .
```

**Step 2.** Extract multi-scale feature on **any vision model** with **one line of code**.

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

## Usage

```python
s2wrapper.forward(
    model,
    input,
    scales=None,
    img_sizes=None,
    max_split_size=None,
    resize_output_to_idx=0,
    num_prefix_token=0,
    output_shape='bnc',
)
```

`model`: Your vision model or any function that takes in BxCxHxW image tensor and outputs BxNxC feature tensor.

`input`: Input image tensor with shape BxCxHxW.

`scales`: A list of scales to extract features on. For example, `scales=[1, 2]` will extract feature on 224<sup>2</sup> and 448<sup>2</sup> scales if default size is 224<sup>2</sup>.

`img_sizes`: Alternatively, instead of assigning `scales`, you can assign the image size for each scale. For example, `img_sizes=[224, 448]` will yeild with same results as `scales=[1, 2]` for default size of 224<sup>2</sup>.

`max_split_size`: The maximum size of sub-images splitted from the large image. For each scale, the image will be splitted into `ceil(img_size_that_scale / max_split_size)**2` sub-images. If `None`, set by default as the size of `input`.

`resize_output_to_idx`: Which scale to resize the final feature map to. Default is the first scale in `scales` or `img_sizes`.

`num_prefix_token`: Number of prefix tokens in the feature map. For example, if the feature map returned by `model` contains 1 \[CLS\] token and other spatial tokens, set `num_prefix_token=1`. Default is 0.

`output_shape`: Shape of the output features. Need to be either `bnc` (e.g., ViT) or `bchw` (e.g., ConvNet). Default is `bnc`.

## Example:  HuggingFace CLIP with S<sup>2</sup>Wrapper

Regular feature extraction using HuggingFace CLIP vision model (reference: [official example](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel.forward.example)):
```python
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt").pixel_values

# model.forward returns an object that contains "last_hidden_state" which is the feature map we need
outputs = model(inputs).last_hidden_state
print(outputs.shape)  # 1*50*768
```

Making it multi-scale:
```python
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt").pixel_values

# wrap the feature extraction process into a single function that
# takes image tensor as input and outputs feature tensor
def forward_features(inputs):
    return model(inputs).last_hidden_state

# extracting features with scales=[1, 2]. Note the output has one [CLS] token
# so setting num_prefix_token=1.
outputs = multiscale_forward(forward_feature, inputs, scales=[1, 2], num_prefix_token=1)
print(outputs.shape)  # 1*50*1536
```
