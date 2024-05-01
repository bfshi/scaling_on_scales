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


## To-Dos

- [x] ~~Adding examples of LLaVA w/ S<sup>2</sup>-Wrapper~~ Please see the PR [here](https://github.com/haotian-liu/LLaVA/pull/1376).
- [ ] Adding pre-trained checkpoints of LLaVA with S<sup>2</sup>-Wrapper.
- [x] ~~Adding support for non-square images~~ Now supporting images of any shape. Please check in branch `dev_any_shape`. Feature still in test.


<!-- ✅ ⬜️  -->


## Quickstart

**Step 1.** Install `s2wrapper` through pip.

```bash
pip install git+https://github.com/bfshi/scaling_on_scales.git
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

Above we assume the input is 224x224 and `s2wrapper` will interpolate it into 448x448. **If the original 448x448 image is already available, we can get better performance if we interpolate from the 448x448 image instead of the 224x224 image**. In this case, extract features at scales of 224x224 and 448x448 by
```python
from s2wrapper import forward as multiscale_forward
mutliscale_feature = multiscale_forward(model, x, scales=[0.5, 1], max_split_size=224)   # x: 32*3*448*448, feature: 32*196*1536, note that we need to set `max_split_size=224` to make it split the 448 image into 4 sub-images.
# mutliscale_feature = multiscale_forward(model, x, img_sizes=[224, 448], max_split_size=224)   # alternatively, set `img_sizes` instead of `scales`
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
    split_forward=False,
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

`split_forward`: Whether to run model on each sub-image separately or batch all sub-images into a single run. Setting to `True` can reduce memory usage (roughly the same GPU memory usage as single-scale during inference). Default is `False`.

## Example:  LLaVA with S<sup>2</sup>Wrapper

Please see the PR [here](https://github.com/haotian-liu/LLaVA/pull/1376).

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


## Citation

```
@article{shi2024we,
  title={When Do We Not Need Larger Vision Models?},
  author={Shi, Baifeng and Wu, Ziyang and Mao, Maolin and Wang, Xin and Darrell, Trevor},
  journal={arXiv preprint arXiv:2403.13043},
  year={2024}
}
```
