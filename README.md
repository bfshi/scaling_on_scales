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


## News
- [2025/02] [NVILA](https://nvlabs.github.io/VILA/) uses Dynamic-S<sup>2</sup>, a new version of S<sup>2</sup>-Wrapper that supports dynamic aspect ratio, as one of its key ingredients. NVILA is a frontier MLLM that achieves superior performance over SOTA MLLMs such as Qwen2-VL across multiple benchmarks while being much faster (1.9x - 5.1x faster in training and 1.2x - 2.8x faster during inference).
<div align="center">
  <image src="assets/nvila_teaser.jpg" width="840px" />
  <p></p>
</div>
<div align="center">
  <image src="assets/nvila_performance.png" width="840px" />
  <p></p>
</div>
- [2024/07] Accepted to ECCV 2024!
- [2024/05] S<sup>2</sup>-Wrapper is officially integrated in NVIDIA [VILA](https://github.com/Efficient-Large-Model/VILA)! We've released the checkpoint for VILA-1.5-3b with S<sup>2</sup> and more checkpoints are on the way! Check it out [here](#example-nvidia-vila-with-s2-wrapper).
- [2024/04] S<sup>2</sup>-Wrapper is officially integrated in [LLaVA](https://github.com/haotian-liu/LLaVA)! We've released the checkpoint for LLaVA-1.5 with S<sup>2</sup>. Try it out [here](#example-llava-with-s2-wrapper).


## To-Dos

- [ ] Adding pre-trained checkpoints of LLaVA-NeXT with S<sup>2</sup>-Wrapper.
- [x] ~~Adding pre-trained checkpoints of LLaVA-1.5 with S<sup>2</sup>-Wrapper.~~
- [x] ~~Adding support for non-square images~~ Now supporting images of any shape. Please check in branch `dev_any_shape`. Feature still in test.
- [x] ~~Adding examples of LLaVA w/ S<sup>2</sup>-Wrapper~~ 


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



## Example: LLaVA with S<sup>2</sup>-Wrapper

S<sup>2</sup>-Wrapper is officially integrated into [LLaVA](https://github.com/haotian-liu/LLaVA) (see the PR [here](https://github.com/haotian-liu/LLaVA/pull/1376)). To use LLaVA with S<sup>2</sup>-Wrapper, simply install this repo and the latest version of LLaVA repo and download the checkpoints listed below. We've released the checkpoints of LLaVA-1.5-7B and LLaVA-1.5-13B with S<sup>2</sup>-Wrapper.


| Model | Size | Schedule | Checkpoint | VQAv2 | VizWiz | TextVQA | MMMU-val | MathVista | MM-Bench | SEED | MM-Vet |
|----------|----------|-----------|-----------|---|---|---|---|---|---|---|---|
| LLaVA-1.5 | 7B | full_ft-1e | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) | 78.5 | 50.0 | 58.2 | 36.2 | 25.2 | 64.3 | 65.7 | 31.1 |
| LLaVA-1.5 | 7B | lora-1e | [liuhaotian/llava-v1.5-7b-lora](https://huggingface.co/liuhaotian/llava-v1.5-7b-lora) | 79.1 | 47.8 | 58.2 | - | - | 66.1 | - | 30.2 |
| **LLaVA-1.5-S2** | 7B | lora-1e | [bfshi/llava-v1.5-7b-s2-lora](https://huggingface.co/bfshi/llava-v1.5-7b-s2-lora) | **80.0** | **50.1** | **61.0** | **37.7** | **25.3** | **66.2** | **67.9** | **32.4** |
| LLaVA-1.5 | 13B | full_ft-1e | [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) | 80.0 | 53.6 | 61.3 | 36.4 | 27.6 | 67.7 | 68.2 | 36.1 |
| LLaVA-1.5 | 13B | lora-1e | [liuhaotian/llava-v1.5-13b-lora](https://huggingface.co/liuhaotian/llava-v1.5-13b-lora) | 80.0 | 58.9 | 60.2 | - | - | 68.5 | - | 38.3 |
| **LLaVA-1.5-S2** | 13B | lora-1e | [bfshi/llava-v1.5-13b-s2-lora](https://huggingface.co/bfshi/llava-v1.5-13b-s2-lora) | **80.9** | 56.0 | **63.1** | **37.4** | **27.8** | 67.9 | **68.9** | 36.4 |

An example script of model inference with LLaVA-1.5-S2:
```bash
python3 -m llava.eval.run_llava \
    --model-path bfshi/llava-v1.5-7b-s2-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --image-file <image> \
    --query <query> \
    --conv-mode vicuna_v1
```

**Training**. To train LLaVA with S<sup>2</sup>-Wrapper, since the current LLaVA repo only supports evaluation with S<sup>2</sup>, please additionally apply the changes [here](https://github.com/bfshi/LLaVA_NeXT_S2_Integration/commit/f73528e265c54e871289f08533d08d72ad8fdfe8)
to your LLaVA repo and you are good to go! 

Training configurations should be the same as training a regular LLaVA **without** anyres (*i.e.*, `image_aspect_ratio="resize"` and `mm_patch_merge_type="flat"`), except for two new model configs:
- `s2=True`. This turns on the usage of S<sup>2</sup>.
- `s2_scales="336,672,1008"`. This specifies the image scales S<sup>2</sup> will extract features on.



## Example: NVIDIA VILA with S<sup>2</sup>-Wrapper

S<sup>2</sup>-Wrapper is officially integrated into NVIDIA [VILA](https://github.com/Efficient-Large-Model/VILA). VILA is a multi-modal LLM that supports multi-image understanding and video understanding
with superb results on multiple benchmarks (*e.g.*, ranked #1 on [MMMU](https://mmmu-benchmark.github.io/#leaderboard) among all open-source models). VILA comes with several model sizes: 3B, 8B, 13B, and 40B, each also with a quantized version (AWQ). 

Currently we've released the checkpoints of VILA-3B with S<sup>2</sup>-Wrapper which is your to-go choice for running MLLM on edge devices. Checkpoints of other model sizes are on the way! Meanwhile, welcome to check out more details [here](https://developer.nvidia.com/blog/visual-language-intelligence-and-edge-ai-2-0/).

| $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ | Prec. | VQAv2 | GQA  | VizWiz | SQA-I | VQA-T | POPE | MME     | MMB  | MMB-CN | SEED | SEED-I | MMMU (val) | MMMU (test) | llava-bench | MM-Vet | Average |
| -------------------------------- | ----- | ----- | ---- | ------ | ----- | ----- | ---- | ------- | ---- | ------ | ---- | ------ | ---------- | ----------- | ----------- | ------ | ------- |
| VILA1.5-3B                       | fp16  | 80.4  | 61.5 | 53.5   | 69.0  | 60.4  | 85.9 | 1442.44 | 63.4 | 52.7   | 60.9 | 67.9   | 33.3       | 30.8        | 75.9        | 35.4   | 60.2    |
| **VILA1.5-3B-S2**                | fp16  | 79.8  | 61.4 | 61.3   | 69.6  | 63.4  | 85.3 | 1431.65 | 62.8 | 52.2   | 60.0 | 66.4   | 32.8       | 31.3        | 76.7        | 38.6   | **60.9**|
| VILA1.5-3B-AWQ                   | int4  | 80.0  | 61.1 | 53.8   | 67.8  | 60.4  | 85.9 | 1437.34 | 63.3 | 51.4   | 59.8 | 66.6   | 32.7       | 31.1        | 75.0        | 37.3   | 59.9    |
| **VILA1.5-3B-S2-AWQ**            | int4  | 79.4  | 61.3 | 62.3   | 69.2  | 63.0  | 85.8 | 1417.06 | 61.6 | 51.5   | 59.1 | 65.7   | 33.4       | 30.4        | 77.1        | 36.7   | **60.5**|

Please refer to the [original repo](https://github.com/Efficient-Large-Model/VILA) of VILA for checkpoints as well as guidance on training, evaluation, and deployment.


## Example:  HuggingFace CLIP with S<sup>2</sup>-Wrapper

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
