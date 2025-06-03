# PrejudiceAcrossProvince

## QuickStart
### Create Environment
```python
conda create -n PAP python==3.11
```
Then
```python
conda activate PAP
```

### Download model
```python
pip install modelscope
modelscope download --model AI-ModelScope/stable-diffusion-3.5-medium --local_dir ./models/sd3.5
modelscope download --model muse/Salesforce-blip-image-captioning-base --local_dir ./models/blip-image-captioning-base
```

## Install dependency
```python
pip install torch
# image generation
pip install diffusers transformers accelerate sentencepiece
# image analysis
pip install sentence-transformers keybert pillow
```

## Exection
Demo code is in txt2img.ipynb(generate image from text prompt) 
To analyze the image composition, you can use the code in analysis_img.ipynb

## Demo of analysis
## 🖼️ 示例效果展示
![Demo Image](./docs/astronaut_rides_horse.png)  
*<sup>图像描述：a man in a space suit riding a horse</sup>*

### 🔍 内容分析
| 关键词          | 置信度 |
|-----------------|--------|
| riding horse    | 0.23   |
| suit riding     | 0.22   |
| space suit      | 0.21   |
| man space       | 0.17   |
| horse           | 0.16   |

### 🎨 风格与情感分析
| 风格特征             | 强度 |
|----------------------|------|
| trending on ArtStation | 0.24 |
| concept art          | 0.23 |
| Unreal Engine        | 0.23 |
| futuristic feel      | 0.23 |
| centered composition | 0.22 |
| 3D rendering         | 0.21 |
| realistic photo      | 0.21 |
| dreamy atmosphere    | 0.21 |
| Octane render        | 0.21 |
| digital painting     | 0.21 |

