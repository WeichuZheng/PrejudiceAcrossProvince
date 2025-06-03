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

### Exection
Demo code is in txt2img.ipynb(generate image from text prompt) 
To analyze the image composition, you can use the code in analysis_img.ipynb
