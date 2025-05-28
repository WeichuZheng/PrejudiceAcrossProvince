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
modelscope download --model AI-ModelScope/stable-diffusion-3.5-medium --local_dir ./models
```

## Install dependency
```python
pip install torch
pip install diffusers transformers accelerate sentencepiece
```

### Exection
Demo code is in txt2img.ipynb
