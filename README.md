# PyTorch Image Segmentation

PyTorch Image Segmentation for my study.

## Requirements

- CUDA 11.8
  
  - [CUDA Toolkit 11.8 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-11-8-0-download-archive)

- Python 3.11.4
  
  - [Python Release Python 3.11.4 | Python.org](https://www.python.org/downloads/release/python-3114/)

- PyTorch 2.0.1
  
  - https://pytorch.org/get-started/locally/
    
    ```python
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

- Other Python Libraries
  
  ```python
  pip install -r requirements.txt
  ```

## Dataset

- VOC2007: [The PASCAL Visual Object Classes Challenge 2007 (VOC2007)](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)

## Model

- UNet

## Training Code Example

- UNet
  
  ```python
  python train_UNet.py --config=./config/UNet_train.json
  ```

## Predict Code Example

- UNet
  
  ```python
  python pred_UNet.py --config=./config/UNet_pred.json
  ```

## Reference

- Olaf Ronneberger, et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." (2015). [[1505.04597] U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)