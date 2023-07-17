# PyTorch Image Segmentation

PyTorch Image Segmentation for my study.

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