# ShearDetect
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6463119.svg)](https://doi.org/10.5281/zenodo.6463119)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6482460.svg)](https://doi.org/10.5281/zenodo.6482459)

A defekt detection model for shearographic images. This model is based on a object detection model with *faster R-CNN* and *ResNet-50* approach.

<img src="https://github.com/ILKGit/ShearDetect/blob/main/imgs/model_1x_001.png" data-canonical-src="https://github.com/ILKGit/ShearDetect/blob/main/imgs/model_1x_001.png" width="400" height="400" />


## Getting Started
### Clone the Code
```
git clone https://github.com/ILKGit/ShearDetect
```
### Requirements
* Python >3.6
* CUDA 11.3 or higher

Install all the python dependencies using pip
```
pip install -r requirements.txt
```


## Training
```
python train_model.py --model=NAME OF YOUR MODEL --epochs=NUMBER OF EPOCHS --save_period=CHECKPOINTS SAVE PERIOD
```

