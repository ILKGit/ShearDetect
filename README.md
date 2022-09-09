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
## Dataset
A dataset can be find here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6463119.svg)](https://doi.org/10.5281/zenodo.6463119)

Strucutre of a custom Dataset has to be as following:
```
|-----train
       |-----annotations
              |-----*.json
       |-----images
              |-----*.tif
|-----validation
       |-----annotations
              |-----*.json
       |-----images
              |-----*.tif
|-----test
       |-----annotations
              |-----*.json
       |-----images
              |-----*.tif

*.json-files contain the following annotations and infos

{
"fileID": "fspecimen_name+image_name",
"Dataset": "specimen_name",
"image": "image_name",
"defect": [[x1, y1, x2, y2],],        #bounding box of defects as list
"specimen": [[x1, y1, x2, y2],].  #bounding box of specimens as list
}
```
## Training / Evaluation
```
python train_model.py --model=NAME OF YOUR MODEL --epochs=NUMBER OF EPOCHS --save_period=CHECKPOINTS SAVE PERIOD
```
## Detection
```
python detect_model.py --model=DIR to Model --data=DIR TO DATA --pred=DIR TO SAVE RESULTS
```
