# apple_photo_ct_workflow
This repository contains the Python code used in the paper (publication in progress) _"Detecting internal disorders in fruit by CT. Part 1: Joint 2D to 3D image registration workflow for comparing multiple slice photographs and CT scans of apple fruit"_. The paper describes a workflow for acquiring slice photographs and registered CT slices of apple fruit in four steps: data acquisition, image segmentation, image registration, and validation.

## Running the code

### Cloning the repository
To clone the repository with submodules use the following command:
```
git clone --recurse-submodules git@github.com:D1rk123/apple_photo_ct_workflow.git
```

### Conda environment
Two conda environments were used to run the code in this repository. Most code uses the *asr* environemnt, but for the reconstruction code the *flexbox* environment was used. To create these conda environment, follow the instructions in conda environment/create_environment.txt or conda environment/create_environment_flexbox.txt. The exact environments are also described in .yml files in the same directory. 

### Dataset
The code was written for registration of a dataset of CT scans and parallel slice photographs of Kanzi apples. This dataset is publicly available on [Zenodo](https://zenodo.org/record/8167285).

### Folder locations script
To run the scripts you need to create an extra script *folder_locations.py* that contains two functions: get\_data\_folder() and get\_results\_folder(). The path returned by get\_data\_folder() has to contain the data i.e. the CT scans and slice photographs. The results will be saved in the path returned by get\_results\_folder(). For example:
```python
from pathlib import Path

def get_data_folder():
    return Path.home() / "scandata" / "apple_slice_registration"
    
def get_results_folder():
    return Path.home() / "experiments" / "apple_slice_registration"
```

### Scripts
Different scripts were used at different parts of the workflow. Here is an overview:

**Data acquisition:**
- (*flexbox* environment) _calc\_flexbox\_bh\_spectrum.py_ - Calculate the spectrum for beam hardening correction
- (*flexbox* environment) _recon\_bh.py_ - Make FDK reconstructions with beam hardening correction
- _click\_apple\_bbox.py_ - Simple user interface to annotate the bounding box of each apple by four clicks
- _process\_apple\_bboxes.py_ - Use the bounding boxes to crop the photgraphs

**Image segmentation:**
- _segment\_apple\_ct.py_ - Segment the CT scans
- _crop_recons.py_ - Crop the CT reconstructions based on their segmentation mask to reduce their size
- _segment\_picture\_slice\_area\_MSD.py_ - Train a neural network for segmenting the slice photographs
- _apply\_segment\_picture\_slice\_area.py_ - Apply a trained neural network instance
- _combine\_nn\_outputs.py_ - Combine the outputs of multiple neural network instances 

**Image registration:**
- _register\_kanzi\_dataset.py_ - Perform image registration on the Kanzi dataset
- _photo\_ct\_registration.py_ - Code for performing the initialization and full optimization of the image registration
- _cost\_functions.py_ - Code for the cost functions
- _transformation\_models.py_ - Code for the transformation model
- _image\_coordinate\_utils.py_ - Code for sampling images
- _calc\_kanzi\_CT\_slices.py_ - Sample the registered CT slices given a fitted transformation model

**Evaluation:**
- _evaluate\_segmentation.py_ - Calculate segmentation metrics
- _click\_core\_endpoints.py_ - Simple user interface to annotate core endpoints on photos
- _click\_ct\_core\_endpoints.py_ - Simple user interface to annotate core endpoints on the registered CT slices
- _calc\_annotation\_distances.py_ - Calculate the distances between annotations

**Method evaluation:**
- _register\_kanzi\_dataset\_separate.py_ - Perform image registration on the Kanzi dataset separately for each slice
- _evaluate\_intersections.py_ - Test for intersecting or incorectly ordered slices
- _register\_kanzi\_dataset\_subsets.py_ - Perform image registration on the Kanzi dataset using subsets of different sizes

**Extra**
- _resize\_images\_for\_publishing.py_ - Resize the result images and convert them to suitable file formats for the paper

The code was written as a proof of concept on one dataset, and has therefore not been optimized for ease of use. If you want to run (parts of) the code and need help, you can contact me by email. You can find my email address on my [CWI contact page](https://www.cwi.nl/en/people/dirk-schut/).

