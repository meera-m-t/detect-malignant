# detect-malignant

## Getting Started
## Prerequisites

Ensure you have all the required packages installed:

```bash
pip install -e .
```

## Running the Code

For training: 

Execute the following command to run the system:

```bash
python -m detect_malignant --config batch/config.json
```
For testing :


```bash
python -m  detect_malignant --exp-name  experiments/example/ --mode test 
```


## Key Features

    - Customized Dataset Handling: Tailored to work with a specific medical image dataset.
    - Fastai Library: Utilizes the Fastai library for streamlined model training and fine-tuning.
    - Pydantic for Configuration: Uses Pydantic for robust data validation and configuration management.
    - Segmentation: Implements  segmentation use cv2 library.
    - YOLOv5 Integration: Modified YOLOv5 code to generate  metric reports and CM.
    - Preprocessing Techniques: Includes preprocessing steps such as Region of Interest (ROI) extraction and hair removal.
    - Clustering for Preprocessing: Clusters data into multiple groups, each requiring different preprocessing methods. More details in preprocessing-ideas.ipynb.

## Project Structure

    batch/config.py: Configuration file for batch processing.
    experiments/: Folder containing all experimental setups and results.
    preprocessing-ideas.ipynb: Jupyter notebook detailing preprocessing ideas and methods.

## Preprocessing Overview

Our preprocessing pipeline focuses on extracting the ROI and removing hair from the images. However, the preprocessing methods need further refinement. We believe that clustering the dataset into various groups, each with tailored preprocessing, will improve model performance. These methods need to be discussed with specialists in medical imaging for better insights and to understand the target goals more clearly.



## Docker Run
### Build

To build the Docker image, use the following command:

```bash
docker build -t detect-malignant .

```

### Run 
To run the Docker container, execute the command below. This allocates 8GB of shared memory and utilizes all available GPUs:

``` bash
sudo docker run --shm-size=8g -it --rm detect-malignant /bin/bash gpus=all


```

    
