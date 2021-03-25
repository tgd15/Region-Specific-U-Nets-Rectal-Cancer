# U-Nets for Post-Treatment Rectal Cancer Imaging

## Overview

2 deep learning U-Nets were developed to segment the outer rectal wall and the lumen on post-treatment rectal cancer MRI. Segmentations can then be viewed in 3D Slicer. You can clone this repository and run these U-Nets in less than 5 minutes.

## Table of Contents

- [Dependencies and Installation](#installation)
- [Running the Code](#running-the-code)
- [Visualizing the Results](#visualizing-the-Results)
- [Troubleshooting](#troubleshooting)

## Dependencies and Installation <a name="installation"></a>

### Dependencies
The following dependencies are required to run the U-Nets:
```bash
- Python 3.7.7
- Tensorflow 1.15.0 or higher
- Keras 2.2.4 or higher
- SimpleITK 1.2.4 or higher
- opencv 3.4.2
- Pillow 7.1.2 or higher
- h5py 2.8.0 or higher
- matplotlib 3.2.2 or higher
- numpy 1.18.1 or higher
```

Note running the U-Nets on higher versions of each package has **not** been tested. If you are running higher versions of the packages listed above, please submit a pull request to indicate whether the code runs or breaks for each dependency.

### Installation

To clone and run this application, you'll need Git nstalled on your computer. From your command line:
```bash
# Clone this repository (assuming you are using the HTML protocol)
git clone https://github.com/tgd15/Post-Treatment_Unet.git

# Go into the repository
cd Post-Treatment_Unet

```

If you have Conda installed, create a Conda environment from the `environment.yml` file in the repository root and activate it.
- On Windows:
```bash
conda env create --file environment.yml
conda activate postcrt
```
- On MacOS/Linux:
```bash
conda env create --file environment.yml
source activate postcrt
```

If you don't use Conda, ensure your environment has the required dependencies noted in the dependencies.

## Running the Code

Once you have everything installed, run the code by executing `run_Unet.py` from your command line. `run_Unet.py` requires two arguments:

1. **--m**, specify the U-Net you want to run as a string. Allowed choices are `Outer_Rectal_Wall` or `Lumen`.
2. **--i**, specify the filepath to .mha file you want to annotate as a string.

Here are examples demostrating how to run the U-Nets:

Running Outer Rectal Wall U-Net:
```bash
python run_Unet.py --m "Outer_Rectal_Wall" --i "/GoogleDrive/Shared drives/INVent_Data/Rectal/newdata/UH/RectalCA145-2/RectalCA145-2_Post_Ax.mha"
```

Running Lumen U-Net:
```bash
python run_Unet.py --m "Lumen" --i "/GoogleDrive/Shared drives/INVent_Data/Rectal/newdata/UH/RectalCA145-2/RectalCA145-2_Post_Ax.mha"
```

Once the code finishes running, a .mha file with `prediction_label` appended to the end of the filename will be saved in the current directory. This .mha file contains the segmentations generated by the U-Net.

## Visualizing the Results

The .mha file ending in `prediction_label` is designed to be view in 3D slicer. Simply drag the file from your file explorer/finder into 3D slicer and overlay it on the MRI.

In 3D slicer, segmentations generated by outer rectal wall U-Net will be label 8. Segmentations generated by lumen U-net will be label 2.

## Troubleshooting

If you have questions, open an issue on the repository.

If you find a bug, please fix it and submit a pull request.

<h1 align="center">
  <br>
  <a href="https://engineering.case.edu/groups/inventlab/home"><img src="https://engineering.case.edu/groups/inventlab/sites/engineering.case.edu.groups.inventlab/files/invent_lab_logo_site_header.png" alt="INVent Lab Logo" width="400"></a>
</h1>
