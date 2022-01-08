# membrane-permeability-predictions
BILLY FILL OUT --> quick intro to research.

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Application](#application)
* [Code structure](#code-structure)

## General info
quick info about model. --> MELISSA FILL (BILLY?)
	
## Technologies
Project is created with:
* Python version: 3.6
* scikit-learn version: 0.22.2

	
## Setup

## Application
When running our pre-trained code (trained on Set1 + Set2 + Set3), use the *rf_classifier_load.py* program.
Input the name of the raw data file used as a command line argument, when running the code.

py py rf_classifier_load.py *datafile*
Example: py rf_classifier_load.py Set1_Set2_Set3_SMILE_3D.csv

When you are using your own pre-trained model, the saved model file is provided as a second command line argument.
Example: py rf_classifier_load.py *datafile* *modelfile*

Ensure, that the following fields are included in your data file: 
1. title =
2. ALOGP =
3. T(N..O) =
4. T(N..N) =
5. piPC10 =
6. piPC04 =
7. piPC02 =
8. nHDon =
9. PSA_w =
10. Es_w =
11. |dEs| =

If any field has a different name in your input file than defined above, this can be adjusted following the prompts in the code.

## Training model on new dataset
Instead of using our pre-trained model, a new model can be trained from scratch on any dataset, which contains the ten defined parameters (as per Section Application).
To train a new model, the *save_trained_model.py* file is used. The model is trained on the datset provided as a command line argument, then dumped into a file and can be
subsequently used as *modelfile* for the *rf_classifier_load.py* program.

## Additional programs used in research study
The following outline of files was used for our model development and analysis.

