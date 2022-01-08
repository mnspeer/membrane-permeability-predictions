# membrane-permeability-predictions
Macrocycles are attractive scaffolds for drug design, as they are large enough to disrupt protein-protein interactions,
and macrocycles can also be constrained into an active, bound conformation, to confer high potency. However, due to their size and often their 
polarity, macrocycles with high potency often have poor oral bioavailability. Specifically, large, flexible macrocycles are often 
cell-membrane impermable, and cannot reach intracellular drug targets. The key determinants of the membrane permeability of small molecules
are well understood, as described by Lipinski's Rule of 5. Here we present the code of a random forest regression model for determination of 
the membrane permeation rates of large, flexible macrocycles, existing outside of Rule of 5 chemical space. Biophysically meaningful features are
used in the RF regression tool, so as to model the thermodynamic entropy and enthalpy of membrane permeation.

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
Please note that pandas and sklearn are dependencies for this code.

When running our pre-trained code (trained on Set1 + Set2 + Set3), use the *rf_classifier_load.py* program.
Input the name of the raw data file used as a command line argument, when running the code.

py rf_classifier_load.py *datafile*
Example: python rf_classifier_load.py Set1_Set2_Set3_SMILE_3D.csv

When you are using your own pre-trained model, the saved model file is provided as a second command line argument.
Example: python rf_classifier_load.py *datafile* *modelfile*

Ensure, that the following fields are included in your data file:
1. title = molecule index number in dataset, starting at 1
2. ALOGP = octanol-water partitioning co-efficient (from SMILES)
3. T(N..O) = sum of topological distances between N and O (from SMILES)
4. nHDon = number of hyrogen bond doners (from SMILES)
5. T(N..N) = sum of topological distances between N and N (from SMILES)
6. piPC10 = multiple molecular path count of order 10 (from SMILES)
7. piPC04 = multiple molecular path count of order 4 (from SMILES)
8. piPC02 = multiple molecular path count of order 2 (from SMILES)
9. MAXDN = maximum electrotopological negative variation (from SMILES)
10. PSA_w = polar surface area in water (Angtroms^2) (from 3D modelling)
11. Es_w = solvation energy in water (kCal mol^-1) (from 3D modelling)

If any field has a different name in your input file than defined above, this can be adjusted following the prompts in the code.

## Training model on new dataset
Instead of using our pre-trained model, a new model can be trained from scratch on any dataset, which contains the ten defined parameters (as per Section Application).
To train a new model, the *save_trained_model.py* file is used. The model is trained on the datset provided as a command line argument, then dumped into a file and can be subsequently used as *modelfile* for the *rf_classifier_load.py* program.
Example: to train the model on the provided *Set1_Set2_Set3_SMILE_3D.csv* file, the command is: python save_trained_model.py Set1_Set2_Set3_SMILE_3D.csv
The file will then be called by default *trained_model.joblib*. If you want a custom name for the file instead, define it via the command line, as follows: python python save_trained_model.py *custom_name* Set1_Set2_Set3_SMILE_3D.csv.
Now the filename will be *custom_name*.joblib instead of *trained_model*.joblib

## Additional programs used in research study
The following outline of files was used for our model development and analysis.

