# membrane-permeability-predictions
Macrocycles are attractive scaffolds for drug design, as they are large enough to disrupt protein-protein interactions with high potencies. However, due to their size and often their polarity, macrocycles usually have poor oral bioavailability. Specifically, they are often (but not always)
cell-membrane impermable and do not reach intracellular drug targets. Models that can acurately predict membrane permeability have great potential to assist the design of potent, membrane permeable macrocyclic compounds as biological tools and new drugs.

Here we present the code for a random forest regression model that aims to predict the membrane permeation rates of large, flexible macrocycles that lie outside of Rule of 5 chemical space. Biophysically meaningful chemical properties (features)
used in the RF regression tool, so as to model the thermodynamic entropy and enthalpy of membrane permeation.

The model uses Python to implement a random forest regression model using chemical features which can be calculated using XXXX.

## Table of contents
* [Technologies](#technologies)
* [Installation](#installation)
* [Application](#application)
	* [Using a pre-trained model](#using-a-pre-trained-model)
	* [Training the model on a new dataset](#training-the-model-on-a-new-dataset)
* [Additional programs used in research study](#additional-programs-used-in-research-study)


## Technologies
Project is created with:
* Python version: 3.6
* scikit-learn version: 0.22.2 (https://scikit-learn.org/) 
* Python for data analysis (https://pandas.pydata.org/)

## Installation
	
XXX Installation  -how to download and install XXX

## Application
Please note that pandas and sklearn are dependencies for this code.

### Using the pre-trained model

To run the code using our pre-trained code (trained on Set1 + Set2 + Set3), use the *rf_classifier_load.py* program.
Include the raw data file as a command line argument.

~~~
python rf_classifier_load.py *datafile*
~~~

Example:

~~~
python rf_classifier_load.py Set1_Set2_Set3_SMILE_3D.csv
~~~

To use your own pre-trained model, provide the saved model file as a second command line argument.

Example:

~~~
python rf_classifier_load.py *datafile* *modelfile*
~~~

Ensure that the following fields are present in your data file:
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

### Training the model on a new dataset
Instead of using our pre-trained model, a new model can be trained from scratch on any dataset, which contains the ten defined parameters (as per Section Application).
To train a new model, the *save_trained_model.py* file is used. The model is trained on the datset provided as a command line argument, then dumped into a file and can be subsequently used as *modelfile* for the *rf_classifier_load.py* program.

Example:
to train the model on the provided *Set1_Set2_Set3_SMILE_3D.csv* dataset, the command is:
~~~
python save_trained_model.py Set1_Set2_Set3_SMILE_3D.csv
~~~

The file will then by default be called *trained_model.joblib*. If you want a custom name for the file instead, define it via the command line, as follows:
~~~
python save_trained_model.py *custom_name* Set1_Set2_Set3_SMILE_3D.csv.
~~~
Now the filename will be *custom_name*.joblib instead of *trained_model*.joblib

The general command is:
~~~
python save_trained_model.py *custom_name* *dataset*
~~~

## Additional programs used in research study
The following outline of files was used for our model development and analysis.

