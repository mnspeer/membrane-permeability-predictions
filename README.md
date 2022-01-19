# membrane-permeability-predictions
Macrocycles are attractive scaffolds for drug design, as they are large enough to disrupt protein-protein interactions with high potencies. However, due to their size and often their polarity, macrocycles usually have poor oral bioavailability. Specifically, they are often (but not always)
cell-membrane impermable and do not reach intracellular drug targets. Models that can acurately predict membrane permeability have great potential to assist the design of potent, membrane permeable macrocyclic compounds as biological tools and new drugs.

Here we present the code for a random forest regression model that aims to predict the membrane permeation rates of large, flexible macrocycles that lie outside of Rule of 5 chemical space. Biophysically meaningful chemical descriptors (features).
used in the RF regression tool, so as to model the thermodynamic entropy and enthalpy of membrane permeation.

The model uses Python to implement a random forest regression model using chemical descriptors calculated using Dragon and Maestro.

## Table of contents
* [Files](#files)
* [Technologies](#technologies)
* [Installation](#installation)
* [Application](#application)
	* [Using a pre-trained model](#using-a-pre-trained-model)
	* [Training the model on a new dataset](#training-the-model-on-a-new-dataset)
* [Additional programs used in research study](#additional-programs-used-in-research-study)

## Files

* Set1_Set2_Set3_SMILE_3D.csv - Input data for compounds in Sets 1-3 calculated using Dragon and Maestro.
* Set1_Set2_Set3_trained_model_3D_nHDon_new.joblib - Saved model calculated using Sets 1-3.
* rf_classifier_load.py - Python program to calculate permeabilities of peptides using random forest models.
* save_trained_model.py - Python program to XXXX

## Technologies
This project was created with:
* Python version: 3.6 (https://www.python.org/)
* Scikit-learn version: 0.22.2 (https://scikit-learn.org/) 
* Python for data analysis (https://pandas.pydata.org/)
* Dragon (https://chm.kode-solutions.net/pf/dragon-7-0/)
* Maestro (https://www.schrodinger.com/products/maestro)

## Installation

Download our code from Gituhub put it in a convenient location.

The code requires Python 3 - you may need the *python3* command to execute this.

Pandas and sklearn are dependencies of this code. These can be installed in several ways - one option is using pip.

~~~
python3 -m pip install pandas
python3 -m pip install scikit-learn
~~~
	


## Application

### Using pre-trained models

Pretrained models are run with *rf_classifier_load.py*

Syntax:

~~~
python rf_classifier_load.py *datafile* *modelfile*
~~~

To use our pre-trained model (Set1_Set2_Set3_trained_model_3D_nHDon_new.joblib, which istrained on Set1 + Set2 + Set3) only the raw data file is required as a command line argument.

Example:

~~~
python rf_classifier_load.py Set1_Set2_Set3_SMILE_3D.csv
~~~

To use your own pre-trained model, provide the saved model file as a second command line argument.

~~~
python rf_classifier_load.py *datafile* *modelfile*
~~~

### Training the model on a new dataset
Instead of using our pre-trained model, a new model can be trained from scratch on any dataset, which contains the ten defined model parameters.

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

To train the program *save_trained_model.py* which creates a new *modelfile* for the *rf_classifier_load.py* program.

The general command is:
~~~
python save_trained_model.py *custom_name* *dataset*
~~~

Examples:
to train the model on the provided *Set1_Set2_Set3_SMILE_3D.csv* dataset, the command is:
~~~
python save_trained_model.py Set1_Set2_Set3_SMILE_3D.csv
~~~

The output modelfile will be called *trained_model.joblib* by default. If you want a custom name for the file instead, define it as follows:

~~~
python save_trained_model.py custom_name Set1_Set2_Set3_SMILE_3D.csv.
~~~
Now the model filename will be *custom_name*.joblib.



## Additional programs used in research study
The following outline of files was used for our model development and analysis.

