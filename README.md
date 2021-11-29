# membrane-permeability-predictions
BILLY FILL OUT --> quick intro to research.

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* Application (#application}
* [Code structure](#code-structure)

## General info
quick info about model. --> MELISSA FILL (BILLY?)
	
## Technologies
Project is created with:
* Python version: 3.6 (for ipynb)
* Python version: 3.8 (for py)
* modAL version: 0.4.1
* scikit-learn version: 0.22.2
* lightgbm version: 2.2.3

	
## Setup

## Application
When running our pre-trained code (trained on Set1 + Set2 + Set3), use the *rf_classifier_load.py* file.
Input the name of the raw data file used as a command line argument, when running the code.

Example: py rf_classifier_load.py Set1_Set2_Set3_SMILE_3D.csv

Ensure, that the following fields are included in your data file: 
1. title =
2. MLOGP =
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
