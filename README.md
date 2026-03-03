# Confidence Sets for the Sample Average Approximation of Stochastic Discrete Optimization Problems

## Description
This repository contains the replication package for the paper **"Confidence Sets for the Sample Average Approximation of Stochastic Discrete Optimization Problems."** The paper proposes a method to build strong confidence sets for the solutions of stochastic discrete optimization problems solved through the sample average approximation (SAA) method, combining the concept of Model Confidence Sets (MCS) with shrinkage estimation of large covariance matrices. 

The code provided here allows users to reproduce the Monte Carlo simulation experiments detailed in Section 5 of the paper, including the generation of Tables I–VI and Figures 1–12, testing various shrinkage estimators.

## Repository Structure
- `code/`: Contains the Python scripts used to run the Monte Carlo experiments, construct the Model Confidence Sets, apply the different covariance matrix estimators, and produce the final metrics.
- `data/`: Used for storing the simulated data sets (the data in the paper is fully simulated).
- `results/`: The output directory where containing the final plots and tables.

## Requirements
The replication code is written in **Python**. To run the scripts, you will need the following libraries:
- Python 3.x
- NumPy
- SciPy
- Scikit-Learn (used for the Ledoit and Wolf 2004a linear shrinkage estimator)
- Pandas (for table formatting)
- Matplotlib / Seaborn (for plotting the heatmaps)
