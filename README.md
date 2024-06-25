# n2ofluxml

# Predicting Nitrous Oxide Flux in Agricultural Systems using Machine Learning

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
Nitrous oxide (N2O) is a potent greenhouse gas significantly contributing to climate change and stratospheric ozone depletion. This project aims to enhance the prediction of N2O emissions from agricultural systems using advanced machine learning models. Our approach involves leveraging long-term high-frequency data from automated flux chambers to build models that can accurately predict daily N2O flux.

## Data
The dataset for this study was obtained from three long-term experiments:
- W.K. Kellogg Biological Station in Michigan (BCSE\_KBS and MCSE-T2)
- Arlington Agricultural Research Station in Wisconsin (Arlington WI)
You can access the dataset [here](https://datadryad.org/stash/dataset/doi:10.5061/dryad.bnzs7h493).
The data includes daily N2O flux measurements along with various predictive variables such as soil moisture, air temperature, cumulative precipitation, and nitrogen fertilization rate.

## Methodology
We employed three machine learning models for predicting N2O flux:
- **Random Forest**
- **XGBoost**
- **Long Short-Term Memory (LSTM) Neural Networks**

The data preprocessing involved handling missing values using iterative imputation, scaling features using a robust scaler, and performing quantile binning on the target variable. The models were evaluated using metrics such as R², RMSE, and MAE.

## Results
The Random Forest model performed the best among the three models, explaining 95.26% of the variability in Arlington WI, 99.38% in BCSE\_KBS, and 96.07% in MCSE-T2. The models were evaluated based on their performance on different experimental setups and vegetation types, demonstrating significant improvements over traditional empirical and biophysical models.

## Installation
To run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/FlyingPotato437/n2ofluxml
   cd n2ofluxml
