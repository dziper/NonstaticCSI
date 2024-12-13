# Nonstatic CSI

Ibrahim Kilinc, Nima Razavi, and Daniel Ziper

{ikilinc, nrazavi, dziper}@ucsd.edu

This repo contains our source code for our ECE 257A research project.

## Setup

We ran our project with python3.9 and Matlab R2024b.

Install Dependencies: matlabengine, tensorflow, matplotlib, statsmodels

Download Data from https://drive.google.com/drive/folders/1-4bHqCZL95eqa0QfCPvBXkezr_9WdWvG?usp=drive_link 
and put into ./data/dataset2

## This Repo

The ./project folder contains files used to run experiments and generate the results in our paper.

Important Files
- simpleTest.ipynb
  - START HERE!
  - Run and compare the LSTM model with the Reference model
- lstm_model.py
  - This contains the LSTM predictor model, as well as other model variations
- reference_impl.py
  - This contains the reimplementation of the Reference code https://github.com/matteonerini/ml-based-csi-feedback
- bulk_test.ipynb
  - Run models with many parameters
- plot_results_v2.ipynb
  - Generate figures from the paper with data from bulk_test.ipynb