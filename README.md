## Supply Chain Late Delivery Risk Prediction Model

## Project Overview:
This project builds a production-ready Machine Learning system to predict Late Delivery Risk in supply chain operations.
The solution integrates:
Automated model selection using FLAML
Imbalance handling with SMOTE
Reproducible pipelines using DVC (DAG-based staging)
Dependency management with uv
Deployment using Streamlit

The system enables proactive risk detection and improves supply chain decision-making.

## Problem Statement:
Late deliveries impact:
1.Customer satisfaction
2.Logistics costs
3.Inventory planning
4.Operational efficiency

The objective is to classify whether an order has a Late Delivery Risk (Late_delivery_risk).

## Project Structure:

SupplyChain_PredictionModel/
│
├── data/
│   └── raw/
│
├── dvc.yaml
├── params.yaml
├── best_model.pkl
├── preprocessor.pkl

├── automl.ipynb
├── flaml.log
├── streamlit_app.py (Streamlit)
└── main.py

