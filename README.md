# Batch AI Optimiser – ML Model

Machine Learning module for **AI-driven batch process optimisation**.
This repository contains the **training pipelines, predictive models, and optimisation logic** used to analyze batch manufacturing data and recommend optimal process configurations.

The ML models predict key performance indicators (KPIs) such as **product quality, energy consumption, and process efficiency**, enabling data-driven decision making for industrial batch processes.

---

## Project Overview

Industrial batch processes often involve multiple parameters such as temperature, pressure, mixing speed, and material ratios.

Changing one parameter may improve one objective but worsen another.
This ML system helps solve that problem by:

* Predicting process outcomes using **trained ML models**
* Identifying relationships between input parameters and output metrics
* Supporting **multi-objective optimisation algorithms** for optimal parameter combinations

This repository focuses on the **machine learning backend**, which can be integrated with frontend applications or APIs.

---

## Repository Purpose

This project is designed to:

* Train ML models using batch process datasets
* Predict key output variables
* Provide input to optimisation algorithms (e.g., NSGA-II)
* Support integration with web applications or APIs

It acts as the **core intelligence layer** of the Batch AI Optimisation system.

---

## Machine Learning Workflow

```
Dataset
   │
   ▼
Data Preprocessing
   │
   ▼
Feature Engineering
   │
   ▼
Model Training
(XGBoost / Regression Models)
   │
   ▼
Model Evaluation
   │
   ▼
Prediction Outputs
   │
   ▼
Optimisation Algorithms (NSGA-II)
```

---

## Key Features

* Batch process data preprocessing
* Machine learning model training
* Prediction of process KPIs
* Support for multi-objective optimisation
* Integration-ready ML models
* Modular code for experimentation

---

## Machine Learning Models

### Regression Models

Used to predict important process outputs such as:

* Hardness
* Dissolution rate
* Power consumption
* Carbon emissions
* Process efficiency

Each target variable can be predicted using **separate regression models** trained on process parameters.

---

### XGBoost

XGBoost is used for high-performance regression tasks because of its:

* High predictive accuracy
* Ability to handle nonlinear relationships
* Efficient training on structured datasets

---

## Optimisation Integration

The predictions from ML models can be used by **multi-objective optimisation algorithms** such as:

* NSGA-II

This allows simultaneous optimisation of multiple objectives like:

* Maximize product quality
* Minimize energy consumption
* Minimize carbon emissions

---

## Tech Stack

**Language**

* Python

**Libraries**

* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib / Seaborn

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Arka-Sengupta/Batch-AI-Optimiser-ML-model.git
cd Batch-AI-Optimiser-ML-model
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Models

Example workflow:

1. Load dataset
2. Preprocess features
3. Train regression models
4. Evaluate model performance
5. Generate predictions

## For running the model API
``
uvicorn main:app --reload --port 8000
``
---

## Example Use Case

A manufacturing engineer wants to improve the performance of a batch production process.

Using this system:

1. Historical batch data is used to train ML models.
2. The model predicts process outcomes for new parameter combinations.
3. Optimisation algorithms identify the best parameter configuration.
4. Engineers select the best trade-off solution.

---

## Future Improvements

* Hyperparameter tuning automation
* AutoML pipeline
* Model explainability (SHAP / LIME)
* Integration with real-time manufacturing data
* Deployment with FastAPI or Flask
* Containerisation with Docker

---

## Related Repository

Frontend and system integration repository:

```
https://github.com/Arka-Sengupta/Batch-AI-optimisation
```

This repository provides the **ML model backend** used by the optimisation platform.

---

## Contributors

Project developed as part of an **AI-based batch optimisation system**.

Contributions, improvements, and suggestions are welcome.
