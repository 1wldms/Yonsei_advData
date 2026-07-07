# Cardiovascular Disease Prediction by Household Type

Predicting cardiovascular disease risk using machine learning models tailored to single-person and multi-person households based on the Korea National Health and Nutrition Examination Survey (KNHANES).

**Live Demo**

[https://cvd-ls7-project.streamlit.app/](https://cvd-ls7-project.streamlit.app/)

---

## Overview

Cardiovascular disease (CVD) remains one of the leading causes of death worldwide. Most existing prediction models focus on individual clinical factors without considering household composition.

This project proposes two independent machine learning models for:

* Single-person households
* Multi-person households

instead of treating household type as just another feature.

The models were trained using KNHANES 2016–2021 data and deployed as an interactive Streamlit web application.

---

## Features

* Separate prediction models for single- and multi-person households
* Temporal validation using 2016–2020 training data and 2021 testing data
* Explainable AI using SHAP
* Interactive Streamlit interface
* Automatic model selection based on household type

---

## Demo

The deployed application allows users to:

* Select household type
* Enter demographic and health information
* Predict cardiovascular disease risk
* View prediction results instantly

---

## Dataset

**Source**

Korea National Health and Nutrition Examination Survey (KNHANES)

Years

* 2016
* 2017
* 2018
* 2019
* 2020
* 2021

Study Population

* Adults aged 50 years or older

Final Dataset

* 20,735 participants
* Single-person households: 3,500
* Multi-person households: 17,235

---

## Methodology

### Data Preprocessing

* Special code handling
* Missing value imputation (MICE & Mode Imputation)
* Outlier removal
* One-hot encoding
* RobustScaler
* Temporal train-test split

---

### Model Development

#### Single-person Household

Model

* Logistic Regression with Focal Loss (PyTorch)

Reason

* Better performance on highly imbalanced data
* Improved Recall for minority class detection

#### Multi-person Household

Model

* Logistic Regression with `class_weight="balanced"`

Reason

* Stable performance with larger sample size
* Better generalization

---

## Explainable AI

SHAP (SHapley Additive Explanations) was applied to interpret feature importance and compare risk factors between household types.

Both global and local explanations were analyzed.

---

## Deployment

The prediction models were deployed using Streamlit Community Cloud.

Application workflow

1. Select household type
2. Input health information
3. Load the corresponding prediction model
4. Predict cardiovascular disease risk
5. Display results

---

## Project Structure

```text
.
├── app.py
├── cvd_model_single.pkl
├── cvd_model_multi.pkl
├── requirements.txt
├── README.md
└── .devcontainer/
```

---

## Tech Stack

| Category      | Technology                |
| ------------- | ------------------------- |
| Language      | Python                    |
| ML            | Scikit-learn, PyTorch     |
| Data          | Pandas, NumPy             |
| Visualization | SHAP, Matplotlib          |
| Web           | Streamlit                 |
| Deployment    | Streamlit Community Cloud |

---

## Results

| Household     | Model                                              |
| ------------- | -------------------------------------------------- |
| Single-person | Logistic Regression + Focal Loss                   |
| Multi-person  | Logistic Regression + Balanced Logistic Regression |

The proposed approach achieved competitive predictive performance while maintaining interpretability through SHAP analysis.

---

## Team

* 연은서
* 민지은
* 박하람

---

## Related Resources

GitHub Repository

[https://github.com/1wldms/Yonsei_advData](https://github.com/1wldms/Yonsei_advData)

Live Demo

[https://cvd-ls7-project.streamlit.app/](https://cvd-ls7-project.streamlit.app/)

Single-person Household Training Code

[https://colab.research.google.com/drive/1Ai86WwaKPpBUeZYJFl4PDN6d-Gy1sph4](https://colab.research.google.com/drive/1Ai86WwaKPpBUeZYJFl4PDN6d-Gy1sph4)

Multi-person Household Training Code

[https://colab.research.google.com/drive/1U1GjjO8BGh1nMTd6JxREOfXK-Ke-rgF0](https://colab.research.google.com/drive/1U1GjjO8BGh1nMTd6JxREOfXK-Ke-rgF0)
