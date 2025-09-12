# ❤️ Heart Disease Prediction Project

![Python](https://img.shields.io/badge/python-3.11-blue.svg)

This project implements an **LGBMClassifier** model to classify heart disease stages from the [UCI Heart Disease Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data).  
It provides:
- Two Jupyter notebooks for **exploration** and **model training**  
- A simple **CLI tool** for inference and evaluation  

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Findings](#-findings)
- [Environment Setup](#️-environment-setup)
- [Dependencies](#-dependencies)
- [Usage](#️-usage)
- [Model Interpretation](#️-model-interpretation)

---

## 🔎 Overview
The goal of this project is to predict the **stage of heart disease** based on patient attributes and to interpret model results.  

To provide an interactive way of using the model, a **CLI application** was built with preprocessing included. The output is the **predicted probability** of having heart disease.  

The dataset is **multivariate**, containing 14 attributes:  
- Age, Sex, Chest Pain Type, Resting Blood Pressure, Serum Cholesterol  
- Fasting Blood Sugar, Resting ECG Results, Max Heart Rate Achieved  
- Exercise-Induced Angina, ST Depression (`oldpeak`), Slope of Peak ST Segment  
- Number of Major Vessels, Thalassemia  

The chosen model is **LightGBM’s LGBMClassifier**, which is efficient for tabular data and handles categorical variables and class imbalance well.  

---

## 📊 Findings
**Results on the test set (20% split):**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Healthy) | 0.82 | 0.80 | 0.81 | 81 |
| 1 | 0.54 | 0.49 | 0.51 | 43 |
| 2 | 0.29 | 0.26 | 0.27 | 27 |
| 3 | 0.28 | 0.32 | 0.30 | 28 |
| 4 (Final stage) | 0.00 | 0.00 | 0.00 | 5 |
| **Weighted Avg** | **0.57** | **0.55** | **0.56** | **184** |

📌 Observations:
- Best performance on **Class 0 (Healthy)**  
- Weakest on **Class 4 (Final Stage)**, likely due to severe **class imbalance**  

**Confusion Matrix Summary:**

| Prediction / Actual | Healthy | Ill |
|---------------------|---------|-----|
| Predicted Healthy   | 65 ✅ (True Negative) | 14 ❌ (False Negative) |
| Predicted Ill       | 16 ❌ (False Positive) | 89 ✅ (True Positive) |

From these results (binary healthy vs ill classification):  
- **Precision**: 84.8%  
- **Recall**: 86.4%  
- **F1-score**: 85.6%  

This shows that the model is reasonably good at detecting ill patients (high recall), while keeping false positives relatively low (high precision).

---

## ⚙️ Environment Setup
This project uses **Conda** for environment management.  

1. Create environment:  
   ```bash
   conda create -n myenv python=3.11
   conda activate myenv
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
💡 If you don’t yet have requirements.txt, generate it from your environment:
pip freeze > requirements.txt

---

## 📦 Dependencies
Main libraries used:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- shap
- lightgbm

---

## ▶️ Usage
Run the CLI with:

   ```bash
   python main.py
   ```

The application will:
1.	Prompt you for patient input data
2.	Preprocess the data
3.	Run inference with the trained model
4.	Show the predicted probability of heart disease

---

## 🧩 Model Interpretation

To better understand which features influence the model’s predictions, SHAP (SHapley Additive exPlanations) was used.  

Below are the feature importance plots for the **Healthy** class (class 0) and the **Ill** class (all other stages grouped). These bar plots highlight the top contributing features for each class.

### 🔹 Feature Importance for Healthy Class
![SHAP Healthy](images/plot_healthy.png)

### 🔹 Feature Importance for Ill Class
![SHAP Ill](images/plot_ill.png)

