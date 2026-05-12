# IoT Botnet Detection using Machine Learning

# Project Overview
This project focuses on detecting **IoT Botnet attacks** using machine learning techniques.  
It builds a full **end-to-end ML pipeline** including:

- Data cleaning & preprocessing
- Feature engineering & selection
- Class balancing (SMOTE)
- Exploratory Data Analysis (EDA)
- Multiple ML model training
- Model evaluation & comparison dashboard

The goal is to identify whether network traffic is:
- **Normal (0)**
- **Botnet Attack (1)**

##  Machine Learning Pipeline
The project follows a structured ML pipeline:

### 1. Data Cleaning
- Removed missing values (`dropna`)
- Removed duplicate records
- Dropped irrelevant/leakage columns:
  - `saddr`, `daddr`, `category`, `subcategory`, `pkSeqID`, `stime`, `sport`, `dport`, `seq`

### 2. Train-Test Split
- Split ratio: **80% training / 20% testing**
- Stratified sampling used to preserve class distribution


### 3. Encoding
- Applied **One-Hot Encoding**
- Converted categorical features:
  - `proto`
  - `flgs`
  - `state`


### 4. Data Balancing (SMOTE)
- Used **Synthetic Minority Over-sampling Technique (SMOTE)**
- Solves class imbalance problem
- Generates synthetic samples for minority class (attack traffic)


### 5. Feature Engineering & Selection
- Used **Random Forest Feature Importance**
- Applied:
  - Permutation Importance
  - SHAP (Explainable AI)
- Selected **Top N features (Top 20)** for training


### 6. Exploratory Data Analysis (EDA)
Visual analysis includes:
- Correlation Heatmap
- Filtered Correlation Heatmap
- Cluster Map
- Top feature visualization
- Statistical summary (attack vs normal traffic)


##  Machine Learning Models Used

This project compares multiple ML algorithms:


## 1. Gaussian Naive Bayes
### Why used:
- Fast and efficient for baseline performance
- Works well with probabilistic classification
- Assumes feature independence

###  Output:
- Accuracy
- ROC-AUC Score
- Confusion Matrix
- ROC Curve
- Learning Curve


## 2. Support Vector Machine (SVM)
### Why used:
- Strong performance on high-dimensional data
- Works well with non-linear boundaries (RBF kernel)
- Handles imbalanced datasets using `class_weight="balanced"`

###  Output:
- Accuracy
- ROC-AUC
- Classification Report
- Learning Curve


## 3. Random Forest Classifier
###  Why used:
- Handles non-linear relationships
- Provides feature importance
- Robust to overfitting

###  Output:
- Accuracy
- ROC-AUC
- Confusion Matrix
- Learning Curve


## 4. Logistic Regression
###  Why used:
- Baseline linear classifier
- Interpretable model
- Good for comparison with complex models

###  Output:
- Accuracy
- ROC-AUC
- Confusion Matrix
- Learning Curve


## 5. K-Nearest Neighbors (KNN)
###  Why used:
- Simple instance-based learning
- No training phase complexity
- Works well for pattern similarity detection

###  Output:
- Accuracy
- ROC-AUC
- Confusion Matrix
- Learning Curve


## 6. Multi-Layer Perceptron (MLP Neural Network)
###  Why used:
- Captures complex non-linear patterns
- Neural network-based deep learning model
- Suitable for high-dimensional data

###  Output:
- Accuracy
- ROC-AUC
- Confusion Matrix
- Learning Curve


## Model Evaluation Metrics

All models are evaluated using:

-  Accuracy Score
-  ROC-AUC Score
-  Confusion Matrix
-  Classification Report
-  ROC Curve
-  Learning Curve (Overfitting analysis)


##  Model Comparison Dashboard

A final comparison dashboard is generated:

### Features:
- ROC-AUC vs Accuracy bar chart
- Model performance table
- Best model identification

##  Explainability (XAI)

To improve model interpretability:

###  Permutation Importance
- Measures impact of each feature on model performance

###  SHAP (SHapley Values)
- Explains individual predictions
- Shows feature contribution for attack detection


##  Libraries Used

### Data Processing
- pandas
- numpy
- openpyxl

### Visualization
- matplotlib
- seaborn

### Machine Learning
- scikit-learn
- imbalanced-learn (SMOTE)

### Explainability
- shap


##  Key Techniques Used

- Data preprocessing pipeline
- Feature selection using ML-based importance
- Class imbalance handling (SMOTE)
- Multi-model comparison
- Explainable AI (SHAP + permutation importance)
- Learning curves for overfitting detection


##  Final Outcome

The system successfully:
- Detects IoT botnet attacks
- Compares multiple ML models
- Identifies best-performing model
- Provides explainable AI insights

##  Project Origin & Development
This project is based on **original research work and implementation conducted in April 2021**.

The entire system, including:
- ML pipeline design
- Feature engineering strategy
- Model development and evaluation
- Explainability integration

has been further refined into a structured, production-style machine learning project.

##  Future Improvements
- Deep learning models (CNN/LSTM for network traffic)
- Real-time attack detection system
- Deployment using Flask / FastAPI

##  Author

Developed as part of **IoT Security and Machine Learning research and implementation**