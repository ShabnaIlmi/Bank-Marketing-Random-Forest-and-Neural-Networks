# Bank-Marketing-Random-Forest-and-Neural-Networks
Machine Learning Coursework
Here's a simplified and structured README template for your GitHub repository:

---

# Machine Learning Models for Bank Marketing Prediction

This project compares two machine learning models, **Neural Networks** and **Random Forest Classifier**, to predict whether a client will subscribe to a term deposit based on the **Bank Marketing Dataset** from the UCI Machine Learning Repository.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models](#models)
  - [Neural Network](#neural-network)
  - [Random Forest](#random-forest)
- [Evaluation](#evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project aims to predict whether a client will subscribe to a term deposit using different machine learning models. The models are trained and evaluated on the **Bank Marketing dataset** and compared in terms of accuracy, precision, recall, F1-score, and ROC-AUC.

## Dataset

The **Bank Marketing dataset** includes information about clients' attributes, banking details, and the results of previous marketing campaigns. The dataset contains the following features:
- Client information (age, job, marital status, etc.)
- Previous campaign outcomes
- Client's response to the current campaign (target variable)

Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

## Preprocessing

Before training the models, the dataset underwent the following preprocessing steps:
- **Handling Missing Values**: Missing data was imputed or removed.
- **Class Imbalance**: Addressed using class weights and oversampling techniques.
- **Encoding Categorical Variables**: Categorical variables were encoded using label encoding or one-hot encoding.
- **Feature Standardization**: Numerical features were standardized to ensure the models perform optimally.

## Models

### Neural Network
- A **feedforward neural network** is implemented using scikit-learn's `MLPClassifier`.
- Hyperparameter tuning is done to find the optimal parameters for better performance.

### Random Forest
- A **Random Forest Classifier** is built using scikit-learn’s ensemble method.
- Hyperparameter tuning is used to optimize the model.

## Evaluation

Models are evaluated using the following metrics:
- **Accuracy**: Measures the proportion of correctly classified instances.
- **Precision**: Evaluates the proportion of true positives among all positive predictions.
- **Recall**: Measures the proportion of true positives among actual positives.
- **F1-Score**: The harmonic mean of precision and recall.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.

Cross-validation and overfitting validation were also used to ensure robustness.

## Results

The results highlight the strengths and limitations of each model in terms of predictive accuracy, computational efficiency, and data handling.

## Installation

To run the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the necessary libraries installed:
   - `scikit-learn`
   - `tensorflow`
   - `pandas`
   - `numpy`
   - `matplotlib`

## Usage

To run the models and train them, execute the following in your terminal:

```bash
python train_model.py
```

This will train both the Neural Network and Random Forest models and output the evaluation results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- UCI Machine Learning Repository for the Bank Marketing dataset.
- scikit-learn and TensorFlow for providing the libraries used in this project.
