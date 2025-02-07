# Restaurant Review Sentiment Analysis

## Overview

This repository contains the Restaurant Review NLP.ipynb Jupyter Notebook, which performs Natural Language Processing (NLP) techniques to analyze and classify restaurant reviews as positive or negative using multiple machine learning models.

## Contents

**Restaurant Review NLP.ipynb:** Jupyter Notebook containing the NLP pipeline for sentiment analysis.

**README.md:** Documentation file providing an overview of the project.

## Requirements

To run this notebook, install the following dependencies:

pip install numpy, pandas, matplotlib, seaborn, scikit-learn, nltk, catboost, lightgbm, xgboost

## Dataset

The dataset consists of restaurant reviews labeled with their food genre. The text data undergoes multiple preprocessing steps before being fed into classification models.

## Key Features

1. Data Preprocessing

- **Text Cleaning:** Removes special characters, punctuation, and converts text to lowercase.

- **Tokenization:** Splits reviews into words using NLTK's word_tokenize.

- **Stopword Removal:** Filters out common English words that do not contribute to sentiment.

- **Stemming:** Reduces words to their root form using NLTK’s PorterStemmer.

2. Feature Engineering

- **TF-IDF Vectorization:** Converts text into numerical representations using TfidfVectorizer.

- **Bag of Words (BoW) Model:** Uses CountVectorizer for word frequency-based features.

- **Truncated SVD (LSA):** Reduces dimensionality while preserving important features.

3. Machine Learning Models

- **Logistic Regression:** A simple yet effective baseline classifier.

- **Random Forest Classifier:** An ensemble learning method for improving accuracy.

- **Support Vector Machines (SVM):** Optimized for text classification.

- **XGBoost:** Gradient boosting framework for performance improvement.

- **CatBoost:** Optimized for categorical data, handles text features well.

- **LightGBM:** Efficient and fast gradient boosting model.

- **Voting Classifier:** Combines multiple models to enhance predictions.

4. Model Pipeline

- The notebook utilizes Scikit-learn Pipelines to structure preprocessing and model training efficiently. The pipeline includes:

  - ColumnTransformer for numerical, categorical, and text preprocessing.

  - Feature scaling using StandardScaler.

  - Text feature extraction using TF-IDF and Truncated SVD.

  - Multiple classifiers combined in a Voting Classifier for optimal results.

5. Model Evaluation

- **Confusion Matrix:** Visualizes the model’s performance in classification tasks.

- **Accuracy Score:** Measures how many correct predictions were made.

- Precision, Recall, and F1-Score: Assess classification performance with imbalanced data.


## Results Summary

- Text preprocessing significantly improves model performance by reducing noise.

- TF-IDF and Truncated SVD improve feature representation compared to raw text input.

- Ensemble models (Voting Classifier) achieve the highest accuracy, leveraging the strengths of multiple classifiers.

- CatBoost and LightGBM outperform traditional models in handling text-based data.

- Final validation F1-Score: ~0.79, demonstrating strong classification performance.

