Twitter Sentiment Analysis Using SVM
Project Overview

This project implements a Twitter sentiment analysis model to classify tweets as positive or negative. The model uses Support Vector Classification (SVC) and TF-IDF vectorization for feature extraction. It includes data preprocessing, model training, evaluation, and deployment using pickle for model serialization.

Features

Data Preprocessing:

Cleans text data by removing non-alphabetic characters.

Converts text to lowercase.

Removes stopwords using NLTK.

Applies stemming using PorterStemmer.

Feature Extraction:

Uses TF-IDF Vectorizer to convert text data into numerical features suitable for machine learning.

Modeling:

Implements Support Vector Classification (SVC) for sentiment prediction.

Evaluates model performance using accuracy score, classification report, and confusion matrix.

Deployment:

Saves the trained model using pickle for future predictions.

Provides functionality to load the saved model and make predictions on new text data.

Installation

Clone the repository:
git clone <repository-url>
Install required dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn nltk
Download NLTK stopwords (if not already installed):
import nltk
nltk.download('stopwords')

Evaluation

Accuracy Score: Measures overall correctness of the model.

Classification Report: Provides precision, recall, and F1-score.

Confusion Matrix: Visualizes true positives, true negatives, false positives, and false negatives.

Future Enhancements

Integrate deep learning models like LSTM or BERT for improved accuracy.

Add real-time Twitter API integration for live sentiment analysis.

Implement GUI/web app for user-friendly sentiment predictions.

Authors

Abdullah Bin Jalal– Developer & Machine Learning Enthusiast

License

This project is licensed under the MIT License.
