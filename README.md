# NLP-Text-Classification-Model
This model create for detect positive and negative sentence 

#Project Description :
This project implements a Natural Language Processing (NLP) text classification model using Python and scikit-learn. The system preprocesses raw text data, converts it into numerical feature vectors, and trains a machine learning model to classify text into categories (such as positive/negative sentiment).
The notebook demonstrates the full NLP pipeline including text cleaning, stopword removal, stemming, feature extraction using Bag-of-Words, model training, prediction, and performance evaluation with accuracy and visualization metrics.

#Technologies Used :
Python
Jupyter Notebook
NLTK (Natural Language Toolkit)
scikit-learn
NumPy
Pandas
Matplotlib

#NLP Techniques Used :
Text Cleaning (Regex)
Lowercasing
Stopword Removal
Stemming (Porter Stemmer)
Tokenization
Bag of Words Model
CountVectorizer

#Machine Learning Model Used:
Based on your notebook structure:
Multinomial Naive Bayes Classifier
Explanation text:
The model uses a Multinomial Naive Bayes classifier, which is well-suited for text classification problems using word frequency features.
It is efficient, fast to train, and performs well on sparse Bag-of-Words representations.

#Workflow / Pipeline:

1. Load dataset
2. Clean text using regex
3. Remove stopwords (keeping important negations like "not")
4. Apply stemming
5. Build cleaned corpus
6. Convert text to numerical features using CountVectorizer
7. Split dataset into train/test sets
8. Train Naive Bayes classifier
9. Predict on test data
10. Evaluate using accuracy, confusion matrix, and ROC curve


#Evaluation Metrics Included:

Accuracy Score
Confusion Matrix
ROC Curve
Precision / Recall / F1 Score
Visualization Charts


#How to Run:
pip install scikit-learn nltk pandas numpy matplotlib
python -m nltk.downloader stopwords


#Skills Demonstrated (Good for Recruiters):
NLP preprocessing
Feature engineering for text
ML model training
Model evaluation
Data visualization
Python ML stack usage

