# Capstone project: Disaster Response Classification App
## Table of Content
1. [Motivation](#motivation)
2. [Installation](#installation)
3. [About the Data](#about-the-data)
4. [About the Methods](#about-the-methods)
5. [Acknowledgements & Licensing](#acknowledgements--licensing)

## Motivation <a name="motivation"/>
Environmental disasters increases due to climate change. Emergency services have a higher impact on disaster responses which they have to classify into categories 
to better process them. Especially during a disaster they have the lowest capacities to process the messages.

As a capstone project for the udacity data science course I build a machine learning model based on python to classifiy these text messages.

## Installation <a name="installation"/>
Python version 3.6 or higher is recommended.

- Clone these repository
- cd into the app directory and run 'python run.py'
- install required packages by run 'pip install -r requirements.txt'
- open http://0.0.0.0:3001/ on your browser to view the web app

## About the Data <a name="about-the-data"/>
There are two datasets used to train and test the model:
- disaster_categories.csv: contains multilabel categories for each message
- disaster_messages.csv: contains the actual text messages, both in the original language as well as in English

The data is highly imbalanced. One category does not even have a single count.

## About the Methods <a name="about-the-methods"/>
The python script 'process_data.py' is used to clean the data.
Categories are split into single columns. Column names for each category are given. Duplicates are removed as well as one category which has no counts. 
Both datasets are concatenated to one dataset. And finally it is saved to a sql database.

The python script 'train_classifier.py' generates the machine learning model.
The english text messages are tokenized, lemmatized and stop words are being removed. The model is build by using the CountVectorizer combined with the custom 
tokenizer and a tfidf transformer. MLSMOTE (Multilabel Synthetic Minority Over-sampling TEchnique) is used to oversample the train data. The model is evaluated 
using the f1-score.

The oversampling is only applied to the train data. Therefore the vectorizer and the model is saved as a pickle file. Both are used to apply the model to new text 
messages in the web application.

## Acknowledgements & Licensing <a name="acknowledgements--licensing"/>
Credits to Figure Eight Inc. to provide the data and Udacity to provide the course and the support.
