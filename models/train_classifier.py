import sys
import pandas as pd
import re
import nltk
import pickle

import sqlite3

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support

# load MLSMOTE
import MLSMOTE


def load_data(database_filepath):
    conn = sqlite3.connect(database_filepath)
    query = 'SELECT * FROM clean_messages'
    df = pd.read_sql(query, con=conn)
    X = df['message'].iloc[0:1000]
    Y = df.iloc[0:1000, 4:]
    
    return X, Y


def tokenize(text):
    # find urls and replace them with 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, 'urlplaceholder', text)
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model(X_train, y_train):
    # initialize vectorizer and tfidt transformer
    vect = CountVectorizer(tokenizer=tokenize)
    tfidf = TfidfTransformer()

    # vectorize and generate tfidf of trainings data
    X_train_counts = vect.fit_transform(X_train)
    X_train_tfidf_matrix = tfidf.fit_transform(X_train_counts)

    # convert np matrix into dataframe
    X_train_tfidf = pd.DataFrame(data=X_train_tfidf_matrix.A)

    # Getting minority samples of the dataframe
    X_sub, y_sub = MLSMOTE.get_minority_samples(X_train_tfidf, y_train)

    # Generating synthetic samples based on the minority samples
    X_res, y_res = MLSMOTE.MLSMOTE(X_sub, y_sub, 100)

    # concatenate augmented data to train data
    X_con = pd.concat([X_train_tfidf, X_res], ignore_index=True)
    y_con = pd.concat([y_train, y_res], ignore_index=True)

    # # fit model
    # pipeline = Pipeline([
    #     ('vect', CountVectorizer(tokenizer=tokenize)),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf', MultiOutputClassifier(RandomForestClassifier(min_samples_split=3)))
    # ])

    model = Pipeline([
        ('clf', MultiOutputClassifier(RandomForestClassifier(min_samples_split=3)))
    ])
    
    return model, X_con, y_con, vect, tfidf


def evaluate_model(model, X_test, Y_test, vect, tfidf):
    # vectorize and generate tfidf of trainings data
    X_test_counts = vect.transform(X_test)
    # X_test_tfidf_matrix = tfidf.transform(X_test_counts)

    # convert np matrix to dataframe
    # X_test_tfidf = pd.DataFrame(X_test_tfidf_matrix.A)
    X_test_counts_df = pd.DataFrame(X_test_counts.A)
    
    # predict data
    y_pred = model.predict(X_test_counts_df)
    # y_pred = model.predict(X_test)

    # print classification report for each feature   
    print(classification_report(Y_test, y_pred, target_names=Y_test.keys()))
    print("f1-score: ", precision_recall_fscore_support(Y_test, y_pred, average='macro')[2])
    

def save_model(model, model_filepath, vect):
    # saving model to pickle file
    pickle.dump((model, vect), open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building models...')
        model, X_con, Y_con, vect, tfidf = build_model(X_train, Y_train)
        # model = build_model(X_train, Y_train)
        
        print('Training models...')
        model.fit(X_con, Y_con)
        
        print('Evaluating models...')
        evaluate_model(model, X_test, Y_test, vect, tfidf)

        print('Saving models...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath, vect)

        print('Trained models saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the models to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
