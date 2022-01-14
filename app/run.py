import json
import plotly
import pandas as pd
import pickle

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
# import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """ tokenizer for text input on web app
    Input: text message
    Output: list of tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('clean_messages', engine)

# load model
# model = joblib.load("../models/classifier.pkl")
with open('../models/classifier.pkl', 'rb') as f:
    model, vect = pickle.load(f)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """ renders master.html and creates graphs for webpage """
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    categories = df.iloc[:, 4:].sum(axis=0)
    categorie_names = df.iloc[:, 4:].columns
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=categorie_names,
                    y=categories
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Pie(
                    values=genre_counts,
                    labels=genre_names
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'textposition': 'none',
                'showlegend': 'true',
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    """ web page that handles user query and displays model results """
    # save user input in query
    query = request.args.get('query', '') 

    X_test = vect.transform([query])

    # use model to predict classification for query
    classification_labels = model.predict(X_test)[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()