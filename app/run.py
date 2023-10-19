import json
import plotly
import plotly.express as px
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('omw-1.4')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import pickle
import sqlalchemy as db
# from sklearn.externals import joblib

app = Flask(__name__)


def message_tokenizer(message):
    '''
    - Parameters : string message
    - Remove any character that's not a digit or a letter
    - Tokenize the passed message
    - remove stop words
    - lemmatize tokens
    - Return : list of clean tokens
    '''
    msg = re.sub(r"[^a-zA-Z0-9]", " ", message)
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(msg)
    tokens = [tk for tk in tokens if tk not in stopwords.words("english")]
    final_tokens = []
    for tk in tokens:
        temp_token = lemmatizer.lemmatize(tk).strip()
        final_tokens.append(temp_token)

    return final_tokens

    
def num_cats (row):
    cats = 0
    for item in row:
        if item == 1:
            cats+= 1
    if cats <= 5:
    	return '0-5 categories'
    elif cats <= 10:
    	return '6-10 categories'
    elif cats <= 15:
    	return '11-15 categories'
    elif cats <= 20:
    	return '16-20 categories'
    else:
    	return '20+ categories'


# load data
engine = db.create_engine('sqlite:///../data/project2.db')
df = pd.read_sql_table('project2_messages', engine)

# load model
model = pickle.load(open('../models/message_classifier.pkl', 'rb'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    cats = df.iloc[:, 4:]
    cats = cats.sum().reset_index(name='sum')
    cats.sort_values(by='sum', ascending=False, inplace=True)
    cats_graph = cats.iloc[0:10, :]
    
    df['cat_count'] = df.apply(lambda x: num_cats(x), axis=1)
    pie_cats = df.cat_count.value_counts().reset_index(name='count')
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
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
                Bar(
                    x=cats_graph['sum'],
                    y=cats_graph['index'].tolist(),
                    orientation='h',
                )
            ],

            'layout': {
                'title': 'Top 10 most common message categories',
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Count"
            }
            }
        }
    ]
    
    fig = px.pie(pie_cats, values='count', names='index', title='Categories per message')
    graphs.append(fig)
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()