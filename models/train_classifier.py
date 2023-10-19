import pandas as pd
import numpy as np
import sqlalchemy as db
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
import pickle
import warnings
warnings.filterwarnings("ignore")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def message_tokenizer(message):
    '''
    - Parameters : string message
    - Tokenize the passed message
    - remove stop words
    - lemmatize tokens
    - Return : list of clean tokens
    '''
    tokens = nltk.tokenize.word_tokenize(message)
    tokens = [tk for tk in tokens if tk not in nltk.corpus.stopwords.words("english")]
    final_tokens = []
    for tk in tokens:
        temp_token = nltk.stem.WordNetLemmatizer().lemmatize(tk).strip()
        final_tokens.append(temp_token)

    return final_tokens


def make_model():
    '''
    This function creates a model that classifies text messages into 36 categories
    First, a pipeline is created with three components :
    1- CountVectorizer
    2- Tf-idf
    3- multioutputclassifier using randomforest
    Return :  pipeline
    '''
    pipeline = Pipeline([
        ('cntvct', CountVectorizer(tokenizer=message_tokenizer)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    return pipeline


def test_model(y_true, y_predicted):
    '''
    - Parameters : true y values and predicted y values :
    - Compare both :
    - Accuracy
    - Recall and precision
    - F1 score
    - Make sure we don't divide by zero when tp + fp = 0 or tp + fn = 0
    - print the results
    '''
    categories = y_true.columns.tolist()
    print('\n-------------------------------------------------------------------')
    for index in range(len(categories)):
        try:
            print('Category ', (index+1), ' : ', categories[index], '\n-------------------------------------------------------------------')
            tn, fp, fn, tp = confusion_matrix(np.array(y_true.iloc[:, index]), y_predicted[:, index], labels=[0, 1]).ravel()
            if (tp + fp) == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)

            if (tp + fn) == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)

            if precision == 0 and recall == 0:
                f1 = 0
            else:
                f1 = 2*(precision * recall) / (precision + recall)

            print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
            print('precision = {}'.format(round(precision, 4)))
            print('recall = {}'.format(round(recall, 4)))

            print('F1 Score = {}'.format(round(f1, 4)))
            print('\n-------------------------------------------------------------------')
        except Exception as e:
            print(e)
            print('F1 Score = ', f1_score(np.array(y_true.iloc[:, index]), y_predicted[:, index]))
            print('No messages in our dataset were found in this category\n')

    print('\nAccuracy = \n', (y_predicted == y_true).mean())


def optimize_model(pipeline):
    '''
    parameters : ML pipeline
    define parameters, pass the pipeline and the parameters to GridSearchCV
    Return : the output of GridSearchCV
    '''
    parameters = {
        'cntvct__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [50, 100, 150],
        'clf__estimator__criterion': ['gini', 'entropy', 'log_loss']
    }

    return GridSearchCV(pipeline, param_grid=parameters)


# Loading clean data from the sqlite database we saved earlier into a pandas dataframe
engine = db.create_engine('sqlite:///../data/project2.db')
df = pd.read_sql_table('project2_messages', engine)
print('Step 1 : Load clean data\n')
print(df.info())

print('\nStep 2 : clean and normalize')
print(' 2.1- Removing capitalization')
df['message'] = df['message'].str.lower()
print('\n 2.2- Punctuation removal')
df.message = df.message.apply(lambda x: re.sub(r"[^a-zA-Z0-9]", " ", x))
print('Taking a look at the text after cleaning\n')
print(df.head(10).to_string())
print('\nStep 3 : Modelling')
print(' 3.1) split the data into train and test')
X_train, X_test, y_train, y_test = train_test_split(df.message, df.iloc[:, 4:], test_size=0.33, random_state=42)
print(' 3.2) call the make_model() function that will return our model')
msg_model = make_model()
print(' 3.3) fit the model to the data')
msg_model.fit(X_train, y_train)
print(' 3.4) test model accuracy')
y_predicted = msg_model.predict(X_test)
test_model(y_test, y_predicted)

print('\nStep 4 : Optimize model using GridSearchCV')
optimized_model = optimize_model(msg_model)
optimized_model.fit(X_train, y_train)
y_predicted2 = optimized_model.predict(X_test)
test_model(y_test, y_predicted2)
print('\nStep 5 : Saving the model in a pkl file')
pickle.dump(optimized_model, open('message_classifier.pkl', 'wb'))
print('... success')