import sys
import nltk
nltk.download(['punkt', 'wordnet'])

# import libraries
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

from sklearn.metrics import confusion_matrix, f1_score, classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

def load_data(database_filepath):
    """
    This function loads from specified database file and returns variables needed
    for Building model
    Input:
        1) file name of database file
    Output:
        1) X = X variables portion of the dataframe
        2) Y = Y variables portion of the dataframe
        3) category_names = names of the categories that will be output to predict
    """
    # load data from database
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df = pd.read_sql('SELECT * FROM msg_categories', engine)

    #filter out y columns from total columns
    y_cols = [col for col in df.columns if col not in ['id','message','original','genre', 'related']]

    X = df['message']
    y = df[y_cols]
    category_names = y.columns.values

    return X, y, category_names


def tokenize(text):
    """
    This function performs needed processing of the inputtted text data and
    returns clean tokens
    Input:
        1) text = raw text data
    Output:
        1) clean_tokens = list containing clean tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
