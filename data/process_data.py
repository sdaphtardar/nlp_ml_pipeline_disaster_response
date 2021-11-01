import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function takes input of file paths of 2 csv files, containing messages and categories.
    Then it reads the 2 files in 2 separate dataframes and merges them on a common column before returning it.

    Inputs:
        1) file path of messages csv file
        2) file path of categories csv file
    Output:
        1) merged dataframe containing both messages and categories

    """
    #Load messages data
    messages = pd.read_csv(messages_filepath)
    #Load categories data
    categories = pd.read_csv(categories_filepath)

    # merge datasets using collon column; id
    df = categories.merge(messages, on='id')
    return df


def clean_data(df):
    """
    This function takes dataframe as input, cleans it and returns it as an output.

    Input:
        1) dataframe with messy data
    Output:
        1) dataframe containing cleaned data
    """

    #Now we need to split data in categories column in df dataframe, as
    #performing data analysis on these values will be much easisr if they are
    #in separate columns

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    #Next step is to convert this dataframe (categories) in a form where column
    #names are unique category names and row data signifies presence or absence
    #of that category.

    # select the first row of the categories dataframe.
    row = categories[:1]
    # use this row to extract a list of new column names for categories.
    category_colnames = (row.apply(lambda x: x.str.slice(0, -2))).values[0].tolist()
    # rename the columns of `categories` dataframe.
    categories.columns = category_colnames

    #Data currently contains category names and digit 0 or 1. we do not need
    #category name in the row as that is redundant.

    #Therefore next lines will remove category names from rows except 1 or 0.
    categories = categories.apply(lambda x: x.str.slice(-1))
    #Convert datatype of columns to int
    categories = categories.astype(int)

    #Next step is to merge this categories dataframe with original dataframe (df)

    #First drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    #Now concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, left_index=True, right_index=True)

    #Let's remove duplicate rows from the dataframe as they serve no purpose.
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df




def save_data(df, database_filename):
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
