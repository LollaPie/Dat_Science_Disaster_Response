# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """ load two datasets from each filepath
    Input: two filepaths
    Output: merged dataframe of both datasets
    """
    # load both dfs
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge dfs
    df = pd.merge(messages, categories, how='inner')
    
    return df


def clean_data(df):
    """ cleans dataframe """
    # split categories column into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    # use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert dataframe to string type
    categories.astype(str)

    # set each value to be the last character of the string and convert it to numeric
    for column in categories:
        categories[column] = categories[column].apply(lambda x: pd.to_numeric(x[-1]))
        
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # drop the category 'child alone' since it has no counts
    categories.drop('child_alone', axis=1, inplace=True)
    
    # replace value 2 with 1 in df
    categories.replace(2, 1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df_con = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df_con.drop_duplicates(inplace=True)
    
    return df_con


def save_data(df, database_filename):
    """ save dataframe to sql database, table name='clean_messages' """
    # use sql alchemy engine to store df to an sqlite db
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('clean_messages', engine, index=False, if_exists='replace')


def main():
    """ main script """
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
