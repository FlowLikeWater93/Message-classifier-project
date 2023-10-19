import pandas as pd
import numpy as np
import sqlalchemy as db


def find_unique_categories(cats):
    '''
    Parameters : categories string 
    - split text by ";"
    - loop through the list 
    - split each item by "-" 
    - select item at index 0
    - save in a list 
    Return : list of unique categories 
    '''
    unique_catgs = cats.split(';')
    for i in range(len(unique_catgs)):
        unique_catgs[i] = unique_catgs[i].split('-')[0]
    return unique_catgs

def remove_duplicates(df):
    '''
    Parameters : pandas dataframe 
    - use pandas drop_duplicates column 
    - group by id column and find any rows with more than one occurence 
    - since there is no way to tell which instance is the correct one, drop messages from the previous step 
    Return : clean dataframe 
    '''
    clean_df = df.drop_duplicates()
    ids = clean_df.groupby('id').id.count().reset_index(name='count').query('count > 1').id.tolist()
    if len(ids)>0:
        clean_df = clean_df[clean_df['id'].isin(ids)==False]
    return clean_df.copy()

def clean_categories (df):
    '''
    Parameters : categories dataframe 
    - Loop through the dataframe using iterrows()
    - create a numpy zeros array of size 37 (36 categories plus message_id)
    - Split categoires column values by ";"
    - Loop through (nested loop) the list of split categories 
    - If category at index J is set to 1, then set numpy array at index J to 1
    - Add message_id to the end of the numpy array 
    - append each numpy array to the master matrix (normalized_categories)
    - get unique categories by calling "find_unique_categories"
    Return : Dataframe that contains the master matrix 
    '''
    normalized_categories = []
    for index, row in df.iterrows():
        categories_list = row[1].split(';')
        message_categories = np.zeros(37)
        for j in range(len(categories_list)):
            if categories_list[j][-1] == '1':
                message_categories[j] = 1

        message_categories[-1] = row[0]
        normalized_categories.append(message_categories.tolist())

    unique_catgs = find_unique_categories(df.iloc[0,1])
    unique_catgs.append('id')
    return pd.DataFrame(data=normalized_categories, columns=unique_catgs)

# Load the datasets from the two available csv files
print('1- Loading the data')
messages_df = pd.read_csv('messages.csv')
categories_df = pd.read_csv('categories.csv')
print(' success ...')
# Explore both dataframes
print('\n2- Exploring the data')
print(" Let's take a look at the size, columns and datatypes of the messages dataframe\n")
print(messages_df.info())
print('\n Print the first 10 rows :')
print(messages_df.head(10).to_string())
print('\n This dataframe has 4 columns : \n 1- message id : a unique numeric identifier\n 2- message : clean message body\n 3- original : original message mostly in other languages (not english). More than half of the values in this columns are empty (null values)\n 4- genre : category of message')
print("\n Let's display the unique values in column genre")
print(messages_df.genre.value_counts())
print("\n Now, Let's take a look at categories dataframe")
print(categories_df.info())
print('\n Print the first 10 rows : ')
print(categories_df.head(10).to_string())
print('\n We can see that this dataframe has only two columns : \n 1- id : message unique numeric identifier (each id here belongs to a row in the messages dataframe)'
      '\n 2- categories: categories of each message (1 or more)')
print(' Looking at the first 10 rows, we can notice that the all of the values in the categories columns are almost identical. We have multiple Categroies seperated by a ";". each category has either a 0 or 1 next to it, denoting wether the message falls into the category or not (1 = category)')
print("\n Let's try to find out how many unique categories we have")
print(' We have {} unique message categories'.format(len(find_unique_categories(categories_df.iloc[0,1]))))
print('\n3- Cleaning the data : ')
print(' 3.1- Dropping duplicates ')
messages_df = remove_duplicates(messages_df)
categories_df = remove_duplicates(categories_df)
print("\n 3.2- Creating 36 columns (one for each unique category)")
print(' If message x is in category y only, column y will have value set as 1 and the rest set as 0')
clean_categories_df = clean_categories(categories_df)
print(clean_categories_df.info())
print(' \n 3.3- Merging the two dataframes on the id column : ')
final_df = messages_df.merge(clean_categories_df, on='id', how='inner')
print(' saving the final dataframe')
engine = db.create_engine('sqlite:///project2.db')
final_df.to_sql('project2_messages', engine, index=False)
