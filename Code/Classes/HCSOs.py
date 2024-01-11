import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import os


class HCSOs():
    '''
    A class designed to hold the HCSOs data set and perform dataframe operations that are specific to it.
    '''
    def __init__(self, data_location, version_name = None):
        '''
        Initialise the class using the location of the data directory holding the csv file and the name of HCSOs csv to be loaded
        '''
        if not version_name:
            try:
                version_name = [i for i in os.listdir(data_location).sorted(ascending=False) if i.endswith("_articles.csv")][0]
            except:
                version_name = "parlrules_ukhoc_3.0.1_articles"

    def _trim_cols(self, input_df):
        '''
        We won't be using all the columns, this just gets rid of them
        '''
        data = input_df.copy()
        output = data.drop(['chamberid', 'releaseid', 'articleid'], axis=1)
        return(output)                            
    
    def _version_to_date(self, input_df):
        '''
        This function creates a publication date column which holds the versionid in datetime format
        '''
        data = input_df.copy()
        output = data.copy()
        output['publication_date'] = pd.to_datetime(data['versionid'])
        return output
    
    def _better_article_number(self, input_df):
        '''
        This converts the article number into separate numeric and alphabetical columns to make sorting possible
        '''
        data = input_df.copy()
        # Add 'article_number' and 'article_letter' columns
        data['article_number'] = data['current_num'].apply(lambda x: int(''.join(filter(str.isdigit, x))) if bool(re.search(r'\d', x)) else np.nan)
        data['article_letter'] = data['current_num'].apply(lambda x: ''.join(filter(str.isalpha, x)) if bool(re.search(r'[a-zA-Z]', x)) else np.nan)
        # Arrange by 'publication_date' first and then by 'article_number' and 'article_letter'
        output = data.sort_values(['publication_date', 'article_number', 'article_letter'], ascending = [True, True, True])
        return output

    def _add_text_index(self, input_df):
        '''
        This function adds the text_index column, a column that holds a unique id for each distinct item in the lower cased text column
        '''
        data = input_df.copy() ## Copy the input so it is only changed in here
        # Remove duplicates based on lowercase 'text' column
        data['text_lower'] = data['text'].str.lower()
        original = data.copy() ## Save a checkpoint of the data here
        data.drop_duplicates(subset='text_lower', keep='first', inplace = True)
        x = [i for i in range(0,len(data),1)]
        data["text_index"] = x
        output = original.merge(data[["text_lower","text_index"]], how = "left", left_on = "text_lower", right_on = "text_lower")
        output.drop("text_lower", axis = 1, inplace = True)
        return output
    
    def refine_data(self):
        '''
        This runs all of the above commands
        '''
        df = self.data.copy()
        df = self._trim_cols(df)
        df = self._version_to_date(df)
        df = self._better_article_number(df)
        df = self._add_text_index(df)
        self.data = df

    def show_cols(self):
        '''
        Print out the columns of the data set
        '''
        for i in self.data:
            print(i)
    
    def all_ids(self):
        '''
        Returns just the text_index and root_num columns
        '''
        return self.data[["text_index","root_num"]]
    
    def unique_text_ids(self, full = False):
        '''
        Returns only unique text indexes + their root_nums
        '''
        if full:
            return self.data.drop_duplicates(subset="text_index").reset_index()
        else:
            return self.all_ids().drop_duplicates(subset="text_index").reset_index()
    
    def _remove_junk(self, text, patterns): 
        '''
        Remove unwanted characters/patterns from the text.
        '''
        for pattern in patterns:
            text = re.sub(pattern, " ", text)
        return text
    
    def _remove_stopwords(self, text, stop_words):
        '''
        Remove stop words from the text.
        '''
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        return ' '.join(filtered_words)
    
    def preprocess(self, patterns = ["\(i+\)","\([a-z]\)","\[newline\]"], stop_words = None):
        '''
        Preprocess the text data. Uses the dataframe stored in the class object as the data source.

        patterns refers to any patterns that should be removed from the text, this defaults to the minimal approach I have chosen to take
        
        stop_words is the stop word dictionary to use. defaults to none, this way it will only remove stop words if they are provided.
        '''
        self.processed = self.data.copy()
        tqdm.pandas(desc = "Removing unwanted patterns")
        self.processed['text_preprocessed'] = self.processed['text'].progress_apply(lambda x: self._remove_junk(x, patterns))
        if stop_words:
            tqdm.pandas(desc = "Removing stop words")
            self.processed['text_no_stop'] = self.processed['text_preprocessed'].progress_apply(lambda x: self._remove_stopwords(x, stop_words))

    def add_version_id(self):
        versions = self.data[['versionid']].drop_duplicates()
        # Assign a new column with numeric order based on the date
        versions['version_num'] = versions['versionid'].rank().astype(int)
        self.data = self.data.merge(versions, on='versionid', how='left')