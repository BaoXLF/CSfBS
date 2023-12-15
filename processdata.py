import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class processData:
    
    def __init__(self, data, patterns, brands, threshold) :
        
        self.data = data
        self.brands = brands
        self.patterns = patterns
        self.threshold = threshold
        
    def preProcessData(self):
        
        # set all words in python into lower case
        self.data['title'] = self.data['title'].str.lower()
        
        # change the non comparable terms
        self.data['title'] = self.data.title.replace(self.patterns, regex=True)
        
        # expand the feature map and drop columns with too many NaN
        data_no_url = self.data.drop(['url'], axis=1) 
        values = data_no_url.join(pd.DataFrame(data_no_url.pop('featuresMap').values.tolist()))
        values['Screen Refresh Rate'] = values['Screen Refresh Rate'].combine_first(values['Refresh Rate'])
        # print(values['Screen Refresh Rate'])
        values['Screen Refresh Rate'] = values['Screen Refresh Rate'].replace(r"\D+", "", regex=True)
        values['Screen Refresh Rate'] = values['Screen Refresh Rate'].fillna('temp')
        values['Screen Refresh Rate'] = values['Screen Refresh Rate'].astype(str) + 'hz'
        values['Screen Refresh Rate'] = values['Screen Refresh Rate'].replace('temphz', np.nan).replace('temp', np.nan)
     
        # get Screen Size
        values['Screen Size Class'] = values['Screen Size Class'].combine_first(values['Screen Size'])
        values['Screen Size Class'] = values['Screen Size Class'].combine_first(values['Screen Size (Measured Diagonally)'])
        
        # get the fist number in the string
        values['Screen Size Class'] = values['Screen Size Class'].str.extract('(\d+\.?\d*)', expand=False).astype(float)

        # Convert the extracted numbers to float, round them to the nearest large integer, and then convert to int
        values['Screen Size Class'] = np.ceil(values['Screen Size Class'])
        values['Screen Size Class'] = values['Screen Size Class'].fillna(-1).astype(int).astype(str) + 'inch'
        values['Screen Size Class'] = values['Screen Size Class'].replace('-1inch', np.nan)
        
        # values['Screen Size Class'] = values['Screen Size Class'].fillna(-1).astype(int)
        # values['Screen Size Class'] =values['Screen Size Class'].astype(str) + 'inch'
        # print(values['Screen Refresh Rate'])
        # nan_count = values['Screen Size Class'] .isna().sum()
        # print(f'The number of NaN values in the column is: {nan_count}')
        # nan_count1 = values['Screen Refresh Rate'] .isna().sum()
        # print(f'The number of NaN values in the column is: {nan_count1}')
        # print( values['Screen Refresh Rate'].isna().sum())
        
        # drop the data have too much NaN and replace 
        data_dropno = values.dropna(thresh = (values.shape[0]) * self.threshold, axis=1)
        features = data_dropno.loc[:, data_dropno.columns.difference(['shop', 'modelID', 'title'])].copy().fillna(' ')
        features_change = features.replace({'\'':'inch ', 'and':' ', '\|': ' ', ',': ' ', '\"': 'inch', ' x ':'x'}, regex=True)
        data_dropnan = data_dropno.loc[:,['shop', 'modelID', 'title']].join(features_change)
        
        # get cleaned data and generate a new column that includes all key words
        cleaned_data = data_dropnan.apply(lambda x: x.astype(str).str.lower())
        cleaned_data['identitifer'] = cleaned_data.iloc[:, ~cleaned_data.columns.isin(['modelID'])].astype(str).agg(' '.join, axis=1)

        # get id        
        id = cleaned_data['identitifer'].str.split().apply(set).apply(list)
        
        # fill in brands
        for i in range(len(cleaned_data["Brand"])):
            if cleaned_data["Brand"][i]:
                for ele in id[i]:
                    if ele in self.brands:
                        cleaned_data["Brand"][i] = ele
                        break
        
        # get unique words
        uniqueWords = []
        for row in id:
            for element in row:
                if element not in uniqueWords and element not in self.brands:
                    uniqueWords.append(element)
        
        return cleaned_data, uniqueWords, id
    
    def getWords(self, cleaned_data):
        unique_words_list = set(cleaned_data['identitifer'].str.split().sum())

        # Convert the set to a list (if needed)
        unique_words_list = list(unique_words_list)
        return unique_words_list
        
    # Define a function to remove duplicate words in a string
    def remove_duplicates(self, text):
        words = text.split()
        unique_words = list(dict.fromkeys(words))
        return ' '.join(unique_words)
    
    def getBrand(self, cleaned_data):
        return cleaned_data['Brand']
    
    def getShop(self, cleaned_data):
        return cleaned_data['shop']
    
    def getScreenSize(self, cleaned_data):
        return cleaned_data['Screen Size Class']
    
    def getRefresh(self, cleaned_data):
        return cleaned_data['Screen Refresh Rate']