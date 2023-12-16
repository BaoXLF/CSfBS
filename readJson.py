import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class readJson:
    def __init__(self, filename) :
        self.filename = filename
        self.df = pd.DataFrame(data = None)
        self.readJson()
        
    def readJson (self):
        
        with open(self.filename, "r") as json_file:
            self.file = json.load(json_file)
        self.df = self.getDataFrame()
            
    def getDataFrame(self):
        
        dataset = [item for sublist in self.file.values() for item in sublist]
        df = pd.DataFrame(dataset)
        return df

            
    def splitData(self, ratio):
        
        (trainData, testData) = train_test_split(self.df, test_size=(1-ratio))
        return trainData,testData
    
    def getBootstrapSamples(self, numbbooststrap, ratio):
        
        bootstrap_samples = []  # List to hold the bootstrap samples
        test_data = [] 
        
        for i in range(numbbooststrap):
            sample = self.df.sample(frac = ratio, replace=True, random_state = i) # Create a bootstrap sample
            df2 = self.df.loc[self.df.index.difference(sample.index)]
            bootstrap_samples.append(sample.reset_index(drop=True))  # Append the sample to the list
            test_data.append(df2.reset_index(drop=True))
        
        return bootstrap_samples, test_data
            