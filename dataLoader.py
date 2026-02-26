#Load dataset & basic cleaning

import pandas as pd

def loadTwitterDataset(datasetPath):
    dataset = pd.read_csv(datasetPath)
    dataset = dataset.dropna()
    return dataset
