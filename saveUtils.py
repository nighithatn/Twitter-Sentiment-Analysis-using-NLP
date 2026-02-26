#Save model & tokenizer

import pickle

def saveModel(model, modelPath):
    model.save(modelPath)

def saveTokenizer(tokenizer, tokenizerPath):
    with open(tokenizerPath, "wb") as file:
        pickle.dump(tokenizer, file)
