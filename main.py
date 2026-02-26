from sklearn.model_selection import train_test_split

from dataLoader import loadTwitterDataset
from preprocessing import cleanText
from tokenizerUtils import createTokenizer, convertTextToPaddedSequences
from modelBuilder import buildBiLstmModel
from trainer import computeClassWeights, trainModel
from evaluator import evaluateModel
from saveUtils import saveModel, saveTokenizer

# Constants
datasetPath = "D:/BIA/DS_mini_project 3/Twitter_Data.csv"
vocabularySize = 20000
maxSequenceLength = 100
embeddingDimension = 128
numberOfClasses = 3

# Load dataset
dataset = loadTwitterDataset(datasetPath)

# Text preprocessing
dataset["cleanedText"] = dataset["text"].apply(cleanText)

# Encode sentiment labels
sentimentMapping = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

dataset["encodedSentiment"] = (
    dataset["sentiment"]
    .str.lower()
    .map(sentimentMapping)
)

# Drop invalid rows
dataset = dataset.dropna(subset=["cleanedText", "encodedSentiment"])

# Features & labels
texts = dataset["cleanedText"]
labels = dataset["encodedSentiment"].astype(int)


# Train-test split
trainTexts, testTexts, trainLabels, testLabels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Tokenization
tokenizer = createTokenizer(trainTexts, vocabularySize)

trainingPaddedSequences = convertTextToPaddedSequences(
    tokenizer, trainTexts, maxSequenceLength
)
testingPaddedSequences = convertTextToPaddedSequences(
    tokenizer, testTexts, maxSequenceLength
)

# Model
model = buildBiLstmModel(
    vocabularySize,
    embeddingDimension,
    maxSequenceLength,
    numberOfClasses
)

# Training
classWeights = computeClassWeights(trainLabels)
trainModel(model, trainingPaddedSequences, trainLabels, classWeights)

# Evaluation
evaluationMetrics = evaluateModel(
    model, testingPaddedSequences, testLabels
)

print(evaluationMetrics["classificationReport"])

# Save artifacts
saveModel(model, "twitterSentimentModel.h5")
saveTokenizer(tokenizer, "tokenizer.pkl")
