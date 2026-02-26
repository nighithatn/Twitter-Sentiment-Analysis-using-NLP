#Model evaluation

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluateModel(model, testData, testLabels):
    predictionProbabilities = model.predict(testData)
    predictedLabels = np.argmax(predictionProbabilities, axis=1)

    metrics = {
        "accuracy": accuracy_score(testLabels, predictedLabels),
        "precision": precision_score(testLabels, predictedLabels, average="weighted"),
        "recall": recall_score(testLabels, predictedLabels, average="weighted"),
        "f1Score": f1_score(testLabels, predictedLabels, average="weighted"),
        "confusionMatrix": confusion_matrix(testLabels, predictedLabels),
        "classificationReport": classification_report(testLabels, predictedLabels)
    }

    return metrics
