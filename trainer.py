#Training with class weights & early stopping

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

def computeClassWeights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )
    return dict(zip(classes, weights))

def trainModel(
    model,
    trainingData,
    trainingLabels,
    classWeights
):
    earlyStopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        trainingData,
        trainingLabels,
        validation_split=0.1,
        epochs=10,
        batch_size=64,
        class_weight=classWeights,
        callbacks=[earlyStopping]
    )

    return history
