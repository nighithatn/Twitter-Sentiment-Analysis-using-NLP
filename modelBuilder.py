#BiLSTM model architecture

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout

def buildBiLstmModel(
    vocabularySize,
    embeddingDimension,
    maxSequenceLength,
    numberOfClasses
):
    model = Sequential()

    model.add(
        Embedding(
            input_dim=vocabularySize,
            output_dim=embeddingDimension,
            input_length=maxSequenceLength
        )
    )

    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(numberOfClasses, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
