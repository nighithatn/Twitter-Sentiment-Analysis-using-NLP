#Tokenization & padding

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



def createTokenizer(texts, vocabularySize):
    tokenizer = Tokenizer(
        num_words=vocabularySize,
        oov_token="<OOV>"
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer

def convertTextToPaddedSequences(tokenizer, texts, maxSequenceLength):
    sequences = tokenizer.texts_to_sequences(texts)
    paddedSequences = pad_sequences(
        sequences,
        maxlen=maxSequenceLength,
        padding="post"
    )
    return paddedSequences
