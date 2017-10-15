# BLSTM for sequence classification
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, TimeDistributed, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras import optimizers, regularizers
from ppro import PreProcessing
from util import Output

# Instantiate pprocessing
ppro = PreProcessing()

# Fix random seed for reproducibility
np.random.seed(7)

MODEL_PATH = 'trained_model.h5'
DATASET = 'Krapivin2009'  # Hulth2003, SemEval2010, Krapivin2009
EMBEDDING = 'glove.6B.50d.txt'
EMBEDDING_LENGTH = 50
MAX_REVIEW_LENGTH = 516

BATCH_SIZE = 32
EPOCHS = 10

x_loaded, y_loaded, files = ppro.load_data(DATASET) # load dataset

word_index = ppro.build_indices(x_loaded) # word_index: dictionary of the words in the docs

embeddings_matrix = ppro.build_dict_embeddings(word_index, EMBEDDING_LENGTH, EMBEDDING) # maps embedd. -> word_index
x = ppro.build_dataset(x_loaded, word_index) # replaces each word for its index in word_index

x = ppro.truncate_pad(x, MAX_REVIEW_LENGTH) # truncate and pad input sequences (to x)
y = ppro.truncate_pad(y_loaded, MAX_REVIEW_LENGTH) # idem (to y)
y = ppro.convert_onehot(y, MAX_REVIEW_LENGTH) # one-hot encoding 

# assigns different weights for each training example (Marco code)
sample_weights = ppro.diff_weights(x.train, y.train)
sample_weights_val = ppro.diff_weights(x.validation, y.validation) # also added for validation set

if not os.path.isfile(MODEL_PATH) : 

        model = Sequential()
        model.add(Embedding(len(word_index) + 1, 
                                                EMBEDDING_LENGTH, 
                                                weights=[embeddings_matrix],
                                                input_length=MAX_REVIEW_LENGTH,
                                                trainable=False, name='embedding'))
        model.add(Bidirectional(LSTM(516, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True), name='bi')) 
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(200, activation='relu',kernel_regularizer=regularizers.l2(0.01)), name='dense_relu'))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(3, activation='softmax'), name='out'))
        #model.load_weights('model_weights_kr.h5') fineTuning
        optimizer = optimizers.Adam(lr=0.006)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'], sample_weight_mode="temporal")
        print(model.summary())
        validation_data=(x.validation, y.validation, sample_weights_val) # will override validation_split.
        history = model.fit(x.train, y.train, 
                                             validation_split = 0.2,
                                             validation_data = validation_data,
                                             epochs=EPOCHS, batch_size=BATCH_SIZE, 
                                             sample_weight=sample_weights)
        scores = model.evaluate(x.test, y.test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        model.save(MODEL_PATH)  # creates a HDF5 file 'my_model.h5'
else :
        print("Found existing model in "+ MODEL_PATH)
        # returns a compiled model identical to the previous one
        model = load_model(MODEL_PATH)
        print("Model loaded from file")


print("Predicting...")
model_predict = model.predict(x=x.test, batch_size=32, verbose=0)
print("Saving to file...")
#instantiate output
output = Output()
output.generate_obtained_file(model_predict=model_predict, x=x_loaded.test, files=files.test)
print("Done.")
