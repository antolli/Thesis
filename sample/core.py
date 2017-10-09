# LSTM for sequence classification in the wikinews dataset
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
from ppro  import PreProcessing
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
from keras.callbacks import LambdaCallback
from keras.layers.wrappers import Bidirectional
from keras.models import load_model
from keras import regularizers
from output import Output

#instantiate pprocessing
ppro = PreProcessing()

# fix random seed for reproducibility
np.random.seed(7)

model_path = 'trained_model.h5'
top_words = 20000
path_train = 'Hulth2003/Training' # path of data train
path_test = 'Hulth2003/Test' # path of data test

x_train, y_train, files_train = ppro.load_data(path_train)
x_test, y_test, files_test = ppro.load_data(path_test)
x_all = x_train + x_test

# Glov usa 100 features per l'embedding
embedding_vector_length = 300

#word_index: dizionario delle parole che esistono nei nostri doc
word_index = ppro.build_indices(x_all, top_words)
#costruisce la matrice di embedding per ogni parola che ho in word_index
embeddings_matrix = ppro.build_dict_embeddings(word_index,embedding_vector_length)

#sostituisce ogni parola della sentenza con l'indice in word_index
X_train, doc_train = ppro.build_dataset(x_train, word_index)
X_test, doc_test = ppro.build_dataset(x_test, word_index)

print(("Max length of training documents : " + str(max([len(seq) for seq in X_train]))))
print(("Max length of test documents     : " + str(max([len(seq) for seq in X_test]))))


# truncate and pad input sequences
max_review_length = 520
saved_y_test = y_test
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length,padding='post',truncating='post')
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length,padding='post',truncating='post')
y_train = sequence.pad_sequences(y_train, maxlen=max_review_length,padding='post',truncating='post')
y_test = sequence.pad_sequences(y_test, maxlen=max_review_length,padding='post',truncating='post')

y_train = ppro.change_y(y_train)
y_test = ppro.change_y(y_test)


# assegna pesi diversi per ogni esempio di training
sample_weights = np.empty(np.shape(X_train))

for i,y in enumerate(y_train) :
    for k,w in enumerate(y) :
        if w[0] == 1 :
            sample_weights[i,k] = 1;
        else :
            sample_weights[i,k] = 50;
             
sample_weights_reshape = np.array(sample_weights).flatten()

# save the losses for each batch iter.
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

if not os.path.isfile(model_path) : 

        # creo il modello di tipo sequenziale
        model = Sequential()
        # aggiungo la prima layer che deve per forza contenere le dimensioni dell'input
        # in effetti l'input sono l'embeddings 

        model.add(Embedding(len(word_index) + 1,
                                        embedding_vector_length,
                                        weights=[embeddings_matrix],
                                        input_length=max_review_length,
                                        trainable=False))
        #print model.output_shape
        model.add(Bidirectional(LSTM(500,activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True))) # il primo parametro sono le "unita": dimensioni dello spazio di output
        #model.add(Dropout(0.25))
        #model.add(TimeDistributed(Dense(200, activation='relu',kernel_regularizer=regularizers.l2(0.01))))
        model.add(Dropout(0.25))
        model.add(TimeDistributed(Dense(3, activation='softmax')))
        # per problemi categorici devo usare queste config.
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], sample_weight_mode="temporal")
        print((model.summary()))
        # in teoria ho creato il mio callbacker
        #history = LossHistory()
        batch_size = 32
        epochs=10
        #validation_data=[X_test,y_test,sample_weights_val]
        history = model.fit(X_train, y_train, validation_split = 0.2,epochs=epochs, batch_size=batch_size, sample_weight = sample_weights)
        # print history.losses
        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print(("Accuracy: %.2f%%" % (scores[1]*100)))

        # plot 
        #plt.plot(np.arange(batch_size*epochs), history.losses)
        #plt.show()

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

        #myFile = open("loss_values.csv","w")
        #myFile.write("Interaction" + "," + "Loss value" +"\n") #column headers

        #for i in range(len(history.losses)):
        #    q=range(i)
        #    g=str(history.losses[i])   
        #    myFile.write(str(i) + "," + g +"\n")
        #myFile.close() 

        model.save(model_path)  # creates a HDF5 file 'my_model.h5'
else :
        print(("Found existing model in "+ model_path))
        # returns a compiled model
        # identical to the previous one
        model = load_model(model_path)
        print("Model loaded from file")


#TEST
print("Predicting...")
output = model.predict(x=X_test, batch_size=32, verbose=0)
print("Saving to file...")
#instantiate output
calcout = Output()
calcout.return_selected_values(d=saved_y_test, y=output, files=files_test, X=X_test, doc_=doc_test)

print(output)
print(X_test)

print("Done.")
