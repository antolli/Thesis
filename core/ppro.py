import os
import numpy as np
import collections
from util import Reader
from entities.data import Data
from keras.preprocessing import sequence
import keras.preprocessing.text as kerasPreProc
from keras.preprocessing.sequence import pad_sequences

class PreProcessing():

        types = ['Training', 'Validation', 'Test'] 
        tokenizer = kerasPreProc.Tokenizer()
        
        
        def load_data(self, dataset):
            util = Reader()
            x, y, files = [], [], [] 
            for i in range(len(PreProcessing.types)):
                x_, y_, f_ = util.read_terms(dataset, PreProcessing.types[i])
                x.append(x_)
                y.append(y_)
                files.append(f_)

            x = Data().make(x[0],x[1],x[2]); # the indices 0, 1 and 2 are the data of each type in "types", 
            y = Data().make(y[0],y[1],y[2]); # that is, Training, Validation and Test for "x", "y" and "files"
            files = Data().make(files[0],files[1],files[2]);

            print('completed: load_data')
            return x, y, files

        def build_indices(self, x_):
            texts = x_.train + x_.validation + x_.test
            PreProcessing.tokenizer.fit_on_texts(texts)
            word_index = PreProcessing.tokenizer.word_index
            print('Found %s unique tokens.' % len(word_index))
            print('completed: build_indices')
            return word_index

        # function based on the following tutorial: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
        def build_dict_embeddings(self, word_index, embedding_length, emb_path):
            embeddings_index = {}
            with open(os.path.join("embeddings/", emb_path)) as f:
                    for line in f:
                        values = line.split()
                        word = values[0]
                        coefs = np.asarray(values[1:], dtype='float32')
                        embeddings_index[word] = coefs
            embedding_matrix = np.zeros((len(word_index) + 1, embedding_length))
            for word, i in word_index.items():
                    embedding_vector = embeddings_index.get(word)
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector # words not found in embedding index will be all-zeros.
            print('completed: build_embeddings')  
            return embedding_matrix

        def build_dataset(self, x, word_index):

            x_train = [kerasPreProc.text_to_word_sequence(text) for text in x.train ]
            x_valid = [kerasPreProc.text_to_word_sequence(text) for text in x.validation ]
            x_test  = [kerasPreProc.text_to_word_sequence(text) for text in x.test ]

            f = lambda word: word_index.get(word.lower()) if word_index.has_key(word.lower()) else -1

            x.train = np.array([[f(word) for word in x_train[s]] for s in range(len(x_train))])
            x.test = np.array([[f(word) for word in x_test[s]] for s in range(len(x_test))])
            x.validation = np.array([[f(word) for word in x_valid[s]] for s in range(len(x_valid))])

            print('completed: build_dataset')  
            return x

        def truncate_pad(self, data, max_len):
            data.train = sequence.pad_sequences(data.train, maxlen=max_len, padding='post',truncating='post')
            data.test = sequence.pad_sequences(data.test, maxlen=max_len, padding='post',truncating='post')
            data.validation = sequence.pad_sequences(data.validation, maxlen=max_len,padding='post',truncating='post') 
            return data

        def convert_onehot(self, y, max_len):   
 
             g = lambda elem: [1,0,0] if elem == 0  else [0,1,0] if elem == 1 else [0,0,1]
             h = lambda a: map(g, a)
             y.train = np.apply_along_axis(h, axis=1, arr=y.train)
             y.test  = np.apply_along_axis(h, axis=1, arr=y.test)
             y.validation = np.apply_along_axis(h, axis=1, arr=y.validation)

             return y

        def diff_weights(self, x_, y_):
            # assigns different weights for each training example (Marco code)
            sample_weights = np.empty(np.shape(x_))
            for i,j in enumerate(y_) :
                for k,w in enumerate(j) :
                    if w[0] == 1 :
                        sample_weights[i,k] = 1;
                    else :
                        sample_weights[i,k] = 50;
             
            sample_weights_reshape = np.array(sample_weights).flatten() # ????

            return sample_weights
