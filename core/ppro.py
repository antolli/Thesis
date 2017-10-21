import os
import collections
import numpy as np
from util import Reader
from entities.data import Data
from entities.dictionary import Dictionary
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences


class PreProcessing():

        types = ['Training', 'Validation', 'Test']    
        dicty = Dictionary()

        def load_data(self, dataset):
            util = Reader()
            x, y, f = {}, {}, {} 
            for i in range(len(PreProcessing.types)):
                typ = PreProcessing.types[i]
                x[typ], y[typ], f[typ] = util.read_terms(dataset, typ)

            x_obj = Data().make(x['Training'],x['Validation'],x['Test']);
            y_obj = Data().make(y['Training'],y['Validation'],y['Test']);
            files = Data().make(f['Training'],f['Validation'],f['Test']);

            print('completed: load_data')
            return x_obj, y_obj, files

        def build_indices(self, x):
            tokenized_documents = x.train + x.validation + x.test
            PreProcessing.dicty.fit_on_texts(tokenized_documents)
            word_index = PreProcessing.dicty.word_index
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

        def build_dataset(self, x):
            data = Data()
            d = x.__dir__()
            for key, value in d.items():
                numb_seq = PreProcessing.dicty.texts_to_sequences(value)
                if key == "train": 
		   data.train = np.array(numb_seq)
		elif key == "test":
                   data.test = np.array(numb_seq)
                else: 
                   data.validation = np.array(numb_seq)
            print('completed: build_dataset')  
            return data

        def truncate_pad(self, data, max_len):
            data.train = sequence.pad_sequences(data.train, maxlen=max_len, padding='post',truncating='post')
            data.test = sequence.pad_sequences(data.test, maxlen=max_len, padding='post',truncating='post')
            data.validation = sequence.pad_sequences(data.validation, maxlen=max_len,padding='post',truncating='post') 
            return data


        def convert_onehot(self, y): 

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
            return sample_weights
