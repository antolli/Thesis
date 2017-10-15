import os
import json
import collections
import numpy as np
from nltk import PorterStemmer
import keras.preprocessing.text as kerasPreProc

class Reader():
                     
        def read_terms(self, dataset, typ):            
            x, y, files_list = [], [], []
            f_text, list_keyphr = [], []
            path = "datasets/%s/%s" % (dataset, typ) 

            for f in os.listdir(path):
                "------------------ HULTH DATASET----------------------------------------------------------"
                if dataset == "Hulth2003":
                    if not f.endswith(".uncontr"):
                        continue
                    f_uncontr = open(os.path.join(path, f), "rU")
                    f_text = open(os.path.join(path, f.replace(".uncontr", ".abstr")), "rU")
                    text = "".join(map(str, f_text))

                    kp_uncontr = "".join(map(str, f_uncontr))
                    list_keyphr = [ kerasPreProc.text_to_word_sequence(kp) for kp in kp_uncontr.split(";")]

                    text_vec = kerasPreProc.text_to_word_sequence(text)
                    x.append(text) 
                    files_list.append(f)
                    y.append(self.calc_expected_values(text_vec, list_keyphr))
                "------------------ SEMEVAL DATASET--------------------------------------------------------"
                if dataset == "SemEval2010":
                    if not list_keyphr:
                       dict_ann = {}
                       name_file_ann = 'train' if typ == 'Training' else 'test' if typ == 'Test' else 'trial'
                       name_file_ann = ''.join([name_file_ann, '.combined.final'])
                       with open(os.path.join(path, name_file_ann)) as f_ann:
                            # these splits are not used for the text processing.
                            linespl = [line.split(':') for line in f_ann]
                            for l in linespl: # It won't be used to feed the network.
                                name_doc = l[0].strip() # split each row by "name_doc" and "kp_found"
                                kp_found = l[1].split(',') 
                                dict_ann[name_doc] = [ kerasPreProc.text_to_word_sequence(kp) for kp in kp_found]
                    if not f.endswith(".txt.final"):
                        continue
                    f_text = open(os.path.join(path, f), "rU")
                    list_keyphr = dict_ann[f.split('.')[0]]
                    content = "".join(map(str, f_text))
                    if typ != 'Test':
                       text = content 
                    else: 
                       stm = lambda string: PorterStemmer().stem(string)
                       text = ''.join(map(stm, content))

                    text_vec = kerasPreProc.text_to_word_sequence(text)
                    x.append(text) 
                    files_list.append(f)
                    y.append(self.calc_expected_values(text_vec, list_keyphr))
                "------------------ KRAPIVIN DATASET----------------------------------------------------------"
                if dataset == "Krapivin2009":
                    file_name = ''.join(['ke20k_', typ.lower(), '.json'])
                    with open(os.path.join(path, file_name)) as f:
		         count_doc = 1
    		         for line in f:
		            if count_doc <= 2000: # needed because 20k is too large
	    		       d = json.loads(line)
			       text = ''.join([d["title"], d["abstract"]])
			       text_vec = kerasPreProc.text_to_word_sequence(text.encode("utf-8"))
			       x.append(text.encode("utf-8"))
			       files_list.append(count_doc) 
			       list_keyphr = [kerasPreProc.text_to_word_sequence(kp.encode("utf-8")) for kp in d["keyword"].split(";")] 
                               y.append(self.calc_expected_values(text_vec, list_keyphr))
                               count_doc = count_doc + 1
                "--------------------------------------------------------------------------------------------"             

            return  x, y, files_list

        def calc_expected_values(self, text_vec, list_keyphr):
            y_inner = np.zeros(np.shape(text_vec))

            f = lambda a,b: [x for x in xrange(len(a)) if a[x:x+len(b)] == b]
            
            for kp in list_keyphr:
                arr_indices = f(text_vec, kp) # returns the indices at which the pattern starts
                for i in arr_indices: 
                    y_inner[i] = 1
                    if len(kp) > 1:
                       y_inner[(i+1):(i+1)+len(kp)-1] = 2
            return y_inner 

class Output():
        
        def generate_obtained_file(self, model_predict, x, files): 
                h = lambda a: a * 100
                percent_values = np.apply_along_axis(h, axis=1, arr=model_predict) # converts in percentual

                doc  = [kerasPreProc.text_to_word_sequence(text) for text in x ]
                with open('../results/obtained.txt','w') as f: 
                     key_phrase = ''
                     for row in range(len(percent_values)) : 
                         # files[row] contains the file name
                         content = ''.join([files[row], ': ']) if row == 0 else ''.join([content, files[row], ': '])
                         for col in range(len(percent_values[row])):
                             category = np.argmax(percent_values[row,col]) # gets the max value's index
                             if (category > 0): # a keyword has index 1 or 2, so key_phrase is composed only by 1's and 2's 
                                 if col < len(doc[row]): key_phrase = ''.join([key_phrase, doc[row][col], ' ']) 
                             else: 
                                 if key_phrase is not '':
                                    content = ''.join([content, key_phrase, ', '])
                                    key_phrase = '' 

                         content = ''.join([content, '\n'])

                     f.write(content)
