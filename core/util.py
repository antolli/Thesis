import os
import json
import collections
import numpy as np
from nltk import PorterStemmer
from entities.tokenizer import Tokenizer
from entities.dictionary import Dictionary

class Reader():

        tokenizer =  Tokenizer('eg')

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
                    
                    list_keyphr = [Reader.tokenizer.word_tokenize(kp) for kp in kp_uncontr.split(";")]
                    list_sentences = Reader.tokenizer.sentence_tokenize(text)

                    text_vec = []
                    for string in list_sentences:
                        for token in Reader.tokenizer.word_tokenize(string):
                            text_vec.append(token)

                    files_list.append(f)
                    x.append(text_vec)
                    y.append(self.calc_expected_values(text_vec, list_keyphr))
                "------------------ SEMEVAL 2010 DATASET----------------------------------------------------"              
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
                                dict_ann[name_doc] = [Reader.tokenizer.word_tokenize(kp.decode('utf-8')) for kp in kp_found]

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

                    list_sentences = Reader.tokenizer.sentence_tokenize(text.decode('utf-8'))

                    text_vec = []
                    for string in list_sentences:
                        for token in Reader.tokenizer.word_tokenize(string):
                            text_vec.append(token)
                    name_file = f.split('.')
                    files_list.append(name_file[0])
                    x.append(text_vec)
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

                               kp_list = d["keyword"].split(";") 
                               list_keyphr = [Reader.tokenizer.word_tokenize(kp.encode('utf-8')) for kp in kp_list]
                               list_sentences = Reader.tokenizer.sentence_tokenize(text.encode('utf-8'))

                               text_vec = []
                               for string in list_sentences:
                                   for token in Reader.tokenizer.word_tokenize(string):
                                       text_vec.append(token)

                               files_list.append(str(count_doc))
                               x.append(text_vec)
                               y.append(self.calc_expected_values(text_vec, list_keyphr))
                               count_doc = count_doc + 1
                
                "------------------ SEMEVAL 2017 DATASET------------------------------------------------------" 
                if dataset == "SemEval2017":
                    if not f.endswith(".ann"):
		       continue
		    f_anno = open(os.path.join(path, f), "rU")
		    f_text = open(os.path.join(path, f.replace(".ann", ".txt")), "rU")
                    text = "".join(map(str, f_text))
                    # code based on utility script (util.py) available on: https://scienceie.github.io/resources.html
                    kp_list = []
		    for l in f_anno:
		        anno_inst = l.strip("\n").split("\t")
		        if len(anno_inst) == 3:
		           anno_inst1 = anno_inst[1].split(" ")
		           if len(anno_inst1) == 3:
		              keytype, start, end = anno_inst1
		           else:
		              keytype, start, _, end = anno_inst1
		           if not keytype.endswith("-of"): # e.g.:Synonym-of
			      keyphr_ann = anno_inst[2] 
			      kp_list.append(keyphr_ann)

                    list_keyphr = [Reader.tokenizer.word_tokenize(kp.decode('utf-8')) for kp in kp_list]
                    list_sentences = Reader.tokenizer.sentence_tokenize(text.decode('utf-8'))

                    text_vec = []
                    for string in list_sentences:
                        for token in Reader.tokenizer.word_tokenize(string):
                            text_vec.append(token)

                    files_list.append(f)
                    x.append(text_vec)
                    y.append(self.calc_expected_values(text_vec, list_keyphr))
                "---------------------------------------------------------------------------------------------"   

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


"""
    Generates the output file to feed "distiller_metrics" system
    :attr dicty: instance of dictionary
    :def generate_obtained_file: 
    :: param model_predict: predictions generated 
    :: param x: the list of documents (each word represented by an index)
    :: param files: the list of file's names
    :: this method produces a txt file with the following aspect:
    ::: 'name_of_doc_1: keyphrase_1, keyphrase_2, ... ,\n' 
    ::: 'name_of_doc_2: keyphrase_1, keyphrase_2, ... ,\n' 
    ::
    :: return: nothing

"""
class Output():

       def __init__(self, dicty):
           self.dicty = dicty

       def generate_obtained_file(self, model_predict, x, files): 
           h = lambda a: a * 100
           percent_values = np.apply_along_axis(h, axis=1, arr=model_predict) # converts in percentual
           content = self.generate_content_file(percent_values, x, files) # generates the content
           with open('../results/obtained.txt','w') as f: 
                f.write(content.encode('utf-8'))

       def generate_content_file(self, percent_values, x, files):
           key_phrase = ''

           for row in range(len(percent_values)) : 
               words = self.dicty.tokens_to_words(x[row])
               
               content = ''.join([files[row], ': ']) if row == 0 else ''.join([content, files[row], ': '])
               for col in range(len(percent_values[row])):
                   category = np.argmax(percent_values[row,col]) # gets the max value's index
                   if (category > 0): # a keyword has index 1 or 2, so key_phrase is composed only by 1's and 2's 
                       if col < len(words):  
                          key_phrase = ''.join([key_phrase, words[col], ' ']) 
                   else: 
                       if key_phrase is not '':
                          content = ''.join([content, key_phrase, ', '])
                          key_phrase = '' 

               content = ''.join([content, '\n'])
              
           return content

