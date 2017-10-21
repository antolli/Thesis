import nltk

class Tokenizer:

      def __init__(self, language):
          pickle = 'english.pickle' # default

          if language == 'pt':
             pickle = 'portuguese.pickle'
          elif language == 'eg':
             pickle = 'english.pickle'
          else:
             pickle = 'english.pickle'

          self.tokenizer = nltk.data.load(''.join(['tokenizers/punkt/', pickle]))

      def sentence_tokenize(self, text):
          sentences = self.tokenizer.tokenize(text) 
          return sentences

      def word_tokenize(self, string):
          return nltk.word_tokenize(string.lower())
