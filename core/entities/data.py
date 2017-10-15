class Data(object):
    train = []
    validation = []
    test = []

    def make(self, train, valid, test):
        data = Data()
	data.train = train
	data.validation = valid
	data.test = test
        return data

    def __dir__(self):
       return dict({"train": self.train, "validation": self.validation, "test": self.test})

