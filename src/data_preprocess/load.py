import pickle
import numpy as np

class TrainingData:
    def __init__(self, training_file='../../data/train.p', validation_file='../../data/valid.p',
                  testing_file='../../data/test.p'):
        self.training_file = training_file
        self.validation_file = validation_file
        self.testing_file = testing_file
        self.load_data(self.training_file, self.validation_file, self.testing_file)

    def load_data(self, training_file='../../data/train.p', validation_file='../../data/valid.p',
                  testing_file='../../data/test.p'):
        with open(training_file, mode='rb') as f:
            train = pickle.load(f)

        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)

        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        self.X_train, self.y_train = train['features'], train['labels']
        self.X_valid, self.y_valid = valid['features'], valid['labels']
        self.X_test, self.y_test = test['features'], valid['labels']

        return self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test

    def print_stats(self):
        n_train = self.X_train.shape[0]
        n_valid = self.X_valid.shape[0]
        n_test = self.X_test.shape[0]
        img_shape = self.X_train.shape[1:]
        n_classes = len(np.unique(self.y_train))

        print("Number of training examples =", n_train)
        print("Number of validation examples =", n_valid)
        print("Number of testing examples =", n_test)
        print("Image data shape =", img_shape)
        print("Number of classes =", n_classes)

if __name__ =='__main__':
    train_data = TrainingData()
    train_data.print_stats()