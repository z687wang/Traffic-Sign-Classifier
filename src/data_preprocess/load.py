import pickle
import numpy as np
import random
from matplotlib import pyplot as plt
import csv

class TrainingData:
    def __init__(self, training_file='../../data/train.p', validation_file='../../data/valid.p',
                  testing_file='../../data/test.p', name_file='../../data/signnames.csv'):
        self.training_file = training_file
        self.validation_file = validation_file
        self.testing_file = testing_file
        self.name_file = name_file
        self.name_list = np.genfromtxt(self.name_file, skip_header=1, dtype=[('myint', 'i8'), ('mysring', 'S55')],
                                    delimiter=',')
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

    def print_distribute(self):
        unique_train, counts_train = np.unique(self.y_train, return_counts=True)
        plt.bar(unique_train, counts_train)
        plt.grid()
        plt.title("Train Dataset Sign Counts")
        plt.savefig('train_sign_counts.png')

        unique_test, counts_test = np.unique(self.y_test, return_counts=True)
        plt.bar(unique_test, counts_test)
        plt.grid()
        plt.title("Test Dataset Sign Counts")
        plt.savefig('test_sign_counts.png')

        unique_valid, counts_valid = np.unique(self.y_valid, return_counts=True)
        plt.bar(unique_valid, counts_valid)
        plt.grid()
        plt.title("Valid Dataset Sign Counts")
        plt.savefig('valid_sign_counts.png')

    def print_examples(self):
        fig, axs = plt.subplots(4, 5, figsize=(15, 6))
        fig.subplots_adjust(hspace=.2, wspace=.001)
        axs = axs.ravel()
        for i in range(20):
            index = random.randint(0, len(self.X_train))
            image = self.X_train[index]
            axs[i].axis('off')
            axs[i].imshow(image)
            axs[i].set_title(self.name_list[self.y_train[index]][1].decode('ascii'), fontsize=10)
        plt.savefig('sign_examples')

if __name__ =='__main__':
    train_data = TrainingData()
    train_data.print_stats()
    train_data.print_distribute()
    train_data.print_examples()