import os
import time
import pickle
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

class Train:
    def __init__(self, df_name, X, y, model_name, model_type = 'logistic_regression', model_path = None, csv_path = None):
        self.model_type_dict = {
            'logistic_regression': LogisticRegression(),
            'sgd_classifier': SGDClassifier(),
            'random_forest': RandomForestClassifier(),
            'linear_svc': LinearSVC(),
            'svc': SVC(),
            'decision_tree': DecisionTreeClassifier(),
            'nn': MLPClassifier()
        }
        
        self.model_path = model_path if csv_path else '../model/'
        self.csv_path = csv_path if csv_path else '../data/LFM/csv/'
        if not os.path.isdir(model_path):
            os.makedir(model_path)
        df = pd.read_csv(os.path.join(self.csv_path, df_name))
        train_X, test_X, train_y, test_y = self.train_test_split(df, X, y)
        self.evaluate(df, X, y, train_X, test_X, train_y, test_y, model_type)
        model = self.train(df[X], df[y])
        self.save_model(model, model_name, self.model_path)
        
    def train_test_split(self, df, X, y, split = 0.05):
        return train_test_split(df[X], df[y], test_size = split, shuffle = True)
    
    def evaluate(self, df, X, y, train_X, test_X, train_y, test_y, model_type):
        model = self.model_type_dict[model_type]
        model.fit(train_X, train_y)
        # model.fit(df[X], df[y])
        print(model_type, 'has an accuracy of:', model.score(test_X, test_y))
    
    def train(self, X, y, model_type):
        start = time.time()
        model = self.model_type_dict[model_type]
        model.fit(X, y)
        print('Model training took {}s'.format(time.time() - start))
        return model
        
    def save_model(self, model, model_name, model_path):
        pickle.dump(model, open(os.path.join(model_path, model_name), 'wb'))
        print('Model saved at:',  os.path.join(model_path, model_name))