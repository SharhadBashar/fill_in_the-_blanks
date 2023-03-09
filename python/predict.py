import os
import pickle
import pandas as pd

class Predict:
    def __init__(self, in_file, out_file, X, model_name = 'model.pkl', model_path = None, csv_path = None):
        self.model_path = model_path if model_path else '../model/'
        self.csv_path = csv_path if csv_path else '../data/LFM/csv/'
        self.model = self.load_model(model_name, self.model_path)
        prediction = self.predict(in_file, X, self.model, self.csv_path)
        self.save(prediction, in_file, out_file, self.csv_path)
        
    def load_model(self, model_name, model_path):
        return pickle.load(open(os.path.join(model_path, model_name), 'rb'))
    
    def predict(self, in_file, X, model, csv_path):
        df = pd.read_csv(os.path.join(csv_path, in_file))
        return model.predict(df[X])
    
    def save(self, prediction, in_file, out_file, csv_path):
        df = pd.read_csv(os.path.join(csv_path, in_file))
        df['gender'] = prediction
        df.to_csv(os.path.join(csv_path, out_file))
        print('Predictions saved at:', os.path.join(csv_path, out_file))
