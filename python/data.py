import os
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

class Data:
    def __init__(self, txt_path = None, csv_path = None):
        self.txt_path = txt_path if txt_path else '../data/LFM/txt/'
        self.csv_path = csv_path if csv_path else '../data/LFM/csv/'
        
    def txt_to_csv(self, filename, index_col = None, to_drop_col = None, to_drop_row = None):
        if index_col:
            df = pd.read_csv(os.path.join(self.txt_path, filename), index_col = index_col, delimiter = '\t')
        else:
            pd.read_csv(os.path.join(self.txt_path, filename), delimiter = '\t')
        if to_drop_col:
            df = df.drop(to_drop_col, axis = 1)
        if to_drop_row:
            df = df.dropna(subset = to_drop_row)
        return df

    def join_df(self, df_1, df_2, on, how = ''):
        how = how if how else 'left'
        return df_1.join(df_2, on = on, how = how, lsuffix = '_left')        
    
    def seperate_train_text(self, df, criteria):
        train = df[df['gender'].isin(criteria)]
        test = df[~df['gender'].isin(criteria)]
        return train, test

    def save_csv(self, df, csv_name):
        df.to_csv(os.path.join(self.csv_path, csv_name))
        print('File saved at:', os.path.join(self.csv_path, csv_name))
        
class Prep_Data:
    def __init__(self, df_name, to_drop, normalize, normalize_cols, csv_name, train = False, csv_path = None):
        self.csv_path = csv_path if csv_path else '../data/LFM/csv/'
        df = pd.read_csv(os.path.join(self.csv_path, df_name))
        df = self.drop_cols(df, to_drop)
        # df = self.normalize(df, normalize, normalize_cols)
        if train:
            df = self.translate_target(df)
        self.save_data(df, csv_name)
    
    def drop_cols(self, df, to_drop):
        return df.drop(to_drop, axis = 1)

    def normalize(self, df, normalize, normalize_cols):
        if (normalize == 'minmaxscaler'):
            scaler = MinMaxScaler()
        elif (normalize == 'standardscaler'):
            scaler = StandardScaler()
        else:
            scaler = Normalizer()
        df[normalize_cols] = scaler.fit_transform(df[normalize_cols])
        return df
    def translate_target(self, df):
        df['gender'] = df['gender'].replace({'f': 0, 'm': 1})
        return df
        
    def save_data(self, df, csv_name):
        df.to_csv(os.path.join(self.csv_path, csv_name))
        print('File saved at:', os.path.join(self.csv_path, csv_name))