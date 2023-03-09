import sys

from train import Train
from predict import Predict
from data import Data, Prep_Data

X = ['rnb', 'rap', 'electronic', 
     'rock', 'new age', 'classical',
     'reggae', 'blues', 'country', 'world',
     'folk', 'easy listening', 'jazz', 'vocal', 
     'children\'s', 'punk', 'alternative', 'spoken word', 
     'pop', 'heavy metal']
X_additional = ['rnb', 'rap', 'electronic', 
     'rock', 'new age', 'classical',
     'reggae', 'blues', 'country', 'world',
     'folk', 'easy listening', 'jazz', 'vocal', 
     'children\'s', 'punk', 'alternative', 'spoken word', 
     'pop', 'heavy metal',
     'rnb_left', 'rap_left', 'electronic_left', 
     'rock_left', 'new age_left', 'classical_left',
     'reggae_left', 'blues_left', 'world_left',
     'folk_left', 'easy listening_left', 'jazz_left', 'vocal_left', 
     'children\'s_left', 'punk_left', 'alternative_left', 'spoken word_left', 
     'pop_left', 'heavy metal_left']

try:
     command = sys.argv[1].lower()
except IndexError:
     print('No command found. Please type help or -h to see list of commands')
     exit()
     
if (command == 'help' or command == '-h'):
     print() 
     print('Commands available:')    
     print('1. convert or -c for data conversion')
     print('2. prep or -dp for Data preparation')
     print('3. train or -t for model training')
     print('4. predict or -p for predictions')
     print()
     
elif (command == 'convert' or command == '-c'):
     data = Data()
     df_users = data.txt_to_csv('LFM-1b_users.txt', 
                         index_col = 'user_id', 
                         to_drop_col = ['playcount', 'registered_unixtime'],
                         to_drop_row = ['country'])
     df_users_additional_data_w = data.txt_to_csv('LFM-1b_user_count_allmusic_w.txt', index_col = 'user_id')
     df_users_additional_data_nw = data.txt_to_csv('LFM-1b_user_count_allmusic_nw.txt', index_col = 'user_id')
     df_additional_data = data.join_df(df_users_additional_data_w, df_users_additional_data_nw, 'user_id', how = 'inner')
     df = data.join_df(df_users, df_additional_data, 'user_id', how = 'inner')
     train, test = data.seperate_train_text(df, ['m', 'f'])
     data.save_csv(train, 'lfm_train.csv')
     data.save_csv(test, 'lfm_test.csv')

elif (command == 'prep' or command == '-dp'):
     Prep_Data('lfm_train.csv', 
               to_drop = ['user_id', 'country_left', 'age'], 
               normalize = '', 
               normalize_cols = X,
               csv_name = 'lfm_train_normalize.csv',
               train = True)

     Prep_Data('lfm_test.csv', 
               to_drop = ['country_left', 'age'], 
               normalize = '', 
               normalize_cols = X,
               csv_name = 'lfm_test_normalize.csv')

elif (command == 'train' or command == '-t'):
     Train('lfm_train_normalize.csv',
          X = X,
          y = 'gender',
          model_name = 'model.pkl',
          model_type = 'random_forest')

elif (command == 'predict' or command == '-p'):
     Predict(in_file = 'lfm_test.csv', out_file = 'lfm_predict.csv', X = X, model_name = 'model.pkl')

else:
     print('Command not recognized. Please type help or -h to see list of commands')
     