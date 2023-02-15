from train import Train
from data import Data, Prep_Data

X = ['rnb', 'rap', 'electronic', 
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

Train('lfm_train_normalize.csv',
      X = X,
      y = 'gender',
      model_name = 'model.pkl',
      model_type = 'random_forest')