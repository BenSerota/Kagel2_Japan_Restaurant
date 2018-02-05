# Mostly copied from:
# https://www.kaggle.com/nitinsurya/surprise-me-2-neural-networks-keras/notebook
# Best NB so far (31.1.18):
# https://www.kaggle.com/chenpu/surprise-me-2-neural-networks

import os
import numpy as np
import pandas as pd
from sklearn import preprocessing, ensemble, metrics
from xgboost import XGBRegressor
import keras
import keras.backend as K
from keras.layers import Embedding, Input, Dense
from keras.models import Model

# Load data

datadir = 'Data'

data = {
	'tra': pd.read_csv(os.path.join(datadir, 'air_visit_data.csv')),
	'as': pd.read_csv(os.path.join(datadir, 'air_store_info.csv')),
	'hs': pd.read_csv(os.path.join(datadir, 'hpg_store_info.csv')),
	'ar': pd.read_csv(os.path.join(datadir, 'air_reserve.csv')),
	'hr': pd.read_csv(os.path.join(datadir, 'hpg_reserve.csv')),
	'id': pd.read_csv(os.path.join(datadir, 'store_id_relation.csv')),
	'tes': pd.read_csv(os.path.join(datadir, 'sample_submission.csv')),
	'hol': pd.read_csv(os.path.join(datadir, 'date_info.csv')).rename(
		columns = {'calendar_date': 'visit_date'})
}

# Prepare data

data['hr'] = pd.merge(data['hr'], data['id'], how = 'inner',
                      on = ['hpg_store_id'])

for df in ['ar', 'hr']:
	data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
	data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
	data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
	data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
	data[df]['reserve_datetime_diff'] = data[df].apply(
		lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis = 1)
	tmp1 = \
	data[df].groupby(['air_store_id', 'visit_datetime'], as_index = False)[
		['reserve_datetime_diff', 'reserve_visitors']].sum().rename(
		columns = {'visit_datetime': 'visit_date',
		           'reserve_datetime_diff': 'rs1', 'reserve_visitors': 'rv1'})
	tmp2 = \
	data[df].groupby(['air_store_id', 'visit_datetime'], as_index = False)[
		['reserve_datetime_diff', 'reserve_visitors']].mean().rename(
		columns = {'visit_datetime': 'visit_date',
		           'reserve_datetime_diff': 'rs2', 'reserve_visitors': 'rv2'})
	tmp3 = \
		data[df].groupby(['air_store_id', 'visit_datetime'], as_index = False)[
			['reserve_datetime_diff', 'reserve_visitors']].var().rename(
				columns = {'visit_datetime': 'visit_date',
				           'reserve_datetime_diff': 'rs3',
				           'reserve_visitors': 'rv3'})
	data[df] = pd.merge(tmp1, tmp2, how = 'inner',
	                    on = ['air_store_id', 'visit_date'])
	data[df] = pd.merge(data[df], tmp3, how = 'inner',
	                    on = ['air_store_id', 'visit_date'])

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(
	lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(
	lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame(
		{'air_store_id': unique_stores, 'dow': [i] * len(unique_stores)}) for i
                    in range(7)], axis = 0, ignore_index = True).reset_index(
	drop = True)

# sure it can be compressed...
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index = False)[
	'visitors'].min().rename(columns = {'visitors': 'min_visitors'})
stores = pd.merge(stores, tmp, how = 'left', on = ['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index = False)[
	'visitors'].mean().rename(columns = {'visitors': 'mean_visitors'})
stores = pd.merge(stores, tmp, how = 'left', on = ['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index = False)[
	'visitors'].median().rename(columns = {'visitors': 'median_visitors'})
stores = pd.merge(stores, tmp, how = 'left', on = ['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index = False)[
	'visitors'].max().rename(columns = {'visitors': 'max_visitors'})
stores = pd.merge(stores, tmp, how = 'left', on = ['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index = False)[
	'visitors'].count().rename(columns = {'visitors': 'count_observations'})
stores = pd.merge(stores, tmp, how = 'left', on = ['air_store_id', 'dow'])

stores = pd.merge(stores, data['as'], how = 'left', on = ['air_store_id'])
# NEW FEATURES FROM Georgii Vyshnia
stores['air_genre_name'] = stores['air_genre_name'].map(
	lambda x: str(str(x).replace('/', ' ')))
stores['air_area_name'] = stores['air_area_name'].map(
	lambda x: str(str(x).replace('-', ' ')))
lbl = preprocessing.LabelEncoder()
for i in range(10):
	stores['air_genre_name' + str(i)] = lbl.fit_transform(
		stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(
			str(x).split(' ')) > i else ''))
	stores['air_area_name' + str(i)] = lbl.fit_transform(
		stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(
			str(x).split(' ')) > i else ''))
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
train = pd.merge(data['tra'], data['hol'], how = 'left', on = ['visit_date'])
test = pd.merge(data['tes'], data['hol'], how = 'left', on = ['visit_date'])

train = pd.merge(train, stores, how = 'inner', on = ['air_store_id', 'dow'])
test = pd.merge(test, stores, how = 'left', on = ['air_store_id', 'dow'])

for df in ['ar', 'hr']:
	train = pd.merge(train, data[df], how = 'left',
	                 on = ['air_store_id', 'visit_date'])
	test = pd.merge(test, data[df], how = 'left',
	                on = ['air_store_id', 'visit_date'])

train['id'] = train.apply(
	lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]),
	axis = 1)

train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2  # weighted?
train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2
train['total_reserv_dt_diff_var'] = (train['rs3_x'] + train['rs3_y'])

test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2
test['total_reserv_dt_diff_var'] = (test['rs3_x'] + test['rs3_y'])

# NEW FEATURES FROM JMBULL
train['date_int'] = train['visit_date'].apply(
	lambda x: x.strftime('%Y%m%d')).astype(int)
test['date_int'] = test['visit_date'].apply(
	lambda x: x.strftime('%Y%m%d')).astype(int)
train['var_max_lat'] = train['latitude'].max() - train['latitude']
train['var_max_long'] = train['longitude'].max() - train['longitude']
test['var_max_lat'] = test['latitude'].max() - test['latitude']
test['var_max_long'] = test['longitude'].max() - test['longitude']

# NEW FEATURES FROM Georgii Vyshnia
train['lon_plus_lat'] = train['longitude'] + train['latitude']
train['lon_minus_lat'] = train['longitude'] - train['latitude']
test['lon_plus_lat'] = test['longitude'] + test['latitude']
test['lon_minus_lat'] = test['longitude'] - test['latitude']

lbl = preprocessing.LabelEncoder()
train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
test['air_store_id2'] = lbl.transform(test['air_store_id'])

col = [c for c in train if
       c not in ['id', 'air_store_id', 'visit_date', 'visitors']]
train = train.fillna(-1)
test = test.fillna(-1)

# # Output the train and test data
# train.to_csv('train_final.csv', index = False)
# test.to_csv('test_final.csv', index = False)

# Neural network preprocessing
value_col = ['holiday_flg', 'min_visitors', 'mean_visitors', 'median_visitors',
             'max_visitors', 'count_observations', 'rs1_x', 'rv1_x', 'rs2_x',
             'rv2_x', 'rs3_x', 'rv3_x', 'rs1_y', 'rv1_y', 'rs2_y', 'rv2_y',
             'rs3_y', 'rv3_y', 'total_reserv_sum', 'total_reserv_mean',
             'total_reserv_dt_diff_mean', 'total_reserv_dt_diff_var',
             'date_int', 'var_max_lat', 'var_max_long', 'lon_plus_lat',
             'lon_minus_lat']

nn_col = value_col + ['dow', 'year', 'month', 'air_store_id2', 'air_area_name',
                      'air_genre_name', 'air_area_name0', 'air_area_name1',
                      'air_area_name2', 'air_area_name3', 'air_area_name4',
                      'air_area_name5', 'air_area_name6', 'air_genre_name0',
                      'air_genre_name1', 'air_genre_name2', 'air_genre_name3',
                      'air_genre_name4']

X = train.copy()
X_test = test[nn_col].copy()

value_scaler = preprocessing.MinMaxScaler()
for vcol in value_col:
	X[vcol] = value_scaler.fit_transform(
		X[vcol].values.astype(np.float64).reshape(-1, 1))
	X_test[vcol] = value_scaler.transform(
		X_test[vcol].values.astype(np.float64).reshape(-1, 1))

X_train = list(X[nn_col].T.as_matrix())
Y_train = np.log1p(X['visitors']).values
nn_train = [X_train, Y_train]
nn_test = [list(X_test[nn_col].T.as_matrix())]

print('Data preprocessed successfully. ')


# Root Mean Squared Error
def RMSE(y, pred):
	return metrics.mean_squared_error(y, pred) ** 0.5

# Define a neural network model
def get_nn_complete_model(train, hidden1_neurons = 35, hidden2_neurons = 15):
	"""
	Input:
		train:           train dataframe(used to define the input size of the
		embedding layer)
		hidden1_neurons: number of neurons in the first hidden layer
		hidden2_neurons: number of neurons in the second hidden layer
	Output:
		return 'keras neural network model'
	"""
	K.clear_session()
	
	air_store_id = Input(shape = (1,), dtype = 'int32', name = 'air_store_id')
	air_store_id_emb = Embedding(len(train['air_store_id2'].unique()) + 1, 15,
	                             input_shape = (1,),
	                             name = 'air_store_id_emb')(air_store_id)
	air_store_id_emb = keras.layers.Flatten(name = 'air_store_id_emb_flatten')(
		air_store_id_emb)
	
	dow = Input(shape = (1,), dtype = 'int32', name = 'dow')
	dow_emb = Embedding(8, 3, input_shape = (1,), name = 'dow_emb')(dow)
	dow_emb = keras.layers.Flatten(name = 'dow_emb_flatten')(dow_emb)
	
	month = Input(shape = (1,), dtype = 'int32', name = 'month')
	month_emb = Embedding(13, 3, input_shape = (1,), name = 'month_emb')(month)
	month_emb = keras.layers.Flatten(name = 'month_emb_flatten')(month_emb)
	
	air_area_name, air_genre_name = [], []
	air_area_name_emb, air_genre_name_emb = [], []
	for i in range(7):
		area_name_col = 'air_area_name' + str(i)
		air_area_name.append(
			Input(shape = (1,), dtype = 'int32', name = area_name_col))
		tmp = Embedding(len(train[area_name_col].unique()), 3,
		                input_shape = (1,),
		                name = area_name_col + '_emb')(air_area_name[-1])
		tmp = keras.layers.Flatten(name = area_name_col + '_emb_flatten')(tmp)
		air_area_name_emb.append(tmp)
		
		if i > 4:
			continue
		area_genre_col = 'air_genre_name' + str(i)
		air_genre_name.append(
			Input(shape = (1,), dtype = 'int32', name = area_genre_col))
		tmp = Embedding(len(train[area_genre_col].unique()), 3,
		                input_shape = (1,),
		                name = area_genre_col + '_emb')(air_genre_name[-1])
		tmp = keras.layers.Flatten(name = area_genre_col + '_emb_flatten')(tmp)
		air_genre_name_emb.append(tmp)
	
	air_genre_name_emb = keras.layers.concatenate(air_genre_name_emb)
	air_genre_name_emb = Dense(4, activation = 'sigmoid',
	                           name = 'final_air_genre_emb')(
			air_genre_name_emb)
	
	air_area_name_emb = keras.layers.concatenate(air_area_name_emb)
	air_area_name_emb = Dense(4, activation = 'sigmoid',
	                          name = 'final_air_area_emb')(air_area_name_emb)
	
	air_area_code = Input(shape = (1,), dtype = 'int32', name = 'air_area_code')
	air_area_code_emb = Embedding(len(train['air_area_name'].unique()), 8,
	                              input_shape = (1,),
	                              name = 'air_area_code_emb')(air_area_code)
	air_area_code_emb = keras.layers.Flatten(
		name = 'air_area_code_emb_flatten')(air_area_code_emb)
	
	air_genre_code = Input(shape = (1,), dtype = 'int32',
	                       name = 'air_genre_code')
	air_genre_code_emb = Embedding(len(train['air_genre_name'].unique()), 5,
	                               input_shape = (1,),
	                               name = 'air_genre_code_emb')(air_genre_code)
	air_genre_code_emb = keras.layers.Flatten(
		name = 'air_genre_code_emb_flatten')(air_genre_code_emb)
	
	holiday_flg = Input(shape = (1,), dtype = 'float32', name = 'holiday_flg')
	year = Input(shape = (1,), dtype = 'float32', name = 'year')
	min_visitors = Input(shape = (1,), dtype = 'float32', name = 'min_visitors')
	mean_visitors = Input(shape = (1,), dtype = 'float32',
	                      name = 'mean_visitors')
	median_visitors = Input(shape = (1,), dtype = 'float32',
	                        name = 'median_visitors')
	max_visitors = Input(shape = (1,), dtype = 'float32', name = 'max_visitors')
	count_observations = Input(shape = (1,), dtype = 'float32',
	                           name = 'count_observations')
	rs1_x = Input(shape = (1,), dtype = 'float32', name = 'rs1_x')
	rv1_x = Input(shape = (1,), dtype = 'float32', name = 'rv1_x')
	rs2_x = Input(shape = (1,), dtype = 'float32', name = 'rs2_x')
	rv2_x = Input(shape = (1,), dtype = 'float32', name = 'rv2_x')
	rs3_x = Input(shape = (1,), dtype = 'float32', name = 'rs3_x')
	rv3_x = Input(shape = (1,), dtype = 'float32', name = 'rv3_x')
	rs1_y = Input(shape = (1,), dtype = 'float32', name = 'rs1_y')
	rv1_y = Input(shape = (1,), dtype = 'float32', name = 'rv1_y')
	rs2_y = Input(shape = (1,), dtype = 'float32', name = 'rs2_y')
	rv2_y = Input(shape = (1,), dtype = 'float32', name = 'rv2_y')
	rs3_y = Input(shape = (1,), dtype = 'float32', name = 'rs3_y')
	rv3_y = Input(shape = (1,), dtype = 'float32', name = 'rv3_y')
	total_reserv_sum = Input(shape = (1,), dtype = 'float32',
	                         name = 'total_reserv_sum')
	total_reserv_mean = Input(shape = (1,), dtype = 'float32',
	                          name = 'total_reserv_mean')
	total_reserv_dt_diff_mean = Input(shape = (1,), dtype = 'float32',
	                                  name = 'total_reserv_dt_diff_mean')
	total_reserv_dt_diff_var = Input(shape = (1,), dtype = 'float32',
	                                  name = 'total_reserv_dt_diff_var')
	date_int = Input(shape = (1,), dtype = 'float32', name = 'date_int')
	var_max_lat = Input(shape = (1,), dtype = 'float32', name = 'var_max_lat')
	var_max_long = Input(shape = (1,), dtype = 'float32', name = 'var_max_long')
	lon_plus_lat = Input(shape = (1,), dtype = 'float32', name = 'lon_plus_lat')
	lon_minus_lat = Input(shape = (1,), dtype = 'float32',
	                      name = 'lon_minus_lat')
	
	date_emb = keras.layers.concatenate([dow_emb, month_emb, year,
	                                     holiday_flg])
	date_emb = Dense(5, activation = 'sigmoid', name = 'date_merged_emb')(
		date_emb)
	
	cat_layer = keras.layers.concatenate(
			[holiday_flg, min_visitors, mean_visitors, median_visitors,
			 max_visitors, count_observations, rs1_x, rv1_x, rs2_x, rv2_x,
			 rs3_x, rv3_x, rs1_y, rv1_y, rs2_y, rv2_y, rs3_y, rv3_y,
			 total_reserv_sum, total_reserv_mean, total_reserv_dt_diff_mean,
			 total_reserv_dt_diff_var, date_int, var_max_lat, var_max_long,
			 lon_plus_lat, lon_minus_lat, date_emb, air_area_name_emb,
			 air_genre_name_emb, air_area_code_emb, air_genre_code_emb,
			 air_store_id_emb])
	
	m = Dense(hidden1_neurons, name = 'hidden1',
	          kernel_initializer = keras.initializers.RandomNormal(
			          mean = 0.0, stddev = 0.05, seed = None))(cat_layer)
	m = keras.layers.LeakyReLU(alpha = 0.2)(m)
	m = keras.layers.BatchNormalization()(m)
	
	m1 = Dense(hidden2_neurons, name = 'hidden2')(m)
	m1 = keras.layers.LeakyReLU(alpha = 0.2)(m1)
	m = Dense(1, activation = 'relu')(m1)
	
	inp_ten = [
		holiday_flg, min_visitors, mean_visitors, median_visitors, max_visitors,
		count_observations,	rs1_x, rv1_x, rs2_x, rv2_x, rs3_x, rv3_x, rs1_y,
		rv1_y, rs2_y, rv2_y, rs3_y, rv3_y, total_reserv_sum, total_reserv_mean,
		total_reserv_dt_diff_mean, total_reserv_dt_diff_var, date_int,
		var_max_lat, var_max_long, lon_plus_lat, lon_minus_lat, dow, year,
		month, air_store_id, air_area_code, air_genre_code
	]
	inp_ten += air_area_name
	inp_ten += air_genre_name
	model = Model(inp_ten, m)
	model.compile(loss = 'mse', optimizer = 'rmsprop', metrics = [])
	
	return model


# Train models and predict

seed = 0
model1 = ensemble.GradientBoostingRegressor(learning_rate = 0.2,
                                            random_state = seed,
                                            n_estimators = 50,
                                            subsample = 0.8,
                                            max_depth = 10,
                                            verbose = 1)
# model2 = ensemble.RandomForestRegressor(random_state = seed,
#                                         n_estimators = 50,
#                                         max_features = 'sqrt',
#                                         max_depth = 10)
model3 = XGBRegressor(learning_rate = 0.1,
                      random_state = seed,
                      n_estimators = 100,
                      subsample = 0.8,
                      colsample_bytree = 0.8,
                      max_depth = 10)
model4 = get_nn_complete_model(train)

model1.fit(train[col], np.log1p(train['visitors'].values))
print("Model1 trained")
# model2.fit(train[col], np.log1p(train['visitors'].values))
# print("Model2 trained")
model3.fit(train[col], np.log1p(train['visitors'].values))
print("Model3 trained")
model4.fit(nn_train[0], nn_train[1],
           epochs = 100,
           batch_size = 512,
           shuffle = True,
           # callbacks = [lr_decay],
           verbose = 2)
print("Model4 trained")

score1 = RMSE(np.log1p(train['visitors'].values), model1.predict(train[col]))
# score2 = RMSE(np.log1p(train['visitors'].values), model2.predict(train[col]))
score3 = RMSE(np.log1p(train['visitors'].values), model3.predict(train[col]))
score4 = RMSE(np.log1p(train['visitors'].values),
              pd.Series(model4.predict(nn_train[0]
                                       ).reshape(-1)).clip(0, 6.8).values)

print('RMSLE GradientBoostingRegressor: ', score1)
# print('RMSLE RandomForestRegressor: ', score2)
print('RMSLE XGBRegressor: ', score3)
print('RMSLE NeuralNetwork: ', score4)

preds1 = model1.predict(test[col])
# preds2 = model2.predict(test[col])
preds3 = model3.predict(test[col])
preds4 = pd.Series(model4.predict(nn_test[0]).reshape(-1)).clip(0, 6.8).values

test['visitors'] = (1 / 3) * preds1 + (1 / 3) * preds3 + (1 / 3) * preds4
test['visitors'] = np.expm1(test['visitors']).clip(lower = 0.)
sub1 = test[['id', 'visitors']].copy()
sub1['preds1'] = np.expm1(pd.Series(preds1))
sub1['preds3'] = np.expm1(pd.Series(preds3))
sub1['preds4'] = np.expm1(pd.Series(preds4))
print("Model predictions done.")

# A weighted average prediction per store, day of week, and holidays (except for
# weekends). The weights are proportional to t^5. The output is somewhat
# amplified, then weight-averaged with previous predictions.

# from hklee
# https://www.kaggle.com/zeemeen/weighted-mean-comparisons-lb-0-497-1st/code

# Reload data
del data
data = {
	'tra': pd.read_csv(os.path.join(datadir, 'air_visit_data.csv')),
	'as': pd.read_csv(os.path.join(datadir, 'air_store_info.csv')),
	'hs': pd.read_csv(os.path.join(datadir, 'hpg_store_info.csv')),
	'ar': pd.read_csv(os.path.join(datadir, 'air_reserve.csv')),
	'hr': pd.read_csv(os.path.join(datadir, 'hpg_reserve.csv')),
	'id': pd.read_csv(os.path.join(datadir, 'store_id_relation.csv')),
	'tes': pd.read_csv(os.path.join(datadir, 'sample_submission.csv')),
	'hol': pd.read_csv(os.path.join(datadir, 'date_info.csv'))
}

wkend_holidays = data['hol'].apply(
		(lambda x: (x.day_of_week == 'Sunday' or
		            x.day_of_week == 'Saturday') and
		           x.holiday_flg == 1),
		axis = 1)
data['hol'].loc[wkend_holidays, 'holiday_flg'] = 0
data['hol']['weight'] = ((data['hol'].index + 1) / len(data['hol'])) ** 5

visit_data = data['tra'].merge(data['hol'], left_on = 'visit_date',
                               right_on = 'calendar_date', how = 'left')
visit_data.drop('calendar_date', axis = 1, inplace = True)
visit_data['visitors'] = visit_data.visitors.map(pd.np.log1p)

wmean = lambda x: ((x.weight * x.visitors).sum() / x.weight.sum())
visitors = visit_data.groupby(
		['air_store_id', 'day_of_week', 'holiday_flg']).apply(
	wmean).reset_index()
visitors.rename(columns = {0: 'visitors'},
                inplace = True)  # cumbersome, should be better ways.

data['tes']['air_store_id'] = data['tes'].id.map(
	lambda x: '_'.join(x.split('_')[:-1]))
data['tes']['calendar_date'] = data['tes'].id.map(
	lambda x: x.split('_')[2])
data['tes'].drop('visitors', axis = 1, inplace = True)
data['tes'] = data['tes'].merge(data['hol'], on = 'calendar_date', how = 'left')
data['tes'] = data['tes'].merge(visitors, on = [
	'air_store_id', 'day_of_week', 'holiday_flg'], how = 'left')

missings = data['tes'].visitors.isnull()
data['tes'].loc[missings, 'visitors'] = data['tes'][missings].merge(
		visitors[visitors.holiday_flg == 0],
		on = ('air_store_id', 'day_of_week'),
		how = 'left')['visitors_y'].values

data['tes']['visitors'] = data['tes'].visitors.map(pd.np.expm1)
sub2 = data['tes'][['id', 'visitors']].copy()
sub2 = sub2.fillna(-1)

# Generate output

def final_visitors(x):
	visitors_x, visitors_y = x['visitors_x'], x['visitors_y']
	if x['visitors_y'] == -1:
		return visitors_x
	else:
		return 0.7 * visitors_x + 0.3 * visitors_y * 1.1

sub_merge = pd.merge(sub1, sub2, on = 'id', how = 'inner')
sub_merge['visitors'] = sub_merge.apply(lambda x: final_visitors(x), axis = 1)
print("Done")
sub_merge.to_csv('results.csv', index = False)  # For debugging
sub_merge[['id', 'visitors']].to_csv('submission.csv', index = False)
