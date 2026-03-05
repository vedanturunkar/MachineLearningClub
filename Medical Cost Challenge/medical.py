import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
ds = pd.read_csv(r"/Users/vedanturunkar/Desktop/PY/ml/insurance.csv")
x = ds.drop(columns = ['charges'])
y = ds['charges'].values
x.loc[:, 'region'] = x['region'].map({'southeast' : 0, 'southwest': 1, 'norhwest': 2, 'northeast': 3})
x.loc[:, 'sex'] = x['sex'].map({'male': 0, 'female': 1})
x.loc[:, 'smoker'] = x['smoker'].map({'yes': 0, 'no': 1})
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42 ) #test size means 20 percent is used for test and random state is the seed
model = RandomForestRegressor
random_forest = model(random_state=42)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
mse = mean_absolute_error(y_test, y_pred)
r = r2_score(y_test, y_pred)

print(mse)
print(r)
