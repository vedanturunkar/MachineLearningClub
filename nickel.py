import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
ds = pd.read_csv(r"/Users/vedanturunkar/Desktop/PY/ml/Folds5x2_pp.csv", delimiter=',')
x = ds[['AT', 'V', 'AP', 'RH']]
y = ds['PE']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42) #test size means 20 percent is used for test and random state is the seed
random_forest = RandomForestRegressor(random_state=42)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
r = r2_score(y_test, y_pred)
print(ds.head(5))
print(r)
# Predict the first 5 rows of the dataset
first_five_predictions = random_forest.predict(x.head(5))

# Display the results
print("Predictions for the first five rows:")
print(first_five_predictions)
