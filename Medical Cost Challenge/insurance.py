import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

dependent_variable = ['charges']
independent_variables = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']


class Insurance:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(self.dataset_path)
        
        y = self.df[dependent_variable].values.ravel()
        x = self.df[independent_variables]

        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 4, 5])], remainder='passthrough')
        x = ct.fit_transform(x)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        sc_x = RobustScaler()
        self.x_train = sc_x.fit_transform(self.x_train)
        self.x_test = sc_x.transform(self.x_test)

        # Parameter grids for grid search (model_name -> grid)
        self.param_grids = {
            'rf': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
            },
            'xgb': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
            },
            'svr': {
                'C': [0.1, 1, 10],
                'epsilon': [0.01, 0.1, 0.2],
                'kernel': ['rbf', 'linear'],
            },
        }

    def train(self):
        self.xgb_model = XGBRegressor(n_estimators=100, random_state=42)
        self.xgb_model.fit(self.x_train, self.y_train)
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(self.x_train, self.y_train)
        self.svr_model = SVR(kernel='rbf')
        self.svr_model.fit(self.x_train, self.y_train)
        self.lr_model = LinearRegression()
        self.lr_model.fit(self.x_train, self.y_train)

    def predict(self):
        self.y_pred_xgb = self.xgb_model.predict(self.x_test)
        self.y_pred_rf = self.rf_model.predict(self.x_test)
        self.y_pred_svr = self.svr_model.predict(self.x_test)
        self.y_pred_lr = self.lr_model.predict(self.x_test)
        

    def evaluate(self, n_sample=10):
        # R2 scores block
        r2_xgb = r2_score(self.y_test, self.y_pred_xgb)
        r2_rf = r2_score(self.y_test, self.y_pred_rf)
        r2_svr = r2_score(self.y_test, self.y_pred_svr)
        r2_lr = r2_score(self.y_test, self.y_pred_lr)
        print('─' * 40)
        print('  R² scores (test set)')
        print('─' * 40)
        print(f'  XGB   {r2_xgb:>8.4f}')
        print(f'  RF    {r2_rf:>8.4f}')
        print(f'  SVR   {r2_svr:>8.4f}')
        print(f'  LR    {r2_lr:>8.4f}')
        print('─' * 40)

        # First n_sample: actual vs predicted table (currency-style)
        n = min(n_sample, len(self.y_test))
        rows = []
        for i in range(n):
            rows.append({
                '#': i + 1,
                'Actual': self.y_test[i],
                'XGB': self.y_pred_xgb[i],
                'RF': self.y_pred_rf[i],
                'SVR': self.y_pred_svr[i],
                'LR': self.y_pred_lr[i],
            })
        df = pd.DataFrame(rows)
        # Format as currency (2 decimals, no scientific notation)
        money_cols = ['Actual', 'XGB', 'RF', 'SVR', 'LR']
        df_display = df.copy()
        for c in money_cols:
            df_display[c] = df_display[c].apply(lambda x: f'{x:>12,.2f}')
        print(f'\n  First {n} predictions (actual vs predicted charges)')
        print('─' * 70)
        print(df_display.to_string(index=False))
        print('─' * 70)


    def grid_search(self, model_name='rf', cv=3, verbose=1):
        """
        Run grid search for the given model. model_name: 'rf', 'xgb', or 'svr'.
        Fits the best estimator and sets the corresponding model attribute.
        """
        if model_name not in self.param_grids:
            raise ValueError(f"model_name must be one of {list(self.param_grids.keys())}")

        models = {
            'rf': RandomForestRegressor(random_state=42),
            'xgb': XGBRegressor(random_state=42),
            'svr': SVR(),
        }
        grid = GridSearchCV(
            models[model_name],
            self.param_grids[model_name],
            cv=cv,
            scoring='r2',
            verbose=verbose,
            n_jobs=-1,
        )
        grid.fit(self.x_train, self.y_train)

        attr = {'rf': 'rf_model', 'xgb': 'xgb_model', 'svr': 'svr_model'}[model_name]
        setattr(self, attr, grid.best_estimator_)

        print(f"Grid search ({model_name}): best R2 (CV) = {grid.best_score_:.4f}")
        print(f"Best params: {grid.best_params_}")
        return grid

insurance = Insurance('dataset/insurance.csv')
insurance.train()


insurance.grid_search('xgb', cv=10)
insurance.predict()
insurance.evaluate()


