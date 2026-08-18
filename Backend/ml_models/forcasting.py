import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
import pickle
import os

class SalesForecaster:

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        self.monthly_data = None
        self.is_trained = False

    def prepare_data(self, df):
        """Create monthly features from raw dataframe."""

        df = df.copy()
        df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, format='mixed')
        df['Year'] = df['Order Date'].dt.year
        df['Month'] = df['Order Date'].dt.month

        monthly = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
        monthly = monthly.sort_values(['Year', 'Month']).reset_index(drop=True)

        monthly['Quarter'] = ((monthly['Month'] - 1) // 3) + 1
        monthly['Month_Sin'] = np.sin(2 * np.pi * monthly['Month'] / 12)
        monthly['Month_Cos'] = np.cos(2 * np.pi * monthly['Month'] / 12)

        monthly['Lag_1'] = monthly['Sales'].shift(1)
        monthly['Lag_2'] = monthly['Sales'].shift(2)
        monthly['Lag_3'] = monthly['Sales'].shift(3)
        monthly['Rolling_3'] = monthly['Sales'].shift(1).rolling(3).mean()

        monthly = monthly.dropna().reset_index(drop=True)
        self.monthly_data = monthly

        return monthly
    
    def train(self, df):
        """Train the model."""

        monthly = self.prepare_data(df)

        features = ['Year', 'Month', 'Quarter', 'Month_Sin', 'Month_Cos', 'Lag_1', 'Lag_2', 'Lag_3', 'Rolling_3']

        X = monthly[features]
        y = monthly['Sales']

        X_train = X.iloc[:-6]
        X_test = X.iloc[-6:]
        y_train = y.iloc[:-6]
        y_test = y.iloc[-6:]

        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print("Model trained!")
        print(f"MAPE : {mape*100:.2f}%")
        print(f"Accuracy : {(1-mape)*100:.2f}%")

        return {
            "mape": round(mape*100, 2),
            "accuracy": round((1 - mape)*100, 2),
            "training_examples": len(X_train)
        }
    
    def predict_next_months(self, months=3):
        """Predict Future Months"""

        if not self.is_trained:
            raise Exception(
                "First train the model!"
            )
        
        last_sales = self.monthly_data['Sales'].values
        last_row = self.monthly_data.iloc[-1]

        predictions = []

        for i in range(months):
            next_month = (int(last_row['Month']) + i)% 12 + 1
            next_year = int(last_row['Year']) + ((int(last_row['Month']) + i)// 12)

            next_features = pd.DataFrame([{
                'Year': next_year,
                'Month': next_month,
                'Quarter': ((next_month - 1)//3) + 1,
                'Month_Sin': np.sin(2*np.pi*next_month/12),
                'Month_Cos': np.cos(2*np.pi*next_month/12),
                'Lag_1': last_sales[-1],
                'Lag_2': last_sales[-2],
                'Lag_3': last_sales[-3],
                'Rolling_3': np.mean(last_sales[-3:])
            }])

            pred = self.model.predict(next_features)[0]

            predictions.append({
                'year': next_year,
                'month': next_month,
                'predicted_sales': round(pred, 2),
                'month_name': pd.Timestamp(year=next_year, month=next_month, day=1).strftime('%B %Y')
            })
            last_sales = np.append(last_sales, pred)

        return predictions
    
    def get_monthly_data(self):
        """return the historical monthly data."""
        if self.monthly_data is None:
            raise Exception("prepate the data first!")
        return self.monthly_data
    
    def save_model(self, path='models/forecaster.pkl'):
        """Save the model."""

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved : {path}")

    @staticmethod
    def load_model(path='model/forecaster.pkl'):
        """load the saved model"""
        with open(path, 'rb') as f:
            return pickle.load(f)
        
df = pd.read_csv('data/train.csv', encoding='latin-1')

forecaster = SalesForecaster()
metrics = forecaster.train(df)

print(f"Metrics : {metrics}")

predictions = forecaster.predict_next_months(3)

print("Next 3 Months : ")
for p in predictions:
    print(f"  {p['month_name']}: "
            f"${p['predicted_sales']:,.2f}"
            )
