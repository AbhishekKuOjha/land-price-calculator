import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# 📊 Sample land price dataset
data = {
    'Location': ['Pune', 'Pune', 'Delhi', 'Delhi', 'Bangalore', 'Bangalore'],
    'Year':     [2021, 2022, 2021, 2022, 2021, 2022],
    'Price':    [3500, 3700, 4200, 4500, 3000, 3300]
}

df = pd.DataFrame(data)

# 🔢 Encode location
le = LabelEncoder()
df['Location_Code'] = le.fit_transform(df['Location'])

# 🧠 Model training
X = df[['Location_Code', 'Year']]
y = df['Price']
model = DecisionTreeRegressor()
model.fit(X, y)

# 💾 Save the model and location mapping
joblib.dump(model, 'land_price_model.pkl')
location_map = dict(zip(df['Location'], df['Location_Code']))
joblib.dump(location_map, 'location_map.pkl')

print("✅ Model and location_map saved successfully.")
