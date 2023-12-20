import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings

warnings.filterwarnings('ignore')

def save_model(model, scaler, model_filename='model.joblib', scaler_filename='scaler.joblib'):

    joblib.dump(model, model_filename)
    
   
    joblib.dump(scaler, scaler_filename)


def load_model(model_filename='model.joblib', scaler_filename='scaler.joblib'):
    
    model = joblib.load(model_filename)
    
   
    scaler = joblib.load(scaler_filename)
    
    
    if isinstance(scaler, LabelEncoder):
        scaler.handle_unknown = 'ignore'

    return model, scaler





df = pd.read_csv('for_sale_only.csv')

df['purpose'] = df['purpose'].replace('For Rent','For Sale')

df = df[df['property_type'] != 'Lower Portion']
df = df[df['property_type'] != 'Room']
df = df[df['property_type'] != 'Upper Portion']

df['property_type'].unique()

df = df.drop('purpose', axis=1)

label_encoder = LabelEncoder()


label_encoder = LabelEncoder()
df['property_type'] = label_encoder.fit_transform(df['property_type'])
df['city'] = label_encoder.fit_transform(df['city'])
df['Area_Type'] = label_encoder.fit_transform(df['Area_Type'])


joblib.dump(label_encoder, 'label_encoder.pkl')


df.head()

X = df.drop('price', axis=1)
y = df['price']
X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


rf = RandomForestRegressor()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


save_model(rf, label_encoder)


loaded_model, loaded_scaler = load_model()


