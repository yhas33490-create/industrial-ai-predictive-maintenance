import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest

def train_and_save_models():
    # Generate synthetic training data matching dataset structure
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Air temperature _K': np.random.uniform(295, 305, n_samples),
        'Process temperature _K': np.random.uniform(305, 315, n_samples),
        'Rotational speed _rpm': np.random.uniform(1200, 2800, n_samples),
        'Torque _Nm': np.random.uniform(10, 75, n_samples),
        'Tool wear _min': np.random.uniform(0, 250, n_samples),
    }
    df = pd.DataFrame(data)
    df['Temp_Diff'] = df['Process temperature _K'] - df['Air temperature _K']
    df['Power'] = df['Rotational speed _rpm'] * df['Torque _Nm']
    df['OSF_Metric'] = df['Tool wear _min'] * df['Torque _Nm']
    
    # Target condition
    y = ((df['OSF_Metric'] > 11000) | (df['Temp_Diff'] < 8.6) | (df['Tool wear _min'] > 200)).astype(int)
    
    # Train region models (H, M, L)
    for region in ['H', 'M', 'L']:
        model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(df, y)
        joblib.dump(model, f'model_{region}.joblib')
        print(f"✅ Saved model_{region}.joblib")
        
    # Train Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(df)
    joblib.dump(iso, 'iso_forest.joblib')
    print("✅ Saved iso_forest.joblib")

if __name__ == "__main__":
    train_and_save_models()
