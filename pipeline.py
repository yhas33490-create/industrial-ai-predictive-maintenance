import pandas as pd
import numpy as np
import joblib

def run_full_enterprise_pipeline(df_input, models, iso_forest):
    thresholds = {'H': 0.35, 'M': 0.15, 'L': 0.41}
    
    df_clean = df_input.copy()
    df_clean.columns = [c.replace('[', '_').replace(']', '').strip() for c in df_clean.columns]
    
    # Physics-Informed Feature Engineering
    df_clean['Temp_Diff'] = df_clean['Process temperature _K'] - df_clean['Air temperature _K']
    df_clean['Power'] = df_clean['Rotational speed _rpm'] * df_clean['Torque _Nm']
    df_clean['Power_kW'] = ((df_clean['Rotational speed _rpm'] * 2 * np.pi / 60) * df_clean['Torque _Nm']) / 1000.0
    df_clean['OSF_Metric'] = df_clean['Tool wear _min'] * df_clean['Torque _Nm']
    
    feature_cols = [
        'Air temperature _K', 'Process temperature _K', 
        'Rotational speed _rpm', 'Torque _Nm', 'Tool wear _min',
        'Temp_Diff', 'Power', 'OSF_Metric'
    ]
    
    anomaly_preds = iso_forest.predict(df_clean[feature_cols]) if iso_forest else [1] * len(df_clean)
    
    results = []
    for idx, row in df_clean.iterrows():
        region = row.get('Type', 'M')
        features_vec = row[feature_cols].values.reshape(1, -1)
        
        prob = models[region].predict_proba(features_vec)[0, 1] if models and region in models else 0.05
        th = thresholds.get(region, 0.25)
        
        osf_limit = 12000 if region == 'M' else 11000
        physics_hazard = (row['OSF_Metric'] >= osf_limit) or (row['Temp_Diff'] < 8.6)
        is_anomaly = anomaly_preds[idx] == -1
        
        causes = []
        if row['Tool wear _min'] >= 200: causes.append("Tool Wear Hazard (>200 min)")
        if row['Temp_Diff'] < 8.6: causes.append("Thermal Dissipation Strain (<8.6 K)")
        if row['OSF_Metric'] >= osf_limit: causes.append(f"Mechanical Stress Overstrain ({int(row['OSF_Metric'])})")
        if row['Torque _Nm'] > 60: causes.append("Excessive Torque (>60 Nm)")

        p_kw = round(row['Power_kW'], 2)
        energy_advice = f"Power Output: {p_kw} kW. " + ("Reduce torque to optimize energy." if row['Torque _Nm'] > 50 else "Balanced operation.")

        if prob >= th or physics_hazard:
            status, action = "🔴 Danger", "Immediate emergency shutdown & overhaul."
            est_cost, roi_saving = "$3,000 (Unplanned Downtime)", "$3,500 (Preventative Savings)"
            if not causes: causes.append("Combined Risk Factors")
        elif prob >= (th * 0.6) or row['Tool wear _min'] >= 170:
            status, action = "🟡 Warning", "Inspect components during scheduled maintenance window."
            est_cost, roi_saving = "$200 (Preventative Inspection)", "$2,200"
            if not causes: causes.append("Elevated Wear Probability")
        else:
            status, action = "🟢 Normal", "Normal operation, no action required."
            causes, est_cost, roi_saving = ["Operational Parameters Safe"], "$0", "$0"

        results.append({
            'UDI': row.get('UDI', idx),
            'Product ID': row.get('Product ID', 'N/A'),
            'Region': region,
            'Status': status,
            'Sensor Anomaly': "⚠️ Anomaly" if is_anomaly else "🟢 Normal",
            'Failure Prob (%)': round(prob * 100, 2),
            'Power (kW)': p_kw,
            'Primary Root Cause': " | ".join(causes),
            'Energy Advice': energy_advice,
            'Est. Downtime Risk Cost': est_cost,
            'Est. Savings via AI': roi_saving,
            'Action Recommended': action
        })
        
    return pd.DataFrame(results)
