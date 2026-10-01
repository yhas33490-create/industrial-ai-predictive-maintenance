import io
import joblib
import numpy as np
import pandas as pd
import requests
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet


def load_all_artifacts():
    """Load pre-trained XGBoost models and Isolation Forest model."""
    models = {r: joblib.load(f'model_{r}.joblib') for r in ['H', 'M', 'L']}
    iso = joblib.load('iso_forest.joblib')
    return models, iso


def send_telegram_alert(bot_token: str, chat_id: str, message: str):
    """Send automated risk alert messages via Telegram Bot API."""
    if not bot_token or not chat_id:
        return False, "Please enter a valid Telegram Bot Token and Chat ID."
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code == 200:
            return True, "Alert successfully sent to Telegram! 📱"
        return False, f"Delivery failed: {res.text}"
    except Exception as ex:
        return False, f"Connection error: {str(ex)}"


def generate_pdf_report(df_res: pd.DataFrame) -> io.BytesIO:
    """Generate an Executive Summary PDF Report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    story = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor("#1A365D"))
    normal_style = styles['Normal']

    story.append(Paragraph("<b>Industrial AI Predictive Maintenance - Executive Summary Report</b>", title_style))
    story.append(Spacer(1, 15))

    total = len(df_res)
    danger = len(df_res[df_res['Status'].str.contains("🔴")])
    warning = len(df_res[df_res['Status'].str.contains("🟡")])
    normal = len(df_res[df_res['Status'].str.contains("🟢")])
    anomalies = len(df_res[df_res['Sensor Anomaly'] == "⚠️ Anomaly Detected"])

    summary_text = f"<b>Fleet Overview:</b> Total Machines Analyzed: {total} | Healthy: {normal} | Warning: {warning} | Critical Risk: {danger} | Sensor Anomalies: {anomalies}"
    story.append(Paragraph(summary_text, normal_style))
    story.append(Spacer(1, 15))

    critical_df = df_res[df_res['Status'].str.contains("🔴|🟡")][
        ['UDI', 'Region', 'Status', 'Failure Prob (%)', 'Primary Root Cause', 'Est. Downtime Risk Cost']
    ].head(15)

    if not critical_df.empty:
        data = [["UDI", "Region", "Status", "Prob (%)", "Root Cause", "Risk Cost"]]
        for _, r in critical_df.iterrows():
            data.append([
                str(r['UDI']), str(r['Region']), str(r['Status'][:10]),
                f"{r['Failure Prob (%)']}%", str(r['Primary Root Cause'])[:30], str(r['Est. Downtime Risk Cost'])
            ])

        t = Table(data, colWidths=[40, 50, 80, 60, 180, 120])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("All machines are operating within optimal safe thresholds.", normal_style))

    doc.build(story)
    buffer.seek(0)
    return buffer


def run_full_enterprise_pipeline(df_input: pd.DataFrame, models: dict, iso_forest) -> pd.DataFrame:
    """Execute full diagnostic pipeline with risk assessment, physics checks, and energy metrics."""
    thresholds = {'H': 0.35, 'M': 0.15, 'L': 0.41}

    df_clean = df_input.copy()
    df_clean.columns = [c.replace('[', '_').replace(']', '').strip() for c in df_clean.columns]

    df_clean['Temp_Diff'] = df_clean['Process temperature _K'] - df_clean['Air temperature _K']
    df_clean['Power'] = df_clean['Rotational speed _rpm'] * df_clean['Torque _Nm']
    df_clean['Power_kW'] = ((df_clean['Rotational speed _rpm'] * 2 * np.pi / 60) * df_clean['Torque _Nm']) / 1000.0
    df_clean['OSF_Metric'] = df_clean['Tool wear _min'] * df_clean['Torque _Nm']

    feature_cols = [
        'Air temperature _K', 'Process temperature _K',
        'Rotational speed _rpm', 'Torque _Nm', 'Tool wear _min',
        'Temp_Diff', 'Power', 'OSF_Metric'
    ]

    anomaly_preds = iso_forest.predict(df_clean[feature_cols])

    results = []
    for idx, row in df_clean.iterrows():
        region = row['Type']
        features_vec = row[feature_cols].values.reshape(1, -1)

        prob = models[region].predict_proba(features_vec)[0, 1]
        th = thresholds.get(region, 0.35)

        osf_limit = 12000 if region == 'M' else 11000
        physics_hazard = (row['OSF_Metric'] >= osf_limit) or (row['Temp_Diff'] < 8.6)
        is_anomaly = anomaly_preds[idx] == -1

        causes = []
        if row['Tool wear _min'] >= 200:
            causes.append("🪚 Tool wear critical (>200 min)")
        if row['Temp_Diff'] < 8.6:
            causes.append("🌡️ Heat dissipation issue (<8.6 K)")
        if row['OSF_Metric'] >= osf_limit:
            causes.append(f"⚡ Overstrain OSF Hazard ({int(row['OSF_Metric'])})")
        if row['Torque _Nm'] > 60:
            causes.append("⚙️ Excessive Torque (>60 Nm)")

        p_kw = round(row['Power_kW'], 2)
        energy_advice = f"💡 Power: {p_kw} kW. " + ("Reduce torque to optimize energy." if row['Torque _Nm'] > 50 else "Balanced operation.")

        if prob >= th or physics_hazard:
            status = "🔴 Danger"
            action = "Immediate shutdown and emergency maintenance."
            est_cost = "$3,000 (Unplanned Downtime)"
            roi_saving = "$3,500 (Preventive Saving)"
            if not causes:
                causes.append("🎲 High Model Failure Probability")
        elif prob >= (th * 0.6) or row['Tool wear _min'] >= 170:
            status = "🟡 Warning"
            action = "Inspect components during scheduled maintenance."
            est_cost = "$200 (Scheduled Service)"
            roi_saving = "$2,200"
            if not causes:
                causes.append("⚠️ Progressive wear trend")
        else:
            status = "🟢 Normal"
            action = "Optimal operating conditions."
            causes = ["✅ Normal Operating Parameters"]
            est_cost = "$0"
            roi_saving = "$0"

        results.append({
            'UDI': row.get('UDI', idx),
            'Product ID': row.get('Product ID', 'N/A'),
            'Region': region,
            'Status': status,
            'Sensor Anomaly': "⚠️ Anomaly Detected" if is_anomaly else "🟢 Normal",
            'Failure Prob (%)': round(prob * 100, 2),
            'Power (kW)': p_kw,
            'Primary Root Cause': " | ".join(causes),
            'Energy Advice': energy_advice,
            'Est. Downtime Risk Cost': est_cost,
            'Est. Savings via AI': roi_saving,
            'Action Recommended': action,
            'Torque _Nm': row['Torque _Nm'],
            'Rotational speed _rpm': row['Rotational speed _rpm'],
            'Tool wear _min': row['Tool wear _min']
        })

    return pd.DataFrame(results)
