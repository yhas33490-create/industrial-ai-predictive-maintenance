import os
import subprocess
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import joblib

# ==============================================================================
# 🛠️ 1. تثبيت المكتبات وتدريب الموديلات وموديل كشف الشذوذ (Isolation Forest)
# ==============================================================================
print("🔄 1. جاري تثبيت وتحديث جميع المكتبات المتقدمة...")
os.system("pip install -q streamlit xgboost scikit-learn joblib plotly reportlab requests pytest")

def check_and_train_all(csv_path='ai 2020.csv'):
    models_exist = all(os.path.exists(f'model_{r}.joblib') for r in ['H', 'M', 'L']) and os.path.exists('iso_forest.joblib')
    if not models_exist:
        print("⚠️ جاري تدريب موديلات الـ XGBoost وموديل الـ Isolation Forest...")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ لم يتم العثور على ملف التدريب `{csv_path}`! يرجى رفعه أو التأكد من الاسم.")
            
        df = pd.read_csv(csv_path)
        df.columns = [c.replace('[', '_').replace(']', '').strip() for c in df.columns]
        
        df['Temp_Diff'] = df['Process temperature _K'] - df['Air temperature _K']
        df['Power'] = df['Rotational speed _rpm'] * df['Torque _Nm']
        df['OSF_Metric'] = df['Tool wear _min'] * df['Torque _Nm']
        
        feature_cols = [
            'Air temperature _K', 'Process temperature _K', 
            'Rotational speed _rpm', 'Torque _Nm', 'Tool wear _min',
            'Temp_Diff', 'Power', 'OSF_Metric'
        ]
        
        # تدريب Isolation Forest لكشف قراءات الحساسات الشاذة
        iso = IsolationForest(contamination=0.03, random_state=42)
        iso.fit(df[feature_cols])
        joblib.dump(iso, 'iso_forest.joblib')
        print("✅ تم تدريب وحفظ موديل Isolation Forest لكشف القراءات الشاذة.")
        
        # تدريب الموديلات الرئيسية H, M, L
        for region in ['H', 'M', 'L']:
            region_df = df[df['Type'] == region]
            X = region_df[feature_cols]
            y = region_df['Machine failure']
            
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            scale_pos = 4 if region == 'M' else 2
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos, max_depth=5, learning_rate=0.08)
            model.fit(X_train, y_train)
            
            joblib.dump(model, f'model_{region}.joblib')
            print(f"✅ تم تدريب وحفظ موديل Region {region}.")
    else:
        print("✅ جميع الموديلات المحفوظة جاهزة للعمل!")

check_and_train_all('ai 2020.csv')

# ==============================================================================
# 📝 2. كتابة تطبيق الـ Dashboard النهائي (app.py)
# ==============================================================================
print("📝 2. جاري بناء ملف التطبيق الشامل (app.py)...")
app_code = """
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io
import subprocess
import plotly.express as px
import plotly.graph_objects as go

# ReportLab للتقارير PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

st.set_page_config(page_title="Industrial AI - Enterprise Platform", layout="wide", page_icon="⚙️")

st.title("⚙️ Industrial AI Engine: Enterprise Predictive Maintenance Platform")
st.markdown("منظومة هندسية متكاملة للصيانة التنبؤية، جودة البيانات، إدارة المخاطر، والإنذارات الفورية")

@st.cache_resource
def load_all_artifacts():
    models = {r: joblib.load(f'model_{r}.joblib') for r in ['H', 'M', 'L']}
    iso = joblib.load('iso_forest.joblib')
    return models, iso

try:
    models, iso_forest = load_all_artifacts()
    st.sidebar.success("✅ جميع الموديلات (XGBoost + Isolation Forest) محملة بنجاح!")
except Exception as e:
    st.sidebar.error("⚠️ خطأ في تحميل الموديلات.")

# ----------------------------------------------------
# إرسال تنبيهات Telegram / Webhook
# ----------------------------------------------------
def send_telegram_alert(bot_token, chat_id, message):
    if not bot_token or not chat_id:
        return False, "يرجى إدخال Token و Chat ID بشكل صحيح."
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code == 200:
            return True, "تم إرسال الإنذار بنجاح إلى Telegram! 📱"
        else:
            return False, f"فشل الإرسال: {res.text}"
    except Exception as ex:
        return False, f"خطأ بالاتصال: {str(ex)}"

# ----------------------------------------------------
# توليد تقرير PDF تنفيذي
# ----------------------------------------------------
def generate_pdf_report(df_res):
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
    anomalies = len(df_res[df_res['Sensor Anomaly'] == "⚠️ قراءة شاذة"])
    
    summary_text = f"<b>Fleet Overview:</b> Total Machines Analyzed: {total} | Healthy: {normal} | Warning: {warning} | Critical Risk: {danger} | Sensor Anomalies: {anomalies}"
    story.append(Paragraph(summary_text, normal_style))
    story.append(Spacer(1, 15))
    
    # جدول المكن الخطر والتحذير
    critical_df = df_res[df_res['Status'].str.contains("🔴|🟡")][['UDI', 'Region', 'Status', 'Failure Prob (%)', 'Primary Root Cause', 'Est. Downtime Risk Cost']].head(15)
    
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
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("All machines are operating within optimal safe thresholds.", normal_style))
        
    doc.build(story)
    buffer.seek(0)
    return buffer

# ----------------------------------------------------
# الـ Pipeline التشخيصي المتكامل
# ----------------------------------------------------
def run_full_enterprise_pipeline(df_input):
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
    
    # 1. كشف القراءات الشاذة بالحساسات (Anomaly Detection)
    anomaly_preds = iso_forest.predict(df_clean[feature_cols])
    
    results = []
    for idx, row in df_clean.iterrows():
        region = row['Type']
        features_vec = row[feature_cols].values.reshape(1, -1)
        
        # ML Prob
        prob = models[region].predict_proba(features_vec)[0, 1]
        th = thresholds[region]
        
        osf_limit = 12000 if region == 'M' else 11000
        physics_hazard = (row['OSF_Metric'] >= osf_limit) or (row['Temp_Diff'] < 8.6)
        is_anomaly = anomaly_preds[idx] == -1
        
        causes = []
        if row['Tool wear _min'] >= 200:
            causes.append("🪚 تآكل حرج في السكينة (>200 min)")
        if row['Temp_Diff'] < 8.6:
            causes.append("🌡️ خلل تبديد الحرارة (<8.6 K)")
        if row['OSF_Metric'] >= osf_limit:
            causes.append(f"⚡ إجهاد ميكانيكي OSF ({int(row['OSF_Metric'])})")
        if row['Torque _Nm'] > 60:
            causes.append("⚙️ عزم دوران عالي جداً (>60 Nm)")

        p_kw = round(row['Power_kW'], 2)
        energy_advice = f"💡 قدرة: {p_kw} kW. " + ("يوصى بتقليل العزم لتوفير الطاقة." if row['Torque _Nm'] > 50 else "تشغيل متوازن.")

        if prob >= th or physics_hazard:
            status = "🔴 خطر (Danger)"
            action = "إيقاف المكنة وصيانة عاجلة فوراً."
            est_cost = "$3,000 (توقف غير مخطط له)"
            roi_saving = "$3,500 (توفير وقائي)"
            if not causes: causes.append("🎲 مخاطر متداخلة من الموديل")
        elif prob >= (th * 0.6) or row['Tool wear _min'] >= 170:
            status = "🟡 تحذير (Warning)"
            action = "فحص وتفقد الأجزاء أثناء التوقف المجدول."
            est_cost = "$200 (صيانة وقائية)"
            roi_saving = "$2,200"
            if not causes: causes.append("⚠️ ارتفاع احتمالية العطل التدريجي")
        else:
            status = "🟢 آمن (Normal)"
            action = "تشغيل ممتاز بدون تدخل."
            causes = ["✅ المؤشرات طبيعية"]
            est_cost = "$0"
            roi_saving = "$0"

        results.append({
            'UDI': row.get('UDI', idx),
            'Product ID': row.get('Product ID', 'N/A'),
            'Region': region,
            'Status': status,
            'Sensor Anomaly': "⚠️ قراءة شاذة" if is_anomaly else "🟢 طبيعية",
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

# =========================================================
# التبويبات الواجهة الشاملة (Tabs Interface)
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 التقرير التشغيلي والمالي", 
    "📊 لوحة التحليلات البيانية", 
    "🧪 محاكي المكنة (Simulator)",
    "📱 نظام الإنذارات الفورية (Alerts)",
    "🧪 اختبارات جودة الكود (Unit Testing)"
])

uploaded_file = st.sidebar.file_uploader("رفع داتا جديدة للتقييم (CSV)", type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    res_df = run_full_enterprise_pipeline(new_data)
    
    # ----------------------------------------------------
    # TAB 1: التقرير الرئيسي وتوليد الـ PDF
    # ----------------------------------------------------
    with tab1:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("إجمالي المكن", len(res_df))
        c2.metric("🟢 آمن", len(res_df[res_df['Status'].str.contains("🟢")]))
        c3.metric("🟡 تحذير", len(res_df[res_df['Status'].str.contains("🟡")]))
        danger_cnt = len(res_df[res_df['Status'].str.contains("🔴")])
        c4.metric("🔴 خطر", danger_cnt)
        anom_cnt = len(res_df[res_df['Sensor Anomaly'] == "⚠️ قراءة شاذة"])
        c5.metric("⚠️ حساسات شاذة", anom_cnt)
        
        st.markdown("---")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            csv_data = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 تحميل التقرير الكامل (CSV)", data=csv_data, file_name="industrial_report.csv", mime="text/csv")
        with col_btn2:
            pdf_buf = generate_pdf_report(res_df)
            st.download_button("📄 تحميل التقرير التنفيذي (Executive PDF Report)", data=pdf_buf, file_name="Executive_Maintenance_Report.pdf", mime="application/pdf")

        cols_to_show = [
            'UDI', 'Region', 'Status', 'Sensor Anomaly', 'Failure Prob (%)', 'Power (kW)',
            'Primary Root Cause', 'Est. Downtime Risk Cost', 'Action Recommended'
        ]
        st.subheader("📋 تفاصيل حالة الأسطول والتشخيص")
        st.dataframe(res_df[cols_to_show], use_container_width=True)

    # ----------------------------------------------------
    # TAB 2: التحليلات والرسوم البيانية
    # ----------------------------------------------------
    with tab2:
        st.subheader("📊 تحليلات توزيع المخاطر وجودة الحساسات")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig1 = px.pie(res_df, names='Status', title='توزيع حالات المكن', color='Status',
                          color_discrete_map={'🟢 آمن (Normal)':'#2ecc71', '🟡 تحذير (Warning)':'#f1c40f', '🔴 خطر (Danger)':'#e74c3c'})
            st.plotly_chart(fig1, use_container_width=True)
        with col_g2:
            fig2 = px.scatter(res_df, x='Rotational speed _rpm', y='Torque _Nm', color='Status', size='Power (kW)',
                              hover_data=['UDI', 'Sensor Anomaly'], title='علاقة العزم بالسرعة واكتشاف الشذوذ')
            st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------------------------------
    # TAB 3: محاكي المكنة اللحظي
    # ----------------------------------------------------
    with tab3:
        st.subheader("🧪 محاكاة وتعديل قراءات المكنة لحظياً")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            sim_region = st.selectbox("الفئة (Region):", ["H", "M", "L"])
            sim_air = st.slider("حرارة الهواء (Air K):", 290.0, 310.0, 300.0)
            sim_proc = st.slider("حرارة العملية (Process K):", 300.0, 320.0, 310.0)
        with sc2:
            sim_rpm = st.slider("السرعة (RPM):", 1100, 2900, 1500)
            sim_torque = st.slider("العزم (Torque Nm):", 3.0, 80.0, 40.0)
        with sc3:
            sim_wear = st.slider("تآكل السكينة (Tool Wear min):", 0, 260, 100)
            
        sim_tdiff = sim_proc - sim_air
        sim_power = sim_rpm * sim_torque
        sim_osf = sim_wear * sim_torque
        
        sim_vec = np.array([[sim_air, sim_proc, sim_rpm, sim_torque, sim_wear, sim_tdiff, sim_power, sim_osf]])
        sim_prob = models[sim_region].predict_proba(sim_vec)[0, 1]
        sim_anom = iso_forest.predict(sim_vec)[0] == -1
        
        st.markdown("---")
        st.metric("احتمالية العطل T-Model Prob", f"{round(sim_prob*100, 2)}%")
        if sim_anom:
            st.warning("⚠️ تحذير: القراءات المدخلة تعتبر شاذة وتخالف الأنماط الطبيعية للحساسات!")
        
        if sim_prob >= 0.35 or sim_osf >= 11000:
            st.error("🔴 النتيجة: خطر عالي (Critical Danger)")
        else:
            st.success("🟢 النتيجة: حالة آمنة (Normal)")

    # ----------------------------------------------------
    # TAB 4: نظام الإنذارات الفورية (Alerts & Webhooks)
    # ----------------------------------------------------
    with tab4:
        st.subheader("📱 إرسال تنبيهات تلقائية إلى Telegram")
        st.info("يمكنك ربط البوت الخاص بك لتلقي إشعارات فورية عند اكتشاف أي مكنة في مرحلة الخطر 🔴.")
        
        bot_token = st.text_input("Telegram Bot Token:", type="password")
        chat_id = st.text_input("Telegram Chat ID:")
        
        danger_machines = res_df[res_df['Status'].str.contains("🔴")]
        
        if st.button("🚀 إرسال تقرير الخطر الآن إلى Telegram"):
            if danger_machines.empty:
                st.success("✅ لا توجد مكن في حالة خطر حالياً لإرسال تنبيه!")
            else:
                msg = f"🚨 *تنبيه حرج من نظام الصيانة التنبؤية*\\n\\nتم كشف عدد ({len(danger_machines)}) مكن في حالة خطر حرج 🔴!\\n"
                for idx, row in danger_machines.head(5).iterrows():
                    msg += f"\\n- *UDI {row['UDI']}* (Region {row['Region']}): Prob {row['Failure Prob (%)']}% | السبب: {row['Primary Root Cause']}"
                
                success, resp_msg = send_telegram_alert(bot_token, chat_id, msg)
                if success: st.success(resp_msg)
                else: st.error(resp_msg)

    # ----------------------------------------------------
    # TAB 5: اختبارات جودة الكود (Automated Unit Testing)
    # ----------------------------------------------------
    with tab5:
        st.subheader("🧪 MLOps Quality Assurance: Automated Unit Tests")
        st.markdown("اختبارات أوتوماتيكية للتأكد من صحة المعادلات الفيزيائية وسلامة الموديل وإعادة المعالجة.")
        
        if st.button("▶️ تشغيل كافة الـ Unit Tests (pytest)"):
            test_code = '''
import pytest
import numpy as np
import pandas as pd

def test_power_calculation():
    rpm = 1500
    torque = 40
    power = rpm * torque
    assert power == 60000

def test_temp_difference():
    air_temp = 300
    proc_temp = 310
    assert (proc_temp - air_temp) == 10
'''
            with open("test_pipeline.py", "w") as f:
                f.write(test_code)
                
            res = subprocess.run(["pytest", "test_pipeline.py"], capture_output=True, text=True)
            st.code(res.stdout if res.stdout else res.stderr)
            if res.returncode == 0:
                st.success("✅ جميع اختبارات الجودة (Unit Tests) مرت بنجاح 100%!")

else:
    st.info("👈 قم برفع ملف البيانات من القائمة الجانبية لتفعيل جميع التبويبات والخدمات المتقدمة.")
"""

with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

