import subprocess
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline import (
    load_all_artifacts,
    run_full_enterprise_pipeline,
    send_telegram_alert,
    generate_pdf_report
)

st.set_page_config(page_title="Industrial AI - Enterprise Platform", layout="wide", page_icon="⚙️")

st.title("⚙️️ Industrial AI Engine: Enterprise Predictive Maintenance Platform")
st.markdown("Integrated Engineering System for Predictive Maintenance, Data Quality, Risk Management, and Real-time Alerting")


# =========================================================
# 1. Resource Caching for Models
# =========================================================
@st.cache_resource
def get_cached_models():
    return load_all_artifacts()


try:
    models, iso_forest = get_cached_models()
    st.sidebar.success("✅ Models Loaded Successfully (XGBoost + Isolation Forest)!")
except Exception as e:
    st.sidebar.error("⚠️ Models not found! Please run `python train.py` first.")


# =========================================================
# 2. Data Caching for Pipeline Computation
# =========================================================
@st.cache_data
def cached_pipeline_execution(df_input: pd.DataFrame) -> pd.DataFrame:
    return run_full_enterprise_pipeline(df_input, models, iso_forest)


# =========================================================
# Main UI Layout & Tabs
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Operational Report",
    "📊 Analytics Dashboard",
    "🧪 Machine Simulator",
    "📱 Real-time Alerts",
    "🧪 Automated Unit Tests"
])

uploaded_file = st.sidebar.file_uploader("Upload New Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    res_df = cached_pipeline_execution(new_data)

    # ----------------------------------------------------
    # TAB 1: Operational Report
    # ----------------------------------------------------
    with tab1:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Fleet", len(res_df))
        c2.metric("🟢 Normal", len(res_df[res_df['Status'].str.contains("🟢")]))
        c3.metric("🟡 Warning", len(res_df[res_df['Status'].str.contains("🟡")]))
        c4.metric("🔴 Danger", len(res_df[res_df['Status'].str.contains("🔴")]))
        c5.metric("⚠️ Sensor Anomalies", len(res_df[res_df['Sensor Anomaly'] == "⚠️ Anomaly Detected"]))

        st.markdown("---")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            csv_data = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full CSV Report", data=csv_data, file_name="industrial_report.csv", mime="text/csv")
        with col_btn2:
            pdf_buf = generate_pdf_report(res_df)
            st.download_button("📄 Download Executive PDF Report", data=pdf_buf, file_name="Executive_Maintenance_Report.pdf", mime="application/pdf")

        cols_to_show = [
            'UDI', 'Region', 'Status', 'Sensor Anomaly', 'Failure Prob (%)', 'Power (kW)',
            'Primary Root Cause', 'Est. Downtime Risk Cost', 'Action Recommended'
        ]
        st.subheader("📋 Machine Fleet Status & Diagnostics")
        st.dataframe(res_df[cols_to_show], use_container_width=True)

    # ----------------------------------------------------
    # TAB 2: Analytics Dashboard
    # ----------------------------------------------------
    with tab2:
        st.subheader("📊 Risk Distribution & Sensor Quality Analytics")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig1 = px.pie(
                res_df, names='Status', title='Fleet Risk Distribution', color='Status',
                color_discrete_map={'🟢 Normal': '#2ecc71', '🟡 Warning': '#f1c40f', '🔴 Danger': '#e74c3c'}
            )
            st.plotly_chart(fig1, use_container_width=True)
        with col_g2:
            fig2 = px.scatter(
                res_df, x='Rotational speed _rpm', y='Torque _Nm', color='Status', size='Power (kW)',
                hover_data=['UDI', 'Sensor Anomaly'], title='Speed vs Torque & Anomaly Mapping'
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------------------------------
    # TAB 3: Machine Simulator
    # ----------------------------------------------------
    with tab3:
        st.subheader("🧪 Real-Time Machine Parameter Simulation")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            sim_region = st.selectbox("Type Region:", ["H", "M", "L"])
            sim_air = st.slider("Air Temp (K):", 290.0, 310.0, 300.0)
            sim_proc = st.slider("Process Temp (K):", 300.0, 320.0, 310.0)
        with sc2:
            sim_rpm = st.slider("Rotational Speed (RPM):", 1100, 2900, 1500)
            sim_torque = st.slider("Torque (Nm):", 3.0, 80.0, 40.0)
        with sc3:
            sim_wear = st.slider("Tool Wear (min):", 0, 260, 100)

        sim_tdiff = sim_proc - sim_air
        sim_power = sim_rpm * sim_torque
        sim_osf = sim_wear * sim_torque

        sim_vec = np.array([[sim_air, sim_proc, sim_rpm, sim_torque, sim_wear, sim_tdiff, sim_power, sim_osf]])
        sim_prob = models[sim_region].predict_proba(sim_vec)[0, 1]
        sim_anom = iso_forest.predict(sim_vec)[0] == -1

        st.markdown("---")
        st.metric("Model Failure Probability", f"{round(sim_prob*100, 2)}%")
        if sim_anom:
            st.warning("⚠️ Warning: Input parameters exhibit anomalous sensor behavior!")

        if sim_prob >= 0.35 or sim_osf >= 11000:
            st.error("🔴 Status: Critical Danger Risk")
        else:
            st.success("🟢 Status: Normal Operational Condition")

    # ----------------------------------------------------
    # TAB 4: Real-time Alerts
    # ----------------------------------------------------
    with tab4:
        st.subheader("📱 Automated Telegram Alert Dispatcher")
        st.info("Configure your bot credentials to receive instant push notifications for machines at risk.")

        bot_token = st.text_input("Telegram Bot Token:", type="password")
        chat_id = st.text_input("Telegram Chat ID:")

        danger_machines = res_df[res_df['Status'].str.contains("🔴")]

        if st.button("🚀 Dispatch Risk Alert Report to Telegram"):
            if danger_machines.empty:
                st.success("✅ No critical machines detected at this time.")
            else:
                msg = f"🚨 *CRITICAL ALERT - Predictive Maintenance Engine*\n\nDetected ({len(danger_machines)}) machine(s) in critical danger state 🔴!\n"
                for idx, row in danger_machines.head(5).iterrows():
                    msg += f"\n- *UDI {row['UDI']}* (Region {row['Region']}): Prob {row['Failure Prob (%)']}% | Root Cause: {row['Primary Root Cause']}"

                success, resp_msg = send_telegram_alert(bot_token, chat_id, msg)
                if success:
                    st.success(resp_msg)
                else:
                    st.error(resp_msg)

    # ----------------------------------------------------
    # TAB 5: Automated Unit Tests
    # ----------------------------------------------------
    with tab5:
        st.subheader("🧪 MLOps Quality Assurance: Automated Unit Testing")
        st.markdown("Execute automated unit testing suites to verify mathematical integrity and pipeline logic.")

        if st.button("▶️ Run Unit Test Suite (pytest)"):
            res = subprocess.run(["pytest", "test_pipeline.py"], capture_output=True, text=True)
            st.code(res.stdout if res.stdout else res.stderr)
            if res.returncode == 0:
                st.success("✅ All Unit Tests Passed Successfully (100% Pass Rate)!")

else:
    st.info("👈 Please upload a CSV dataset using the sidebar to activate analytics and diagnostic features.")
