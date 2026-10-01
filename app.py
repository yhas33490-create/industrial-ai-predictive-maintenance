import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pipeline import run_full_enterprise_pipeline

st.set_page_config(page_title="Industrial AI Enterprise Engine", page_icon="⚙️", layout="wide")

st.title("⚙️ Industrial AI Engine: Physics-Informed Predictive Maintenance")
st.markdown("### Operational Intelligence, Financial ROI & Anomaly Detection Platform")

# Sidebar - Configuration
st.sidebar.header("🕹️ Operational Controls")
st.sidebar.info("Upload fleet sensor telemetry CSV to trigger the Physics-Informed ML Inference Engine.")

uploaded_file = st.sidebar.file_uploader("Upload Telemetry Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    
    # Run Physics & ML Pipeline
    with st.spinner("Processing Physics-Informed Feature Synthesis & Model Inference..."):
        df_results = run_full_enterprise_pipeline(df_raw, models=None, iso_forest=None)
    
    # Navigation Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Fleet Overview & ROI", 
        "📈 Telemetry & Physics Analysis", 
        "🎛️ Interactive Parameter Simulator",
        "🚨 Anomaly & Risk Alerts",
        "📄 Executive Report Generation"
    ])
    
    # TAB 1: Financial & Fleet Overview
    with tab1:
        st.subheader("💡 Financial Impact & Operational Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        total_machines = len(df_results)
        danger_count = len(df_results[df_results['Status'].str.contains("Danger")])
        warning_count = len(df_results[df_results['Status'].str.contains("Warning")])
        normal_count = len(df_results[df_results['Status'].str.contains("Normal")])
        
        col1.metric("Total Equipment Fleet", f"{total_machines} Units")
        col2.metric("Critical Hazards", f"{danger_count}", delta_color="inverse")
        col3.metric("Maintenance Warnings", f"{warning_count}", delta_color="off")
        col4.metric("Optimal Condition", f"{normal_count}")
        
        st.markdown("---")
        st.write("### 📋 Predictive Maintenance Fleet Summary")
        st.dataframe(df_results, use_container_width=True)
        
    # TAB 2: Telemetry Plots
    with tab2:
        st.subheader("📈 Sensor Telemetry & Physical Distributions")
        col_a, col_b = st.columns(2)
        
        if 'Torque _Nm' in df_results.columns and 'Rotational speed _rpm' in df_results.columns:
            fig_scatter = px.scatter(
                df_results, x='Rotational speed _rpm', y='Torque _Nm', 
                color='Status', size='Failure Prob (%)',
                title="Torque vs. Rotational Speed by Failure Risk",
                hover_data=['UDI', 'Primary Root Cause']
            )
            col_a.plotly_chart(fig_scatter, use_container_width=True)
            
        fig_prob = px.histogram(
            df_results, x='Failure Prob (%)', color='Status',
            title="Failure Probability Distribution across Fleet", barmode="overlay"
        )
        col_b.plotly_chart(fig_prob, use_container_width=True)

    # TAB 3: Interactive Simulator
    with tab3:
        st.subheader("🎛️ Real-Time Machine Operating Condition Simulator")
        st.markdown("Simulate sensor telemetry input to predict failure modes live.")
        
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        sim_temp = sim_col1.slider("Air Temp (K)", 290.0, 310.0, 300.0)
        sim_p_temp = sim_col1.slider("Process Temp (K)", 295.0, 320.0, 310.0)
        sim_rpm = sim_col2.slider("Rotational Speed (RPM)", 1100, 2900, 1500)
        sim_torque = sim_col2.slider("Torque (Nm)", 3.0, 80.0, 40.0)
        sim_wear = sim_col3.slider("Tool Wear (min)", 0, 260, 120)
        
        p_diff = sim_p_temp - sim_temp
        p_kw = ((sim_rpm * 2 * np.pi / 60) * sim_torque) / 1000.0
        osf = sim_wear * sim_torque
        
        st.write("#### ⚡ Calculated Physics Parameters:")
        st.write(f"- **Temperature Delta ($\Delta T$):** {round(p_diff, 2)} K")
        st.write(f"- **Mechanical Power Output:** {round(p_kw, 2)} kW")
        st.write(f"- **Overstrain Factor (OSF):** {int(osf)}")

    # TAB 4: Anomaly & Risk Alerts
    with tab4:
        st.subheader("🚨 Detected Outliers & Telemetry Anomaly Stream")
        anomalies_df = df_results[df_results['Sensor Anomaly'].str.contains("Anomaly")]
        if not anomalies_df.empty:
            st.error(f"⚠️ {len(anomalies_df)} Sensor Telemetry Outliers / Corrupted Data Points Detected!")
            st.dataframe(anomalies_df[['UDI', 'Region', 'Status', 'Sensor Anomaly', 'Primary Root Cause']])
        else:
            st.success("🟢 No sensor anomalies detected in current batch.")

    # TAB 5: Report Generation
    with tab5:
        st.subheader("📄 Export Executive Diagnostic Briefing")
        st.markdown("Generate automated summaries for plant managers and operational directors.")
        st.download_button(
            label="📥 Download CSV Analysis Summary",
            data=df_results.to_csv(index=False).encode('utf-8'),
            file_name="Industrial_AI_Diagnostic_Report.csv",
            mime="text/csv"
        )

else:
    st.info("👋 Welcome to the Industrial AI Engine! Please upload a telemetry CSV file from the sidebar to activate all analysis tabs.")
    st.markdown("""
    #### 📌 Expected CSV Columns:
    `UDI`, `Product ID`, `Type`, `Air temperature _K`, `Process temperature _K`, `Rotational speed _rpm`, `Torque _Nm`, `Tool wear _min`
    """)
