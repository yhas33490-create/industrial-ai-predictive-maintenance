# ⚙️ Industrial AI & Physics-Informed Predictive Maintenance Suite
🚀 **Live Demo:** [Click here to launch the interactive app](https://industrial-ai-suite.streamlit.app/)

An enterprise-grade, domain-driven MLOps predictive maintenance solution designed to monitor equipment health, apply physics-informed feature engineering, detect telemetry anomalies using unsupervised learning, and deliver real-time operational intelligence.

---

## 💡 Core Capabilities & Services
- **Physics-Informed Feature Synthesis:** Integrates fundamental mechanical & thermal equations (Power in $kW$, Temperature Differentials, Mechanical Stress Strain) directly into model training to improve failure detection accuracy.
- **Multi-Region Predictive Models:** Region-specific XGBoost classification pipelines tailored to diverse operational environments (H, M, L tiers).
- **Telemetry Anomaly Detection:** Integrated unsupervised **Isolation Forest** to identify corrupted sensor readings and extreme telemetry outliers prior to model inference.
- **Root Cause Analysis (RCA) & Financial Analytics:** Automated diagnostic engine explaining primary failure drivers along with financial risk cost vs. AI savings estimates ($2K–$4K per machine).
- **Interactive Streamlit Dashboard:** User-friendly interface allowing operators to upload fresh telemetry datasets, run instant batch predictions, simulate parameters, and generate executive PDF reports.
- **Automated QA Suite:** Integrated `pytest` unit tests ensuring code reliability and precision.

---

## 🛠️ Tech Stack & Architecture
- **Language:** Python
- **Machine Learning:** Scikit-Learn, XGBoost, Isolation Forest
- **Data Science & Visualization:** Pandas, NumPy, Plotly
- **Application & Utilities:** Streamlit, ReportLab (PDF Generation), Requests (Telegram Integration), Pytest

---

## 📂 Repository Structure
```text
├── app.py              # Interactive Streamlit UI & Multi-Tab Operational Dashboard
├── pipeline.py         # Core Physics Calculations & ML Inference Engine
├── test_pipeline.py    # Unit tests for domain equations and data transformations
├── requirements.txt    # Project dependencies
└── README.md           # Project Documentation & Overview
