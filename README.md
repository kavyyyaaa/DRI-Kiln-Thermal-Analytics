# DRI Kiln Thermal Analytics & AI Forecasting System

An industrial-grade monitoring and predictive maintenance ecosystem designed for Direct Reduced Iron (DRI) rotary kilns. This project visualizes real-time shell temperatures via an interactive 3D thermal mesh and leverages deep learning to forecast future thermal deviations, hotspots, and accretion risks.

---

## 🚀 System Architecture Overview

The application is split into two specialized components that communicate over a secure server-to-server proxy loop, eliminating browser CORS limitations and keeping the deep learning models highly accessible yet secure:

1. **Dashboard UI & Backend Proxy (Flask + Plotly):** Processes the industrial SQLite data pipeline, calculates 90-day zonal regression metrics, and renders the real-time 3D rotary kiln cylinder map.
2. **Deep Learning Prediction Engine (FastAPI + LSTM):** An independent neural network microservice that ingests thermal time-series sequences to output multi-step operational forecasts.

---

## ✨ Key Features

* **Live 3D Thermal Mapping:** Renders a high-fidelity 250x1200 grid mapped to the physical dimensions of a 70-meter industrial kiln using an advanced Plotly engine.
* **LSTM-Powered Predictive Analytics:** Multi-step forecasts predicting future shell temperatures 10 days out with an automated data segmentation split line.
* **Dynamic Operator Diagnostics:** Real-time risk index generation, true temperature deviation math from corporate baselines, and operator-aligned action items (Normal: 225°C - 275°C, Warning: >325°C, Critical: >400°C).
* **Advanced Anomaly & Accretion Detection:** Employs Scikit-Learn (K-Means Clustering and Isolation Forests) to track physical accretion formation and refractory lining degradation over 90 days.
* **Automated Executive Reporting:** One-click streaming export of spatial data points into structured Excel worksheets with pre-computed linear regression thermal slopes.

---

## 🛠️ Tech Stack

* **Frontend:** HTML5, CSS3 (Tailwind-inspired industrial dashboard), Plotly.js (3D Surface Rendering)
* **Backend Frameworks:** Flask (Web application gateway), FastAPI (High-performance AI inference engine)
* **Deep Learning & ML:** TensorFlow/Keras (LSTM architecture), Scikit-Learn (Anomaly detection, KMeans clustering)
* **Data Processing:** NumPy, Pandas, OpenPyXL, SQLite3 (WAL journal mode)
* **Infrastructure Tunnels:** Ngrok Edge Secure Layer

---

---

### 💡 Markdown mein Screenshot Kaise Add Karein?
Apne dashboard ka screenshot upload karne ke liye aapko bas upar diye gaye README code ke **Key Features** section ke theek niche ye line insert karni hogi:

```markdown
## Dashboard Interface Preview

![DRI Kiln Dashboard]
<img width="1587" height="855" alt="image" src="https://github.com/user-attachments/assets/4889611e-aae8-4bc4-88ee-796efdac2de6" />


## 📁 Repository Structure

```text
├── dashboard_code/
│   ├── dri_kiln_thermal_analytics_routes.py  # Flask routing proxy and database manager
│   └── interactive-dashboard.html            # Core HTML5 dashboard view with 3D Plotly canvas
├── model/
│   └── lstm_patch.h5                         # Pre-trained core LSTM weights
├── scaler/
│   └── scaler.pkl                            # Scikit-Learn data scaling configurations
├── predict_patch_api.py                      # FastAPI endpoint configuration
├── structure.py                              # Core infrastructure schema definitions
├── test_predict.py                           # Local execution test scripts
├── train_lstm_patch.py                       # LSTM training configurations and dataset definitions
└── .gitignore                                # Prevents local caching and venv conflicts

---


