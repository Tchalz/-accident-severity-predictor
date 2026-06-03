import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="🚦",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.95) !important;
    border-right: 1px solid rgba(249,202,36,0.2) !important;
}
[data-testid="stSidebar"] * { color: #dfe6e9 !important; }
[data-testid="stSidebarNav"] { display: none; }

.block-container { padding: 1.5rem 2rem 4rem 2rem; }

/* Header */
.header-wrap {
    text-align: center;
    padding: 1.5rem 0 1rem 0;
    border-bottom: 1px solid rgba(249,202,36,0.2);
    margin-bottom: 1.5rem;
}
.header-icon { font-size: 3rem; line-height: 1; display: block; margin-bottom: 0.3rem; }
.header-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 6px;
    background: linear-gradient(90deg, #f9ca24, #f0932b, #eb4d4b, #6ab04c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1;
}
.header-sub {
    font-family: 'DM Sans', sans-serif;
    color: #b2bec3;
    font-size: 0.95rem;
    margin-top: 0.4rem;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1;
    min-width: 140px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(249,202,36,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    letter-spacing: 2px;
    color: #f9ca24;
    display: block;
}
.metric-card .lbl {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    color: #b2bec3;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Section headers */
.section-head {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.3rem;
    letter-spacing: 3px;
    color: #f9ca24;
    border-bottom: 1px solid rgba(249,202,36,0.2);
    padding-bottom: 6px;
    margin: 1.5rem 0 1rem 0;
}

/* Labels */
label {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #f0e6ff !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(249,202,36,0.3) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: #f9ca24 !important;
}

/* Number input */
[data-testid="stNumberInput"] input {
    background: rgba(249,202,36,0.1) !important;
    border: 1.5px solid rgba(249,202,36,0.5) !important;
    border-radius: 8px !important;
    color: #f9ca24 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}
[data-testid="stNumberInput"] button {
    background: rgba(249,202,36,0.15) !important;
    border: 1px solid rgba(249,202,36,0.3) !important;
    color: #f9ca24 !important;
    border-radius: 6px !important;
}

/* Slider */
[data-testid="stSlider"] [role="slider"] {
    background: #ffffff !important;
    border: 3px solid #f9ca24 !important;
    box-shadow: 0 0 8px rgba(249,202,36,0.6) !important;
}
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
    color: #f9ca24 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
}

/* Predict button */
[data-testid="stButton"] > button {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.4rem !important;
    letter-spacing: 4px !important;
    background: linear-gradient(90deg, #eb4d4b, #f0932b, #f9ca24) !important;
    color: #0f0c29 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.8rem 2rem !important;
    width: 100% !important;
    margin-top: 1rem !important;
    box-shadow: 0 4px 24px rgba(235,77,75,0.4) !important;
    transition: transform 0.15s ease !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 36px rgba(235,77,75,0.6) !important;
}

/* Result alert */
[data-testid="stAlert"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    margin-top: 1rem !important;
}

/* Tabs */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #b2bec3 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #f9ca24 !important;
    border-bottom: 2px solid #f9ca24 !important;
}

/* Sidebar labels */
.sidebar-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 2px;
    color: #f9ca24;
    margin-bottom: 0.5rem;
    display: block;
}

hr { border-color: rgba(249,202,36,0.15) !important; }
[data-testid="column"] { padding: 0 0.4rem !important; }

/* Chart backgrounds */
.stPlotlyChart, [data-testid="stImage"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load('model.pkl')
    encoder  = joblib.load('encoder.pkl')
    selector = joblib.load('selector.pkl')
    lb       = joblib.load('label_encoder.pkl')
    features = joblib.load('features.pkl')
    return model, encoder, selector, lb, features

model, encoder, selector, lb, features = load_artifacts()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
  <span class="header-icon">🚦</span>
  <div class="header-title">Accident Severity Predictor</div>
  <div class="header-sub">ML-powered road accident injury severity classification · Random Forest · 95% Accuracy</div>
</div>
""", unsafe_allow_html=True)

# ── Model accuracy dashboard ───────────────────────────────────────────────────
st.markdown('<div class="metric-row">'
    '<div class="metric-card"><span class="val">95%</span><span class="lbl">Model Accuracy</span></div>'
    '<div class="metric-card"><span class="val">500</span><span class="lbl">Decision Trees</span></div>'
    '<div class="metric-card"><span class="val">17</span><span class="lbl">Input Features</span></div>'
    '<div class="metric-card"><span class="val">31K</span><span class="lbl">Training Samples</span></div>'
    '<div class="metric-card"><span class="val">3</span><span class="lbl">Severity Classes</span></div>'
    '<div class="metric-card"><span class="val">SMOTE</span><span class="lbl">Balancing Method</span></div>'
'</div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯  Predict Severity", "📊  Model Dashboard", "ℹ️  About"])

# ════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-head">🧑 Driver & Casualty Info</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        day        = st.selectbox('Day of week', ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        age_driver = st.selectbox('Age band of driver', ['18-30','31-50','Over 51','Under 18','Unknown'])
        sex_driver = st.selectbox('Sex of driver', ['Male','Female','Unknown'])
    with c2:
        education  = st.selectbox('Educational level', ['Above high school','High school','Elementary school','Junior high school','Illiterate','Writing & reading','Unknown'])
        experience = st.selectbox('Driving experience', ['1-2yr','2-5yr','5-10yr','Above 10yr','No Licence','Below 1yr','Unknown'])
        relation   = st.selectbox('Vehicle driver relation', ['Employee','Owner','Other','Unknown'])
    with c3:
        sex_casualty = st.selectbox('Sex of casualty', ['Male','Female','Unknown'])
        age_casualty = st.selectbox('Age band of casualty', ['18-30','31-50','Over 51','Under 18','Unknown'])
        casualties   = st.number_input('Number of casualties', min_value=1, max_value=10, value=1, step=1)

    st.markdown('<div class="section-head">🚗 Vehicle Info</div>', unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    with v1:
        vehicle_type = st.selectbox('Type of vehicle', ['Automobile','Public (> 45 seats)','Lorry (11-40Q)','Public (13-45 seats)','Lorry (40-100Q)','Long lorry','Taxi','Pickup flat bed','Motorcycle','Other','Unknown'])
    with v2:
        service_year = st.selectbox('Service year of vehicle', ['Above 10yr','5-10yr','2-5yr','1-2yr','Below 1yr','Unknown'])
    with v3:
        vehicles = st.number_input('Number of vehicles involved', min_value=1, max_value=10, value=1, step=1)

    st.markdown('<div class="section-head">🛣️ Road & Accident Info</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        area      = st.selectbox('Area accident occurred', ['Residential areas','Office areas','School areas','Church areas','Industrial areas','Other','Unknown'])
        junction  = st.selectbox('Type of junction', ['No junction','Y Shape','T Shape','Crossing','O Shape','Other','Unknown'])
    with r2:
        collision = st.selectbox('Type of collision', ['Vehicle with vehicle collision','Collision with roadside objects','Rollover','Fall from vehicle','Collision with animals','Other','Unknown'])
        cause     = st.selectbox('Cause of accident', ['Moving Backward','Overtaking','Changing lane to the right','Changing lane to the left','Driving carelessly','Driving at high speed','Driving to the left','No distancing','Other','Unknown'])
    with r3:
        hour = st.slider('Hour of day', 0, 23, 12)

    st.markdown("<br>", unsafe_allow_html=True)

    # Validation
    inputs_valid = True
    if vehicles > 8:
        st.warning("⚠️ Number of vehicles above 8 is unusual — please verify.")
    if casualties > vehicles * 5:
        st.warning("⚠️ Casualties seem high relative to vehicles — please verify.")

    if st.button('🔍  PREDICT SEVERITY', use_container_width=True):
        input_data = pd.DataFrame([[
            day, vehicles, casualties, area, junction,
            age_driver, sex_driver, education, relation,
            vehicle_type, experience, service_year,
            collision, sex_casualty, age_casualty, cause, hour
        ]], columns=features)

        encoded    = encoder.transform(input_data)
        selected   = selector.transform(encoded)
        proba      = model.predict_proba(selected)[0]
        confidence = np.max(proba) * 100

        # Threshold adjustment to correct for class imbalance
        # (84.5% Slight Injury in original data skews raw predictions)
        thresholds = {0: 0.05, 1: 0.20, 2: 0.60}
        if proba[0] >= thresholds[0]:
            label = 'Fatal injury'
        elif proba[1] >= thresholds[1]:
            label = 'Serious Injury'
        else:
            label = 'Slight Injury' 

        if label == 'Fatal injury':
            st.error(f'☠️ Predicted Severity: **{label}** — Immediate emergency response required.  \n📊 Model confidence: **{confidence:.1f}%**')
        elif label == 'Serious Injury':
            st.warning(f'🔶 Predicted Severity: **{label}** — Urgent medical attention needed.  \n📊 Model confidence: **{confidence:.1f}%**')
        else:
            st.success(f'✅ Predicted Severity: **{label}** — Minor injuries expected.  \n📊 Model confidence: **{confidence:.1f}%**')

        # Probability bar chart
        st.markdown('<div class="section-head">📊 Class Probabilities</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({'Severity': lb.classes_, 'Probability': proba * 100})
        fig, ax = plt.subplots(figsize=(7, 2.5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        colors = ['#eb4d4b', '#f0932b', '#6ab04c']
        bars = ax.barh(prob_df['Severity'], prob_df['Probability'], color=colors, height=0.5)
        for bar, val in zip(bars, prob_df['Probability']):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', color='white',
                    fontsize=10, fontweight='bold')
        ax.set_xlim(0, 115)
        ax.set_xlabel('Probability (%)', color='#b2bec3', fontsize=9)
        ax.tick_params(colors='#b2bec3', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('rgba(255,255,255,0.1)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ════════════════════════════════════════════════════════
with tab2:
    d1, d2 = st.columns(2)

    with d1:
        st.markdown('<div class="section-head">🌲 Feature Importance</div>', unsafe_allow_html=True)
        importances = model.feature_importances_
        feat_names  = selector.get_feature_names_out()
        fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=True).tail(15)

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        fig2.patch.set_facecolor('#1a1a2e')
        ax2.set_facecolor('#1a1a2e')
        colors_fi = plt.cm.YlOrRd(np.linspace(0.4, 1.0, len(fi_df)))
        ax2.barh(fi_df['Feature'], fi_df['Importance'], color=colors_fi, height=0.6)
        ax2.set_xlabel('Importance Score', color='#b2bec3', fontsize=9)
        ax2.tick_params(colors='#b2bec3', labelsize=8)
        ax2.set_title('Top 15 Feature Importances', color='#f9ca24', fontsize=11, pad=10)
        for spine in ax2.spines.values():
            spine.set_edgecolor('rgba(255,255,255,0.08)')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with d2:
        st.markdown('<div class="section-head">🎯 Model Performance</div>', unsafe_allow_html=True)

        # Static classification report metrics from training
        report_data = {
            'Class': ['Fatal injury', 'Serious Injury', 'Slight Injury'],
            'Precision': [0.99, 0.98, 0.89],
            'Recall':    [0.99, 0.88, 0.98],
            'F1-Score':  [0.99, 0.93, 0.93]
        }
        report_df = pd.DataFrame(report_data)

        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        fig3.patch.set_facecolor('#1a1a2e')
        ax3.set_facecolor('#1a1a2e')
        x     = np.arange(len(report_data['Class']))
        width = 0.25
        colors_m = ['#f9ca24', '#f0932b', '#6ab04c']
        for i, (metric, color) in enumerate(zip(['Precision','Recall','F1-Score'], colors_m)):
            ax3.bar(x + i*width, report_df[metric], width, label=metric, color=color, alpha=0.9)
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(report_data['Class'], rotation=15, ha='right', color='#b2bec3', fontsize=8)
        ax3.set_ylim(0, 1.1)
        ax3.set_ylabel('Score', color='#b2bec3', fontsize=9)
        ax3.set_title('Precision · Recall · F1 by Class', color='#f9ca24', fontsize=11, pad=10)
        ax3.legend(facecolor='#1a1a2e', edgecolor='none', labelcolor='#b2bec3', fontsize=8)
        ax3.tick_params(colors='#b2bec3', labelsize=8)
        for spine in ax3.spines.values():
            spine.set_edgecolor('rgba(255,255,255,0.08)')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        # Confusion matrix
        st.markdown('<div class="section-head">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
        cm_data = np.array([[2065, 18, 2], [128, 1848, 124], [6, 37, 2021]])
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        fig4.patch.set_facecolor('#1a1a2e')
        ax4.set_facecolor('#1a1a2e')
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='YlOrRd',
                    xticklabels=lb.classes_, yticklabels=lb.classes_,
                    ax=ax4, linewidths=0.5, linecolor='#1a1a2e',
                    annot_kws={'size': 9, 'color': 'black'})
        ax4.set_xlabel('Predicted', color='#b2bec3', fontsize=9)
        ax4.set_ylabel('Actual', color='#b2bec3', fontsize=9)
        ax4.tick_params(colors='#b2bec3', labelsize=7, rotation=15)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

# ════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style="font-family:'DM Sans',sans-serif; color:#dfe6e9; line-height:1.8; max-width:700px;">
    <div class="section-head">🧠 About This App</div>
    <p>This application predicts the severity of road traffic accidents using a <b style="color:#f9ca24">Random Forest Classifier</b>
    trained on the RTA (Road Traffic Accident) Dataset.</p>

    <div class="section-head">📦 Pipeline</div>
    <p>
    <b style="color:#f9ca24">1. Data Cleaning</b> — Null values filled with 'Unknown', trailing spaces stripped<br>
    <b style="color:#f9ca24">2. Encoding</b> — OrdinalEncoder converts categorical features to numbers<br>
    <b style="color:#f9ca24">3. Feature Selection</b> — SelectKBest with chi² selects top 50 features<br>
    <b style="color:#f9ca24">4. Balancing</b> — SMOTE oversamples minority classes to 10,415 each<br>
    <b style="color:#f9ca24">5. Training</b> — Random Forest with 500 trees, class weights {Fatal:10, Serious:5, Slight:1}<br>
    <b style="color:#f9ca24">6. Deployment</b> — Streamlit app with joblib-saved pipeline artifacts
    </p>

    <div class="section-head">📁 Saved Artifacts</div>
    <p>
    <code style="color:#f9ca24">model.pkl</code> — Trained Random Forest classifier<br>
    <code style="color:#f9ca24">encoder.pkl</code> — OrdinalEncoder (fitted on training features)<br>
    <code style="color:#f9ca24">selector.pkl</code> — SelectKBest chi² feature selector<br>
    <code style="color:#f9ca24">label_encoder.pkl</code> — LabelEncoder for target classes<br>
    <code style="color:#f9ca24">features.pkl</code> — Ordered feature names list
    </p>

    <div class="section-head">📊 Dataset</div>
    <p>RTA Dataset · 12,316 records · 33 original features · 3 target classes<br>
    Source: Ethiopian Road Traffic Authority</p>
    </div>
    """, unsafe_allow_html=True)
