import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Accident Severity Predictor", page_icon="🚦", layout="wide")

PLOT_BG   = '#1a1033'
PLOT_FONT = '#b2bec3'
PLOT_GRID = '#ffffff15'

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 1.5rem 2rem 4rem 2rem; }
.header-wrap { text-align:center; padding:1.5rem 0 1rem 0; border-bottom:1px solid #f9ca2433; margin-bottom:1.5rem; }
.header-icon { font-size:3rem; line-height:1; display:block; margin-bottom:0.3rem; }
.header-title { font-family:'Bebas Neue',sans-serif; font-size:3.2rem; letter-spacing:6px;
    background:linear-gradient(90deg,#f9ca24,#f0932b,#eb4d4b,#6ab04c);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin:0; line-height:1; }
.header-sub { font-family:'DM Sans',sans-serif; color:#b2bec3; font-size:0.95rem; margin-top:0.4rem; }
.metric-row { display:flex; gap:1rem; margin-bottom:1.5rem; flex-wrap:wrap; }
.metric-card { flex:1; min-width:130px; background:rgba(255,255,255,0.05); border:1px solid #f9ca2440;
    border-radius:12px; padding:1rem 1.2rem; text-align:center; }
.metric-card .val { font-family:'Bebas Neue',sans-serif; font-size:2rem; letter-spacing:2px; color:#f9ca24; display:block; }
.metric-card .lbl { font-family:'DM Sans',sans-serif; font-size:0.72rem; color:#b2bec3; text-transform:uppercase; letter-spacing:1px; }
.section-head { font-family:'Bebas Neue',sans-serif; font-size:1.3rem; letter-spacing:3px; color:#f9ca24;
    border-bottom:1px solid #f9ca2433; padding-bottom:6px; margin:1.5rem 0 1rem 0; }
label { font-family:'DM Sans',sans-serif !important; font-weight:500 !important; color:#f0e6ff !important;
    font-size:0.8rem !important; letter-spacing:0.5px !important; text-transform:uppercase !important; }
[data-testid="stSelectbox"] > div > div { background:rgba(255,255,255,0.07) !important;
    border:1px solid #f9ca2450 !important; border-radius:8px !important; color:#ffffff !important; }
[data-testid="stNumberInput"] input { background:rgba(249,202,36,0.1) !important;
    border:1.5px solid #f9ca2480 !important; border-radius:8px !important; color:#f9ca24 !important; font-weight:600 !important; }
[data-testid="stNumberInput"] button { background:rgba(249,202,36,0.15) !important;
    border:1px solid #f9ca2450 !important; color:#f9ca24 !important; border-radius:6px !important; }
[data-testid="stButton"] > button { font-family:'Bebas Neue',sans-serif !important; font-size:1.4rem !important;
    letter-spacing:4px !important; background:linear-gradient(90deg,#eb4d4b,#f0932b,#f9ca24) !important;
    color:#0f0c29 !important; border:none !important; border-radius:10px !important;
    width:100% !important; margin-top:1rem !important; }
[data-testid="stAlert"] { font-family:'DM Sans',sans-serif !important; font-size:1.1rem !important;
    font-weight:600 !important; border-radius:12px !important; margin-top:1rem !important; }
[data-testid="stTabs"] [role="tab"] { font-family:'DM Sans',sans-serif !important; color:#b2bec3 !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:#f9ca24 !important; border-bottom:2px solid #f9ca24 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    try:
        model    = joblib.load('model.pkl')
        encoder  = joblib.load('encoder.pkl')
        selector = joblib.load('selector.pkl')
        lb       = joblib.load('label_encoder.pkl')
        features = joblib.load('features.pkl')
        return model, encoder, selector, lb, features
    except FileNotFoundError as e:
        st.error(f"Model artifacts not found: {e}")
        st.stop()

model, encoder, selector, lb, features = load_artifacts()

st.markdown("""
<div class="header-wrap">
  <span class="header-icon">🚦</span>
  <div class="header-title">Accident Severity Predictor</div>
  <div class="header-sub">ML-powered road accident injury severity classification · Random Forest · 95% Accuracy</div>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="metric-row">
    <div class="metric-card"><span class="val">95%</span><span class="lbl">Model Accuracy</span></div>
    <div class="metric-card"><span class="val">500</span><span class="lbl">Decision Trees</span></div>
    <div class="metric-card"><span class="val">17</span><span class="lbl">Input Features</span></div>
    <div class="metric-card"><span class="val">31K</span><span class="lbl">Training Samples</span></div>
    <div class="metric-card"><span class="val">3</span><span class="lbl">Severity Classes</span></div>
    <div class="metric-card"><span class="val">SMOTE</span><span class="lbl">Balancing Method</span></div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯  Predict Severity", "📊  Model Dashboard", "ℹ️  About"])

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

    if casualties > vehicles * 5:
        st.warning("⚠️ Casualties seem high relative to vehicles — please verify.")

    if st.button('🔍  PREDICT SEVERITY', use_container_width=True):
        try:
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

            thresholds = {0: 0.05, 1: 0.20, 2: 0.60}
            if proba[0] >= thresholds[0]:
                label = 'Fatal injury'
            elif proba[1] >= thresholds[1]:
                label = 'Serious Injury'
            else:
                label = 'Slight Injury'

            if label == 'Fatal injury':
                st.error(f'☠️ **Predicted Severity: {label}** — Immediate emergency response required.  \n📊 Confidence: **{confidence:.1f}%**')
            elif label == 'Serious Injury':
                st.warning(f'🔶 **Predicted Severity: {label}** — Urgent medical attention needed.  \n📊 Confidence: **{confidence:.1f}%**')
            else:
                st.success(f'✅ **Predicted Severity: {label}** — Minor injuries expected.  \n📊 Confidence: **{confidence:.1f}%**')

            st.markdown('<div class="section-head">📊 Class Probabilities</div>', unsafe_allow_html=True)
            fig_prob = go.Figure(go.Bar(
                x=proba * 100, y=lb.classes_, orientation='h',
                marker_color=['#eb4d4b','#f0932b','#6ab04c'],
                text=[f'{p*100:.1f}%' for p in proba],
                textposition='outside', textfont=dict(color='white', size=11)
            ))
            fig_prob.update_layout(
                paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                font=dict(color=PLOT_FONT, family='DM Sans'),
                xaxis=dict(range=[0,115], gridcolor=PLOT_GRID, title='Probability (%)'),
                yaxis=dict(gridcolor=PLOT_GRID),
                height=200, margin=dict(l=10,r=10,t=10,b=10), showlegend=False
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # Download prediction result
            result_df = pd.DataFrame({
                'Input': features,
                'Value': [day, vehicles, casualties, area, junction, age_driver, sex_driver,
                          education, relation, vehicle_type, experience, service_year,
                          collision, sex_casualty, age_casualty, cause, hour]
            })
            result_df['Prediction'] = label
            result_df['Confidence'] = f'{confidence:.1f}%'
            result_df['Fatal_prob'] = f'{proba[0]*100:.1f}%'
            result_df['Serious_prob'] = f'{proba[1]*100:.1f}%'
            result_df['Slight_prob'] = f'{proba[2]*100:.1f}%'

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Prediction Report",
                data=csv,
                file_name="accident_severity_prediction.csv",
                mime="text/csv"
            )

            with st.expander("🔍 View Risk Factor Analysis"):
                risk_factors = []
                if hour < 6 or hour > 22:
                    risk_factors.append("• Late night/early morning accident")
                if age_driver in ["Under 18", "Over 51"]:
                    risk_factors.append(f"• Driver age risk factor: {age_driver}")
                if cause in ["Driving at high speed", "Driving carelessly"]:
                    risk_factors.append(f"• High-risk behavior: {cause}")
                if collision == "Rollover":
                    risk_factors.append("• High-severity collision type: Rollover")
                if risk_factors:
                    for f in risk_factors:
                        st.write(f)
                else:
                    st.write("No major risk factors identified.")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

with tab2:
    d1, d2 = st.columns(2)
    with d1:
        st.markdown('<div class="section-head">🌲 Feature Importance</div>', unsafe_allow_html=True)
        fi_df = pd.DataFrame({'Feature': selector.get_feature_names_out(), 'Importance': model.feature_importances_})
        fi_df = fi_df.sort_values('Importance', ascending=True).tail(15)
        fig_fi = go.Figure(go.Bar(
            x=fi_df['Importance'], y=fi_df['Feature'], orientation='h',
            marker=dict(color=fi_df['Importance'], colorscale='YlOrRd', showscale=False)
        ))
        fig_fi.update_layout(
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=PLOT_FONT, family='DM Sans'),
            xaxis=dict(gridcolor=PLOT_GRID, title='Importance Score'),
            yaxis=dict(gridcolor=PLOT_GRID),
            title=dict(text='Top 15 Feature Importances', font=dict(color='#f9ca24')),
            height=450, margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with d2:
        st.markdown('<div class="section-head">🎯 Model Performance</div>', unsafe_allow_html=True)
        classes = ['Fatal injury','Serious Injury','Slight Injury']
        fig_perf = go.Figure()
        for metric, vals, color in zip(
            ['Precision','Recall','F1-Score'],
            [[0.99,0.98,0.89],[0.99,0.88,0.98],[0.99,0.93,0.93]],
            ['#f9ca24','#f0932b','#6ab04c']
        ):
            fig_perf.add_trace(go.Bar(name=metric, x=classes, y=vals, marker_color=color))
        fig_perf.update_layout(
            barmode='group', paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=PLOT_FONT, family='DM Sans'),
            xaxis=dict(gridcolor=PLOT_GRID), yaxis=dict(gridcolor=PLOT_GRID, range=[0,1.1], title='Score'),
            title=dict(text='Precision · Recall · F1 by Class', font=dict(color='#f9ca24')),
            legend=dict(bgcolor=PLOT_BG, bordercolor='#555'),
            height=280, margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        st.markdown('<div class="section-head">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
        fig_cm = go.Figure(go.Heatmap(
            z=[[2065,18,2],[128,1848,124],[6,37,2021]],
            x=classes, y=classes, colorscale='YlOrRd',
            text=[[2065,18,2],[128,1848,124],[6,37,2021]],
            texttemplate='%{text}', textfont=dict(size=11), showscale=False
        ))
        fig_cm.update_layout(
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=PLOT_FONT, family='DM Sans'),
            xaxis=dict(title='Predicted'), yaxis=dict(title='Actual', autorange='reversed'),
            height=280, margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

with tab3:
    st.markdown("""
    <div style="font-family:'DM Sans',sans-serif; color:#dfe6e9; line-height:1.8; max-width:700px;">
    <div class="section-head">🧠 About This App</div>
    <p>This application predicts road accident severity using a <b style="color:#f9ca24">Random Forest Classifier</b> trained on the RTA Dataset.</p>
    <div class="section-head">📦 Pipeline</div>
    <p>
    <b style="color:#f9ca24">1. Data Cleaning</b> — Null values filled with 'Unknown'<br>
    <b style="color:#f9ca24">2. Encoding</b> — OrdinalEncoder converts categories to numbers<br>
    <b style="color:#f9ca24">3. Feature Selection</b> — SelectKBest chi² selects top features<br>
    <b style="color:#f9ca24">4. Balancing</b> — SMOTE oversamples minority classes<br>
    <b style="color:#f9ca24">5. Training</b> — 500 trees · class weights {Fatal:10, Serious:5, Slight:1}<br>
    <b style="color:#f9ca24">6. Threshold Adjustment</b> — Corrects for 84.5% class imbalance<br>
    <b style="color:#f9ca24">7. Deployment</b> — Streamlit + joblib artifacts
    </p>
    <div class="section-head">📁 Artifacts</div>
    <p>
    <code style="color:#f9ca24">model.pkl</code> · <code style="color:#f9ca24">encoder.pkl</code> ·
    <code style="color:#f9ca24">selector.pkl</code> · <code style="color:#f9ca24">label_encoder.pkl</code> ·
    <code style="color:#f9ca24">features.pkl</code>
    </p>
    <div class="section-head">📊 Dataset</div>
    <p>RTA Dataset · 12,316 records · 33 features · Source: Ethiopian Road Traffic Authority</p>
    <div class="section-head">⚠️ Disclaimer</div>
    <p>For informational purposes only. Always follow official road safety and emergency protocols.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#b2bec3;font-size:0.8rem;'>🚗 Drive Safe · Built with Streamlit · Random Forest Classifier</div>", unsafe_allow_html=True)
