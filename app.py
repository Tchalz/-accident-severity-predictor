import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="🚦",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }

.block-container {
    padding: 2rem 2rem 4rem 2rem;
    max-width: 860px;
}

/* Header block */
.header-block {
    text-align: center;
    margin-bottom: 1rem;
}
.header-block .icon {
    font-size: 3.5rem;
    line-height: 1;
    display: block;
    margin-bottom: 0.3rem;
}
.header-block h1 {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 3.4rem !important;
    letter-spacing: 5px !important;
    background: linear-gradient(90deg, #f9ca24, #f0932b, #eb4d4b, #6ab04c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 !important;
    line-height: 1 !important;
    display: block;
}
.header-block p {
    font-family: 'DM Sans', sans-serif !important;
    color: #b2bec3 !important;
    font-size: 1rem !important;
    margin-top: 0.5rem !important;
}

/* Section headers */
h3 {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.4rem !important;
    letter-spacing: 3px !important;
    color: #f9ca24 !important;
    border-bottom: 2px solid #f9ca2430;
    padding-bottom: 6px;
    margin-top: 2rem !important;
}

/* Labels */
label {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #f0e6ff !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(249,202,36,0.35) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: #f9ca24 !important;
    box-shadow: 0 0 0 2px rgba(249,202,36,0.15) !important;
}

/* Number input */
[data-testid="stNumberInput"] input {
    background: rgba(249,202,36,0.12) !important;
    border: 1.5px solid #f9ca24 !important;
    border-radius: 8px !important;
    color: #f9ca24 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #ffffff !important;
    box-shadow: 0 0 0 2px rgba(249,202,36,0.3) !important;
    color: #ffffff !important;
}
/* Number input +/- buttons */
[data-testid="stNumberInput"] button {
    background: rgba(249,202,36,0.2) !important;
    border: 1px solid rgba(249,202,36,0.4) !important;
    color: #f9ca24 !important;
    border-radius: 6px !important;
}
[data-testid="stNumberInput"] button:hover {
    background: rgba(249,202,36,0.4) !important;
}

/* Slider track and thumb */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #f9ca24, #f0932b) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #ffffff !important;
    border: 3px solid #f9ca24 !important;
    width: 20px !important;
    height: 20px !important;
    box-shadow: 0 0 8px rgba(249,202,36,0.6) !important;
}
/* Slider current value */
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
    color: #f9ca24 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

/* Predict button */
[data-testid="stButton"] > button {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.5rem !important;
    letter-spacing: 4px !important;
    background: linear-gradient(90deg, #eb4d4b, #f0932b, #f9ca24) !important;
    color: #0f0c29 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.8rem 2rem !important;
    width: 100% !important;
    margin-top: 1.5rem !important;
    cursor: pointer !important;
    box-shadow: 0 4px 24px rgba(235,77,75,0.45) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 36px rgba(235,77,75,0.65) !important;
}

/* Result alerts */
[data-testid="stAlert"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    border-left: 4px solid !important;
    margin-top: 1.5rem !important;
}

hr {
    border-color: rgba(249,202,36,0.15) !important;
    margin: 1.5rem 0 !important;
}

[data-testid="column"] { padding: 0 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
model    = joblib.load('model.pkl')
encoder  = joblib.load('encoder.pkl')
selector = joblib.load('selector.pkl')
lb       = joblib.load('label_encoder.pkl')
features = joblib.load('features.pkl')

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
  <span class="icon">🚦</span>
  <h1>Accident Severity Predictor</h1>
  <p>Fill in the accident details below to predict the injury severity outcome.</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Inputs ─────────────────────────────────────────────────────────────────────
st.markdown("### 🧑 Driver & Casualty Info")
col1, col2 = st.columns(2)

with col1:
    day        = st.selectbox('Day of week', ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    age_driver = st.selectbox('Age band of driver', ['18-30','31-50','Over 51','Under 18','Unknown'])
    sex_driver = st.selectbox('Sex of driver', ['Male','Female','Unknown'])
    education  = st.selectbox('Educational level', ['Above high school','High school','Elementary school','Junior high school','Illiterate','Writing & reading','Unknown'])
    experience = st.selectbox('Driving experience', ['1-2yr','2-5yr','5-10yr','Above 10yr','No Licence','Below 1yr','Unknown'])

with col2:
    sex_casualty = st.selectbox('Sex of casualty', ['Male','Female','Unknown'])
    age_casualty = st.selectbox('Age band of casualty', ['18-30','31-50','Over 51','Under 18','Unknown'])
    casualties   = st.number_input('Number of casualties', 1, 10, 1)
    vehicles     = st.number_input('Number of vehicles involved', 1, 10, 1)
    hour         = st.slider('Hour of day', 0, 23, 12)

st.markdown("### 🚗 Vehicle Info")
col3, col4 = st.columns(2)

with col3:
    vehicle_type = st.selectbox('Type of vehicle', ['Automobile','Public (> 45 seats)','Lorry (11-40Q)','Public (13-45 seats)','Lorry (40-100Q)','Long lorry','Taxi','Pickup flat bed','Motorcycle','Other','Unknown'])
    service_year = st.selectbox('Service year of vehicle', ['Above 10yr','5-10yr','2-5yr','1-2yr','Below 1yr','Unknown'])

with col4:
    relation = st.selectbox('Vehicle driver relation', ['Employee','Owner','Other','Unknown'])

st.markdown("### 🛣️ Road & Accident Info")
col5, col6 = st.columns(2)

with col5:
    area      = st.selectbox('Area accident occurred', ['Residential areas','Office areas','School areas','Church areas','Industrial areas','Other','Unknown'])
    junction  = st.selectbox('Type of junction', ['No junction','Y Shape','T Shape','Crossing','O Shape','Other','Unknown'])
    collision = st.selectbox('Type of collision', ['Vehicle with vehicle collision','Collision with roadside objects','Rollover','Fall from vehicle','Collision with animals','Other','Unknown'])

with col6:
    cause = st.selectbox('Cause of accident', ['Moving Backward','Overtaking','Changing lane to the right','Changing lane to the left','Driving carelessly','Driving at high speed','Driving to the left','No distancing','Other','Unknown'])

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button('PREDICT SEVERITY'):
    input_data = pd.DataFrame([[
        day, vehicles, casualties, area, junction,
        age_driver, sex_driver, education, relation,
        vehicle_type, experience, service_year,
        collision, sex_casualty, age_casualty, cause, hour
    ]], columns=features)

    encoded    = encoder.transform(input_data)
    selected   = selector.transform(encoded)
    prediction = model.predict(selected)
    label      = lb.inverse_transform(prediction)[0]

    if label == 'Fatal injury':
        st.error(f'☠️ Predicted Severity: **{label}** — Immediate emergency response required.')
    elif label == 'Serious Injury':
        st.warning(f'🔶 Predicted Severity: **{label}** — Urgent medical attention needed.')
    else:
        st.success(f'✅ Predicted Severity: **{label}** — Minor injuries expected.')
