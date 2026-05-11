import json
import warnings

import joblib
import pandas as pd
import streamlit as st

from predict import (
    ROAD_COMPLAINT_TYPES, COMPLAINT_SEVERITY, NUMERIC_FEATURES,
    LEVEL_NAMES, _map_descriptor,
)

warnings.filterwarnings('ignore')

MODEL_PATH    = 'road_xgb_model.joblib'
SCALER_PATH   = 'road_xgb_scaler.joblib'
FEATURES_PATH = 'road_xgb_features.json'

LEVEL_COLORS = {'Low':'#2ecc71','Medium':'#f39c12','High':'#e67e22','Critical':'#e74c3c'}
LEVEL_ICONS  = {'Low':'🟢','Medium':'🟡','High':'🟠','Critical':'🔴'}
BOROUGHS  = ['BRONX','BROOKLYN','MANHATTAN','QUEENS','STATEN ISLAND','Unspecified']
STATUSES  = ['Closed','Open','In Progress','Pending','Started','Unspecified']
CHANNELS  = ['ONLINE','MOBILE','PHONE','OTHER','UNKNOWN']
DAY_NAMES = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
COMPLAINT_LIST = sorted(ROAD_COMPLAINT_TYPES) + ['Illegal Parking','Noise - Residential','UNSANITARY CONDITION']

st.set_page_config(page_title='Road Deterioration — Predict', page_icon='🛣️', layout='wide')


def sidx(lst, val):
    return lst.index(val) if val in lst else 0


@st.cache_resource(show_spinner='Loading model …')
def load_artifacts():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH) as f:
        feature_names = json.load(f)
    return model, scaler, feature_names


def predict_single(complaint_type, descriptor, borough, status, channel,
                   hour, day_of_week_int, month, resolution_hours,
                   model, scaler, feature_names):
    descriptor_cat  = _map_descriptor(descriptor)
    severity_weight = COMPLAINT_SEVERITY.get(complaint_type, 1)
    row = {col: 0 for col in feature_names}
    row.update({
        'hour_of_day': hour, 'day_of_week': day_of_week_int,
        'month': month, 'is_weekend': int(day_of_week_int >= 5),
        'severity_weight': severity_weight, 'resolution_hours': resolution_hours,
    })
    for prefix, value in [('borough', borough), ('open_data_channel_type', channel),
                           ('status', status), ('descriptor_cat', descriptor_cat)]:
        col = f'{prefix}_{value}'
        if col in row:
            row[col] = 1
    df_row   = pd.DataFrame([row])[feature_names]
    num_cols = [c for c in NUMERIC_FEATURES if c in df_row.columns]
    df_row[num_cols] = scaler.transform(df_row[num_cols])
    pred  = model.predict(df_row)[0]
    proba = model.predict_proba(df_row)[0]
    return int(pred), LEVEL_NAMES[int(pred)], proba


try:
    model, scaler, feature_names = load_artifacts()
    artifacts_ok = True
except Exception as e:
    st.error(f'Could not load model artifacts: {e}')
    artifacts_ok = False

st.title('🔍 Road Deterioration Prediction')
st.markdown('Fill in the complaint details below, or load a level example.')

if not artifacts_ok:
    st.stop()

EXAMPLES = {
    'Low':      {'complaint_type':'Blocked Driveway','descriptor':'DRIVEWAY','borough':'BRONX','status':'Closed','channel':'ONLINE','hour':9,'day':'Tuesday','month':2,'resolution_hours':1.0},
    'Medium':   {'complaint_type':'Street Light Condition','descriptor':'STREET LIGHT','borough':'QUEENS','status':'Closed','channel':'ONLINE','hour':14,'day':'Wednesday','month':2,'resolution_hours':6.0},
    'High':     {'complaint_type':'Curb Condition','descriptor':'BROKEN CURB','borough':'MANHATTAN','status':'Closed','channel':'ONLINE','hour':10,'day':'Tuesday','month':3,'resolution_hours':2.0},
    'Critical': {'complaint_type':'Street Condition','descriptor':'POTHOLE','borough':'BROOKLYN','status':'Open','channel':'ONLINE','hour':10,'day':'Monday','month':2,'resolution_hours':300.0},
}

for k, v in EXAMPLES['Low'].items():
    if f'form_{k}' not in st.session_state:
        st.session_state[f'form_{k}'] = v

st.markdown('**Load example by level:**')
for col, (lvl, ex) in zip(st.columns(4), EXAMPLES.items()):
    if col.button(f'{LEVEL_ICONS[lvl]} {lvl}', use_container_width=True, key=f'btn_{lvl}'):
        for k, v in ex.items():
            st.session_state[f'form_{k}'] = v
        st.rerun()
st.markdown('---')

with st.form('predict_form'):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Complaint Details')
        complaint_type = st.selectbox('Complaint Type', COMPLAINT_LIST,
            index=sidx(COMPLAINT_LIST, st.session_state['form_complaint_type']))
        descriptor = st.text_input('Descriptor', value=st.session_state['form_descriptor'],
            help='e.g. DRIVEWAY, STREET LIGHT, BROKEN CURB, POTHOLE, SNOW')
        borough = st.selectbox('Borough',   BOROUGHS, index=sidx(BOROUGHS,  st.session_state['form_borough']))
        status  = st.selectbox('Status',    STATUSES, index=sidx(STATUSES,  st.session_state['form_status']))
        channel = st.selectbox('Submission Channel', CHANNELS, index=sidx(CHANNELS, st.session_state['form_channel']))
    with col2:
        st.subheader('Temporal & Resolution')
        hour  = st.slider('Hour of Day', 0, 23, int(st.session_state['form_hour']))
        day   = st.selectbox('Day of Week', DAY_NAMES, index=sidx(DAY_NAMES, st.session_state['form_day']))
        month = st.slider('Month', 1, 12, int(st.session_state['form_month']))
        resolution_hours = st.number_input(
            'Resolution Hours (0 = not yet resolved)',
            min_value=0.0, max_value=720.0,
            value=float(st.session_state['form_resolution_hours']), step=1.0)
        st.markdown('---')
        st.caption('**Scoring:** severity (1–3) + descriptor bonus (0–1) + resolution bucket (0–2).  '
                   'Score ≤1→Low · =2→Medium · =3→High · ≥4→Critical')
    submitted = st.form_submit_button('🔮 Predict Deterioration Level', use_container_width=True)

if submitted:
    pred_int, level_name, proba = predict_single(
        complaint_type, descriptor, borough, status, channel,
        hour, DAY_NAMES.index(day), month, resolution_hours,
        model, scaler, feature_names)
    color = LEVEL_COLORS[level_name]
    icon  = LEVEL_ICONS[level_name]
    st.markdown('---')
    st.subheader('Prediction Result')
    res_col, prob_col = st.columns([1, 2])
    with res_col:
        st.markdown(
            f'<div style="background:{color};border-radius:14px;padding:30px;text-align:center;color:white">'
            f'<div style="font-size:52px">{icon}</div>'
            f'<div style="font-size:28px;font-weight:bold">{level_name}</div>'
            f'<div style="font-size:16px;margin-top:6px">Confidence: {proba[pred_int]*100:.1f}%</div>'
            f'</div>', unsafe_allow_html=True)
    with prob_col:
        st.markdown('**Class Probabilities**')
        for lvl, prob in zip(LEVEL_NAMES, proba):
            pct  = int(prob * 100)
            bold = 'bold' if lvl == level_name else 'normal'
            st.markdown(
                f'<div style="margin-bottom:8px">'
                f'<span style="width:90px;display:inline-block">{LEVEL_ICONS[lvl]} <b>{lvl}</b></span>'
                f'<div style="display:inline-block;width:55%;background:#eee;border-radius:6px;height:20px;vertical-align:middle">'
                f'<div style="width:{pct}%;background:{LEVEL_COLORS[lvl]};border-radius:6px;height:20px"></div></div>'
                f'<span style="margin-left:8px;font-weight:{bold}">{prob*100:.1f}%</span></div>',
                unsafe_allow_html=True)
    with st.expander('Input summary'):
        st.json({'complaint_type':complaint_type,'descriptor':descriptor,
                 'descriptor_cat':_map_descriptor(descriptor),'borough':borough,
                 'status':status,'channel':channel,'hour_of_day':hour,'day_of_week':day,
                 'month':month,'resolution_hours':resolution_hours,
                 'severity_weight':COMPLAINT_SEVERITY.get(complaint_type,1)})
