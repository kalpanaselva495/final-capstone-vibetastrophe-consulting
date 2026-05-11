"""
Model 5 Innovation — Road Deterioration Prediction
===================================================
Streamlit app for XGBoost-based road deterioration severity prediction.

Run:  streamlit run webapp/app_model5_only.py
"""
import streamlit as st
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Road Deterioration Prediction",
    page_icon="🛣️",
    layout="wide",
)

st.title("🛣️ Model 5: Road Deterioration Prediction")
st.write("Predict NYC 311 road complaint deterioration severity using XGBoost.")

# Load Model 5 (XGBoost Road Deterioration)
@st.cache_resource
def load_model5_xgb():
    base_path = Path(__file__).resolve().parent.parent / "models" / "model5_innovation" / "saved_model"
    
    try:
        model = joblib.load(str(base_path / "road_xgb_model.joblib"))
        scaler = joblib.load(str(base_path / "road_xgb_scaler.joblib"))
        
        with open(str(base_path / "road_xgb_features.json"), 'r') as f:
            features = json.load(f)
        
        return {"model": model, "scaler": scaler, "features": features}
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model5_bundle = load_model5_xgb()

if model5_bundle is None:
    st.stop()

model = model5_bundle["model"]
scaler = model5_bundle["scaler"]
feature_names = model5_bundle["features"]

# Constants
ROAD_COMPLAINT_TYPES = {
    'Street Condition', 'Snow or Ice', 'Traffic Signal Condition',
    'Blocked Driveway', 'Street Light Condition',
    'Highway Sign - Damaged', 'Highway Sign - Missing',
    'Sidewalk Condition', 'Curb Condition', 'Pothole',
}

COMPLAINT_SEVERITY = {
    'Street Condition': 3, 'Pothole': 3,
    'Sidewalk Condition': 2, 'Curb Condition': 2,
    'Snow or Ice': 2, 'Traffic Signal Condition': 2,
    'Blocked Driveway': 1, 'Street Light Condition': 1,
    'Highway Sign - Damaged': 2, 'Highway Sign - Missing': 1,
}

LEVEL_NAMES = ['🟢 Low', '🟡 Medium', '🟠 High', '🔴 Critical']
NUMERIC_FEATURES = [
    'hour_of_day', 'day_of_week', 'month', 'is_weekend',
    'severity_weight', 'resolution_hours',
]

BOROUGHS = ['BRONX', 'BROOKLYN', 'MANHATTAN', 'QUEENS', 'STATEN ISLAND']
STATUSES = ['Closed', 'Open', 'In Progress', 'Pending', 'Started']
CHANNELS = ['ONLINE', 'MOBILE', 'PHONE', 'OTHER']
DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def _map_descriptor(desc):
    """Map descriptor text to category."""
    if not isinstance(desc, str):
        return 'Other'
    d = desc.upper()
    street_keywords = ['STREET', 'SIDEWALK', 'CURB', 'POTHOLE', 'ASPHALT', 'PAVED']
    if any(kw in d for kw in street_keywords):
        return 'Street_Sidewalk'
    return 'Other'

def predict_single(complaint_type, descriptor, borough, status, channel,
                   hour, day_of_week_int, month, resolution_hours):
    descriptor_cat = _map_descriptor(descriptor)
    severity_weight = COMPLAINT_SEVERITY.get(complaint_type, 1)
    
    # Build feature row
    row = {str(col): 0 for col in feature_names}
    row.update({
        'hour_of_day': hour, 'day_of_week': day_of_week_int,
        'month': month, 'is_weekend': int(day_of_week_int >= 5),
        'severity_weight': severity_weight, 'resolution_hours': resolution_hours,
    })
    
    # One-hot encoding
    for prefix, value in [('borough', borough), ('open_data_channel_type', channel),
                           ('status', status), ('descriptor_cat', descriptor_cat)]:
        col = f'{prefix}_{value}'
        if col in row:
            row[col] = 1
    
    # Create DataFrame and scale
    df_row = pd.DataFrame([row])[feature_names]
    num_cols = [c for c in NUMERIC_FEATURES if c in df_row.columns]
    df_row[num_cols] = scaler.transform(df_row[num_cols])
    
    # Predict
    pred = model.predict(df_row)[0]
    proba = model.predict_proba(df_row)[0]
    return int(pred), LEVEL_NAMES[int(pred)], proba

# Example presets
EXAMPLES = {
    'Low':      {'complaint_type': 'Blocked Driveway', 'descriptor': 'DRIVEWAY', 'borough': 'BRONX', 'status': 'Closed', 'channel': 'ONLINE', 'hour': 9, 'day': 'Tuesday', 'month': 2, 'resolution_hours': 1.0},
    'Medium':   {'complaint_type': 'Street Light Condition', 'descriptor': 'STREET LIGHT', 'borough': 'QUEENS', 'status': 'Closed', 'channel': 'ONLINE', 'hour': 14, 'day': 'Wednesday', 'month': 2, 'resolution_hours': 6.0},
    'High':     {'complaint_type': 'Curb Condition', 'descriptor': 'BROKEN CURB', 'borough': 'MANHATTAN', 'status': 'Closed', 'channel': 'ONLINE', 'hour': 10, 'day': 'Tuesday', 'month': 3, 'resolution_hours': 24.0},
    'Critical': {'complaint_type': 'Street Condition', 'descriptor': 'POTHOLE', 'borough': 'BROOKLYN', 'status': 'Open', 'channel': 'ONLINE', 'hour': 10, 'day': 'Monday', 'month': 2, 'resolution_hours': 300.0},
}

# Initialize session state with defaults
for k, v in EXAMPLES['Low'].items():
    if f'model5_{k}' not in st.session_state:
        st.session_state[f'model5_{k}'] = v

# Load examples
st.markdown('**Load example by severity level:**')
ex_cols = st.columns(4)
for col, (lvl, ex) in zip(ex_cols, EXAMPLES.items()):
    if col.button(f'📊 {lvl}', use_container_width=True, key=f'btn_m5_{lvl}'):
        for k, v in ex.items():
            st.session_state[f'model5_{k}'] = v
            st.session_state[f'model5_{k}_input'] = v  # keep widget keys in sync
        st.rerun()

st.markdown('---')

# Input form
_complaint_opts = sorted(ROAD_COMPLAINT_TYPES)
_ct_default      = st.session_state.get('model5_complaint_type', _complaint_opts[0])
_borough_default = st.session_state.get('model5_borough', BOROUGHS[0])
_status_default  = st.session_state.get('model5_status', STATUSES[0])
_channel_default = st.session_state.get('model5_channel', CHANNELS[0])
_day_default     = st.session_state.get('model5_day', DAY_NAMES[1])

with st.form('model5_form'):
    col1, col2 = st.columns(2)
    with col1:
        input_complaint = st.selectbox(
            "Complaint Type",
            options=_complaint_opts,
            index=_complaint_opts.index(_ct_default) if _ct_default in _complaint_opts else 0,
            key='model5_complaint_type_input',
        )
        input_descriptor = st.text_input(
            "Descriptor (keyword)",
            value=st.session_state.get('model5_descriptor', 'DRIVEWAY'),
            key='model5_descriptor_input',
        )
        input_borough = st.selectbox(
            "Borough",
            options=BOROUGHS,
            index=BOROUGHS.index(_borough_default) if _borough_default in BOROUGHS else 0,
            key='model5_borough_input',
        )
        input_status = st.selectbox(
            "Status",
            options=STATUSES,
            index=STATUSES.index(_status_default) if _status_default in STATUSES else 0,
            key='model5_status_input',
        )

    with col2:
        input_channel = st.selectbox(
            "Channel",
            options=CHANNELS,
            index=CHANNELS.index(_channel_default) if _channel_default in CHANNELS else 0,
            key='model5_channel_input',
        )
        input_hour = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=st.session_state.get('model5_hour', 12),
            key='model5_hour_input',
        )
        input_day = st.selectbox(
            "Day of Week",
            options=DAY_NAMES,
            index=DAY_NAMES.index(_day_default) if _day_default in DAY_NAMES else 1,
            key='model5_day_input',
        )
        input_month = st.slider(
            "Month",
            min_value=1,
            max_value=12,
            value=st.session_state.get('model5_month', 6),
            key='model5_month_input',
        )
        input_resolution = st.number_input(
            "Resolution Hours",
            min_value=0.0,
            max_value=1000.0,
            value=float(st.session_state.get('model5_resolution_hours', 24.0)),
            step=1.0,
            key='model5_resolution_input',
        )
    
    submit = st.form_submit_button("🔍 Predict Deterioration Level", use_container_width=True)
    
    if submit:
        day_idx = DAY_NAMES.index(input_day)
        pred_level, pred_name, pred_proba = predict_single(
            complaint_type=input_complaint,
            descriptor=input_descriptor,
            borough=input_borough,
            status=input_status,
            channel=input_channel,
            hour=input_hour,
            day_of_week_int=day_idx,
            month=input_month,
            resolution_hours=input_resolution,
        )
        
        # Display results
        col1, col2 = st.columns([2, 1])
        with col1:
            st.success(f"**Predicted Level:** {pred_name}")
            st.write(f"**Confidence:** {pred_proba[pred_level]:.1%}")
        
        with col2:
            # Show all class probabilities
            st.write("**All Levels:**")
            for i, prob in enumerate(pred_proba):
                level_name = LEVEL_NAMES[i].split()[1]
                st.write(f"{LEVEL_NAMES[i]}: {prob:.1%}")

st.markdown("---")
st.info("💡 Tip: Use the example buttons above to load pre-configured scenarios!")
