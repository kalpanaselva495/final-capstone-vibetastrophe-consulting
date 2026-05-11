"""
Capstone Web Application
========================
Integrates all 5 models into a single web interface using Streamlit.

Run locally:  streamlit run webapp/app.py
Deploy:       Push to GitHub, then connect to Streamlit Community Cloud
              https://streamlit.io/cloud (free hosting)
"""
import streamlit as st
from pathlib import Path
import sys
from PIL import Image
import joblib
import numpy as np
import pandas as pd
from transformers import ( 
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

traditional_feature_controls = { 
    'Distance(mi)': { 'label': 'Distance (mi)', 'min': 0, 'max': 100, 'control': 'slider' },
    'Timezone': { 'label': 'Timezone', 'options': {"UTC+5": 5, "UTC+6": 6,"UTC+7": 7,"UTC+8": 8}, 'control': 'selectbox' },
    'Temperature(F)': { 'label': 'Temperature (F)', 'min': -50, 'max': 150, 'control': 'slider' },
    'Wind_Chill(F)': { 'label': 'Wind Chill (F)', 'min': -50, 'max': 150, 'control': 'slider' },
    'Humidity(%)': { 'label': 'Humidity (%)', 'min': 0, 'max': 100, 'control': 'slider' },
    'Pressure(in)': { 'label': 'Pressure (in)', 'min': 28, 'max': 32, 'control': 'slider' },
    'Visibility(mi)': { 'label': 'Visibility (mi)', 'min': 0, 'max': 10, 'control': 'slider' },
    'Wind_Speed(mph)': { 'label': 'Wind Speed (mph)', 'min': 0, 'max': 100, 'control': 'slider' },
    'Precipitation(in)': { 'label': 'Precipitation (in)', 'min': 0, 'max': 10, 'control': 'slider' },
    'Amenity': { 'label': 'Amenity', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Bump': { 'label': 'Bump', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Crossing': { 'label': 'Crossing', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Give_Way': { 'label': 'Give Way', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Junction': { 'label': 'Junction', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'No_Exit': { 'label': 'No Exit', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Railway': { 'label': 'Railway', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Roundabout': { 'label': 'Roundabout', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Station': { 'label': 'Station', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Stop': { 'label': 'Stop', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Traffic_Calming': { 'label': 'Traffic Calming', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Traffic_Signal': { 'label': 'Traffic Signal', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Turning_Loop': { 'label': 'Turning Loop', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Sunrise_Sunset': { 'label': 'Sunrise Sunset', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Civil_Twilight': { 'label': 'Civil Twilight', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Nautical_Twilight': { 'label': 'Nautical Twilight', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Astronomical_Twilight': { 'label': 'Astronomical Twilight', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'hour': { 'label': 'Hour', 'min': 0, 'max': 23, 'control': 'slider' },
    'day_of_week': { 'label': 'Day of Week', 'options': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 'control': 'selectbox' },
    'month': { 'label': 'Month', 'options': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], 'control': 'selectbox' },
    'is_weekend': { 'label': 'Is Weekend', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'is_morning_rush': { 'label': 'Is Morning Rush', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'is_evening_rush': { 'label': 'Is Evening Rush', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'is_rush_hour': { 'label': 'Is Rush Hour', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'duration_min': { 'label': 'Duration (min)', 'min': 0, 'max': 120, 'control': 'slider' },
    'wind_dir_deg': { 'label': 'Wind Direction (deg)', 'min': 0, 'max': 360, 'control': 'slider' },
    'weather_cond_num': { 'label': 'Weather Condition', 'min': 0, 'max': 10, 'control': 'slider' },
    'weather_data_available': { 'label': 'Weather Data Available', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'is_freezing': { 'label': 'Is Freezing', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'low_visibility': { 'label': 'Low Visibility', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'accident_dir': { 'label': 'Accident Direction', 'options': ['North', 'South', 'East', 'West'], 'control': 'selectbox' },
    'lat_bin': { 'label': 'Latitude Bin', 'min': 0, 'max': 10, 'control': 'slider' },
    'n_road_features': { 'label': 'Number of Road Features', 'min': 0, 'max': 10, 'control': 'slider' },
    'has_traffic_control': { 'label': 'Has Traffic Control', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Severity_Binary': { 'label': 'Severity Binary', 'options': ['Yes', 'No'], 'control': 'selectbox' }
}

from models.model3_cnn.inference import THRESHOLD, predict_single_image

def convert_feature_value(feature_name, value):
    control_config = traditional_feature_controls[feature_name]

    # Dictionary-backed selectbox, e.g. Timezone
    if isinstance(control_config.get("options"), dict):
        return control_config["options"][value]

    # Yes/No selectboxes
    if value == "Yes":
        return 1
    if value == "No":
        return 0

    # Day names
    day_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }

    if feature_name == "day_of_week":
        return day_map[value]

    # Month names
    month_map = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }

    if feature_name == "month":
        return month_map[value]

    # Direction labels
    accident_dir_map = {
        "North": 0,
        "East": 1,
        "South": 2,
        "West": 3,
    }

    if feature_name == "accident_dir":
        return accident_dir_map[value]

    return value

# Page config
st.set_page_config(
    page_title="AI Capstone Dashboard",
    page_icon="🔬",
    layout="wide",
)

st.title("AI Capstone Dashboard")
st.write("Select a model from the sidebar to make predictions.")

# Sidebar navigation
model_choice = st.sidebar.selectbox(
    "Choose a Model",
    [
        "Home",
        "Model 1: Traditional ML",
        "Model 2: Deep Learning",
        "Model 3: CNN (Image Classification)",
        "Model 4: NLP (Text Classification)",
        "Model 5: Innovation",
    ],
)

# ---------------------------------------------------------------------------
# Helper: Cache model loading so it only happens once
# ---------------------------------------------------------------------------
# Use @st.cache_resource for models — they load once and stay in memory.
#
# Example:
#     @st.cache_resource
#     def load_model1():
#         import joblib
#         return joblib.load("models/model1_traditional_ml/saved_model/model.joblib")
#
#     @st.cache_resource
#     def load_model3():
#         import tensorflow as tf
#         return tf.keras.models.load_model("models/model3_cnn/saved_model/model.keras")

# ---------------------------------------------------------------------------
# Model pages — fill these in with your model loading and prediction logic
# ---------------------------------------------------------------------------

if model_choice == "Home":
    st.write("Welcome! Use the sidebar to navigate between models.")
    st.write("Each model page lets you input data and see predictions in real time.")

elif model_choice == "Model 1: Traditional ML":
    st.header("Model 1: Traditional ML")

    # ---- INTEGRATION PATTERN (uncomment and adapt) ----
    @st.cache_resource
    def load_model1():
        return joblib.load("models/model1_traditional_ml/saved_model/model.joblib")
    
    @st.cache_resource
    def load_scaler1():
        return joblib.load("models/model1_traditional_ml/saved_model/scaler.joblib")
    
    @st.cache_resource
    def load_feature_cols():
        return joblib.load("models/model1_traditional_ml/saved_model/feature_columns.joblib")
    
    loaded_feature_cols = load_feature_cols()
    feature_cols = [
        col for col in loaded_feature_cols if col not in {"Severity"}
    ]
    model = load_model1()
    scaler = load_scaler1()
    total_columns = len(feature_cols)
    # Create input fields for your features
    col1, col2 = st.columns(2)
    with col1:
        for i in range(total_columns // 2):
            if traditional_feature_controls[feature_cols[i]]['control'] == 'number_input':
                st.number_input(
                    traditional_feature_controls[feature_cols[i]]['label'],
                    min_value=traditional_feature_controls[feature_cols[i]]['min'],
                    max_value=traditional_feature_controls[feature_cols[i]]['max'],
                    key=feature_cols[i]
                )
            elif traditional_feature_controls[feature_cols[i]]['control'] == 'slider':
                st.slider(
                    traditional_feature_controls[feature_cols[i]]['label'],
                    min_value=traditional_feature_controls[feature_cols[i]]['min'],
                    max_value=traditional_feature_controls[feature_cols[i]]['max'],
                    key=feature_cols[i]
                )
            elif traditional_feature_controls[feature_cols[i]]['control'] == 'selectbox':
                st.selectbox(
                    traditional_feature_controls[feature_cols[i]]['label'],
                    options=traditional_feature_controls[feature_cols[i]]['options'],
                    key=feature_cols[i]
                )
        # feature_1 = st.number_input("Feature 1", value=0.0)
        # feature_2 = st.selectbox("Feature 2", ["Option A", "Option B"])
    with col2:
        for i in range(total_columns // 2, total_columns):
            if traditional_feature_controls[feature_cols[i]]['control'] == 'number_input':
                st.number_input(
                    traditional_feature_controls[feature_cols[i]]['label'],
                    min_value=traditional_feature_controls[feature_cols[i]]['min'],
                    max_value=traditional_feature_controls[feature_cols[i]]['max'],
                    key=feature_cols[i]
                )
            elif traditional_feature_controls[feature_cols[i]]['control'] == 'selectbox':
                st.selectbox(
                    traditional_feature_controls[feature_cols[i]]['label'],
                    options=traditional_feature_controls[feature_cols[i]]['options'],
                    key=feature_cols[i]
                )
            elif traditional_feature_controls[feature_cols[i]]['control'] == 'slider':
                st.slider(
                    traditional_feature_controls[feature_cols[i]]['label'],
                    min_value=traditional_feature_controls[feature_cols[i]]['min'],
                    max_value=traditional_feature_controls[feature_cols[i]]['max'],
                    key=feature_cols[i]
                )

    if st.button("Predict"):
        
        input_data = {}

        for feature in feature_cols:
            raw_value = st.session_state[feature]
            input_data[feature] = convert_feature_value(feature, raw_value)

        # input_data.pop("Severity_Binary", None)  # Remove target if accidentally included 
        input_df = pd.DataFrame([input_data])
        input_df[feature_cols] = scaler.transform(input_df[feature_cols])

        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        st.success(f"Prediction: {prediction[0]}")
        st.write(f"Confidence: {probability.max():.2%}")
    # ---- END PATTERN ----

    st.info("Not yet implemented — load your model and add input fields here.")

elif model_choice == "Model 2: Deep Learning":
    st.header("Model 2: Deep Learning")
    # TODO: Load your DNN model and add prediction interface
    # Same pattern as Model 1, but load with:
    import tensorflow as tf

    @st.cache_resource
    def load_model2():
         return tf.keras.models.load_model("models/model2_deep_learning/saved_model/model.keras")
    # model = tf.keras.models.load_model("models/model2_deep_learning/saved_model/model.keras")

    @st.cache_resource
    def load_scaler2():
        return joblib.load("models/model2_deep_learning/saved_model/scaler.joblib")
    
        # The 29-column list used at training time (includes intentional duplicates)
    model2_features = [
        'Distance(mi)', 'Timezone',
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
        'wind_dir_deg', 'weather_cond_num', 'accident_dir',
        'hour', 'day_of_week', 'month', 'is_weekend',
        'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
        'duration_min', 'wind_dir_deg', 'weather_cond_num', 'weather_data_available',
        'is_freezing', 'low_visibility', 'accident_dir', 'lat_bin',
        'n_road_features', 'has_traffic_control'
    ]
    # Unique features for UI (order-preserved, no duplicates)
    features = list(dict.fromkeys(model2_features))

    model = load_model2()
    scaler = load_scaler2()

    total_columns = len(features)
    col1, col2 = st.columns(2)
    with col1:
        for i in range(total_columns // 2):
            feature = features[i]
            control_config = traditional_feature_controls[feature]
            if control_config['control'] == 'number_input':
                st.number_input(
                    control_config['label'],
                    min_value=control_config['min'],
                    max_value=control_config['max'],
                    key=feature
                )
            elif control_config['control'] == 'slider':
                st.slider(
                    control_config['label'],
                    min_value=control_config['min'],
                    max_value=control_config['max'],
                    key=feature
                )
            elif control_config['control'] == 'selectbox':
                st.selectbox(
                    control_config['label'],
                    options=control_config['options'],
                    key=feature
                )
    with col2:
        for i in range(total_columns // 2, total_columns):
            feature = features[i]
            control_config = traditional_feature_controls[feature]
            if control_config['control'] == 'number_input':
                st.number_input(
                    control_config['label'],
                    min_value=control_config['min'],
                    max_value=control_config['max'],
                    key=feature
                )
            elif control_config['control'] == 'slider':
                st.slider(
                    control_config['label'],
                    min_value=control_config['min'],
                    max_value=control_config['max'],
                    key=feature
                )
            elif control_config['control'] == 'selectbox':
                st.selectbox(
                    control_config['label'],
                    options=control_config['options'],
                    key=feature
                )

    

    if st.button("Predict"):
        input_data = {}

        for feature in features:
            raw_value = st.session_state[feature]
            input_data[feature] = convert_feature_value(feature, raw_value)
            
        input_df = pd.DataFrame([input_data])[model2_features]  # 29 cols with duplicates
        scaled_input_df = scaler.transform(input_df)
        predictions = model.predict(scaled_input_df)
        label = "High Severity" if predictions[0][0] >= 0.5 else "Low Severity"
        st.success(f"Prediction: {label}")
        st.write(f"Confidence: {predictions[0][0]:.2%}")
        

elif model_choice == "Model 3: CNN (Image Classification)":
    st.header("Model 3: CNN — Image Classification")

    # ---- INTEGRATION PATTERN (uncomment and adapt) ----
    @st.cache_resource
    def load_model3():
        import tensorflow as tf
        return tf.keras.models.load_model("models/model3_cnn/saved_model/efficientnet_model.keras")
    
    model = load_model3()
    
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
        if st.button("Classify"):
            result = predict_single_image(model, uploaded_file, threshold=THRESHOLD)
            if result["confidence"] >= THRESHOLD:
                st.success(f"Prediction: {result['label']}")
            else:
                st.warning(f"Prediction: {result['label']}")
            st.write(f"Confidence: {result['confidence']:.2%}")
            st.caption(f"Decision threshold: {result['threshold']:.2f}")
    # ---- END PATTERN ----

    st.info("Not yet implemented — add image upload and classification here.")

elif model_choice == "Model 4: NLP (Text Classification)":
    st.header("Model 4: NLP — Text Classification")

    categories = {
        0: "Blocked Driveway",
        1: "Heat/Hot water",
        2: "Illegal Parking",
        3: "Noise - Residential",
        4: "Other",
        5: "Snow or Ice"
    }

    # ---- INTEGRATION PATTERN (uncomment and adapt) ----
    @st.cache_resource
    def load_model4():
        model = DistilBertForSequenceClassification.from_pretrained("models/model4_nlp_classification/saved_model/")
        tokenizer = DistilBertTokenizerFast.from_pretrained("models/model4_nlp_classification/saved_model/")
        model.eval()
        return model, tokenizer
    
    model, tokenizer = load_model4()
    
    user_text = st.text_area("Enter text to classify:", height=150)
    if st.button("Classify") and user_text:
        inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        prediction = outputs.logits.argmax(dim=1).item()
        confidence = outputs.logits.softmax(dim=1).max().item()
        st.success(f"Predicted Category: {categories[prediction]}")
        st.write(f"Confidence: {confidence:.2%}")
    # ---- END PATTERN ----

elif model_choice == "Model 5: Innovation":
    st.header("Model 5: Innovation")
    st.write("Predict 311 complaint category and expected resolution time.")
    
    @st.cache_resource
    def load_model5():
        base_path = Path("models/model5_innovation/saved_model")
        return {
            "clf_model": joblib.load(base_path / "clf_model.joblib"),
            "clf_le": joblib.load(base_path / "clf_le.joblib"),
            "reg_model": joblib.load(base_path / "reg_model.joblib"),
            "reg_le": joblib.load(base_path / "reg_le.joblib"),
            "reg_feature_cols": joblib.load(base_path / "reg_feature_cols.joblib"),
        }

    model_bundle = load_model5()
    clf_model = model_bundle["clf_model"]
    clf_le = model_bundle["clf_le"]
    reg_model = model_bundle["reg_model"]
    reg_le = model_bundle["reg_le"]

    status_mapping = {
        "Open": 0,
        "Assigned": 1,
        "Started": 2,
        "In Progress": 3,
        "Pending": 4,
        "Closed": 5,
        "Unspecified": 0,
    }

    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    season_mapping = {
        12: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 1,
        6: 2,
        7: 2,
        8: 2,
        9: 3,
        10: 3,
        11: 3,
    }

    def get_hour_bucket(hour):
        if hour < 6:
            return 0
        if hour < 12:
            return 1
        if hour < 18:
            return 2
        return 3

    def build_onehot_value(feature_name):
        if feature_name.startswith("borough_"):
            return feature_name.split("borough_", 1)[1]
        if feature_name.startswith("open_data_channel_type_"):
            return feature_name.split("open_data_channel_type_", 1)[1]
        if feature_name.startswith("agency_"):
            return feature_name.split("agency_", 1)[1]
        return None

    def build_feature_row(feature_names, *, status, created_hour, created_dayofweek, created_month,
                          is_resolved, borough, channel, agency, resolution_hours=None,
                          complaint_type_label=None):
        row_data = {str(col): 0 for col in feature_names}
        row_data["status"] = status_mapping[status]
        row_data["created_hour"] = int(created_hour)
        row_data["created_dayofweek"] = day_names.index(created_dayofweek)
        row_data["created_month"] = month_names.index(created_month) + 1
        row_data["is_resolved"] = 1 if is_resolved else 0
        row_data["is_weekend"] = 1 if row_data["created_dayofweek"] >= 5 else 0
        row_data["season"] = season_mapping[row_data["created_month"]]
        row_data["hour_bucket"] = get_hour_bucket(row_data["created_hour"])

        borough_col = f"borough_{borough}"
        channel_col = f"open_data_channel_type_{channel}"
        agency_col = f"agency_{agency}"

        if borough_col in row_data:
            row_data[borough_col] = 1
        if channel_col in row_data:
            row_data[channel_col] = 1
        if agency_col in row_data:
            row_data[agency_col] = 1

        if "resolution_hours" in row_data:
            row_data["resolution_hours"] = float(resolution_hours if resolution_hours is not None else 0.0)

        if "complaint_type_enc" in row_data and complaint_type_label is not None:
            row_data["complaint_type_enc"] = int(reg_le.transform([complaint_type_label])[0])

        return pd.DataFrame([[row_data[col] for col in feature_names]], columns=feature_names)

    clf_features = [str(col) for col in clf_model.feature_names_in_]
    reg_features = [str(col) for col in reg_model.feature_names_in_]

    borough_options = [build_onehot_value(c) for c in clf_features if str(c).startswith("borough_")]
    channel_options = [
        build_onehot_value(c)
        for c in clf_features
        if str(c).startswith("open_data_channel_type_")
    ]
    agency_options = [build_onehot_value(c) for c in clf_features if str(c).startswith("agency_")]

    col1, col2 = st.columns(2)
    with col1:
        input_status = st.selectbox("Status", options=list(status_mapping.keys()), index=3)
        input_hour = st.slider("Created Hour", min_value=0, max_value=23, value=12)
        input_day = st.selectbox("Created Day of Week", options=day_names, index=0)
        input_month = st.selectbox("Created Month", options=month_names, index=0)
        input_is_resolved = st.selectbox("Is Resolved", options=["No", "Yes"], index=0)
    with col2:
        input_borough = st.selectbox("Borough", options=borough_options, index=0)
        input_channel = st.selectbox("Open Data Channel Type", options=channel_options, index=0)
        input_agency = st.selectbox("Agency", options=agency_options, index=0)
        input_resolution_hours = st.number_input(
            "Resolution Hours (required for classification model)",
            min_value=0.0,
            max_value=10000.0,
            value=24.0,
            step=1.0,
        )

    is_resolved_bool = input_is_resolved == "Yes"

    tab1, tab2 = st.tabs(["Classification", "Regression"])

    with tab1:
        st.caption("Classification model predicts complaint type.")
        if st.button("Predict Complaint Type", key="predict_model5_classification"):
            clf_input = build_feature_row(
                clf_features,
                status=input_status,
                created_hour=input_hour,
                created_dayofweek=input_day,
                created_month=input_month,
                is_resolved=is_resolved_bool,
                borough=input_borough,
                channel=input_channel,
                agency=input_agency,
                resolution_hours=input_resolution_hours,
            )

            pred_encoded = int(clf_model.predict(clf_input)[0])
            pred_label = clf_le.inverse_transform([pred_encoded])[0]
            proba = clf_model.predict_proba(clf_input)[0]
            confidence = float(np.max(proba))

            st.success(f"Predicted Complaint Type: {pred_label}")
            st.write(f"Confidence: {confidence:.2%}")

    with tab2:
        st.caption("Regression model predicts expected resolution time.")
        complaint_options = [str(c) for c in reg_le.classes_]
        reg_complaint_label = st.selectbox(
            "Complaint Type for Regression",
            options=complaint_options,
            index=0,
            key="model5_reg_complaint_type",
        )

        if st.button("Predict Resolution Time", key="predict_model5_regression"):
            reg_input = build_feature_row(
                reg_features,
                status=input_status,
                created_hour=input_hour,
                created_dayofweek=input_day,
                created_month=input_month,
                is_resolved=is_resolved_bool,
                borough=input_borough,
                channel=input_channel,
                agency=input_agency,
                complaint_type_label=reg_complaint_label,
            )

            pred_log_hours = float(reg_model.predict(reg_input)[0])
            pred_hours = max(float(np.expm1(pred_log_hours)), 0.0)
            pred_days = pred_hours / 24.0

            st.success(f"Predicted Resolution Time: {pred_hours:.1f} hours")
            st.write(f"Approximate Days: {pred_days:.2f}")
