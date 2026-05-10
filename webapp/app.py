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
    model = tf.keras.models.load_model("models/model2_deep_learning/saved_model/model.keras")
    
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
        predictions = model.predict(input_df)
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

    # ---- INTEGRATION PATTERN (uncomment and adapt) ----
    @st.cache_resource
    def load_model4():
        # import joblib
        # model = joblib.load("models/model4_nlp_classification/saved_model/model.joblib")
        # vectorizer = joblib.load("models/model4_nlp_classification/saved_model/vectorizer.joblib")
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
        st.success(f"Predicted Category: {prediction}")
        st.write(f"Confidence: {confidence:.2%}")
    # ---- END PATTERN ----

elif model_choice == "Model 5: Innovation":
    st.header("Model 5: Innovation")
    # TODO: Add your custom model interface
    st.info("Not yet implemented — add your innovation model interface here.")
