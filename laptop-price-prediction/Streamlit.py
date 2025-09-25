import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

#Produces and saves:
# origin_df.csv (raw pre-encoding dataframe used to populate GUI dropdowns)
# onehot_encoder.joblib (fitted OneHotEncoder)
# laptop_price_model.joblib (trained RandomForest model)  

st.title(" 💸 Laptop Price Predictor 💻")

# Load the trained pipeline
@st.cache_resource
def load_model():
    return joblib.load("laptop_price_model.joblib")

@st.cache_resource
def load_encoder():
    return joblib.load("onehot_encoder.joblib")

model = load_model()
ohe = load_encoder()

origin_df = pd.read_csv("origin_df.csv") if os.path.exists("origin_df.csv") else None

st.markdown("Enter laptop details below or upload a CSV with the same columns as training (except Price).")

# Option 1: Single prediction through manual inputs
st.title("💸 Laptop Price Predictor 💻")
st.markdown("Enter laptop details below (friendly inputs will be encoded behind the scenes).")

# Categorical column names must match what you used during fit
categorical_columns = ['Manufacturer', 'Category', 'Cpu brand', 'Gpu brand', 'OS']

# Build dropdown options from origin_df or from encoder categories_
if origin_df is not None:
    categories_lists = [sorted(origin_df[col].astype(str).unique().tolist()) for col in categorical_columns]
elif hasattr(ohe, 'categories_'):
    categories_lists = [list(map(str, cat)) for cat in ohe.categories_]
else:
    categories_lists = None

st.header("Single Prediction")
user_raw = {}

# categorical inputs
for i, col in enumerate(categorical_columns):
    if categories_lists and i < len(categories_lists):
        user_raw[col] = st.selectbox(col, categories_lists[i])
    else:
        user_raw[col] = st.text_input(col, "")

# numeric / binary inputs — ensure names match training numeric columns
user_raw['RAM'] = st.selectbox('RAM (GB)', [2,4,6,8,12,16,24,32,64], index=3)
user_raw['Weight'] = st.number_input('Weight (kg)', min_value=0.5, value=2.5, step=0.1)

touch = st.selectbox('Touchscreen', ['No','Yes'])
user_raw['Touchscreen'] = 1 if touch == 'Yes' else 0

ips = st.selectbox('IPS', ['No','Yes'])
user_raw['Ips'] = 1 if ips == 'Yes' else 0

screen_size = st.slider('Screen size (inches)', min_value=10.0, max_value=18.0, value=13.0, step=0.1)
resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
X_res = int(resolution.split('x')[0]); Y_res = int(resolution.split('x')[1])
user_raw['ppi'] = ((X_res**2 + Y_res**2)**0.5) / screen_size

user_raw['HDD'] = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048], index=2)
user_raw['SSD'] = st.selectbox('SSD (GB)', [0,8,128,256,512,1024], index=2)

if st.button("Predict Price"):
    try:
        # Build raw categorical DataFrame in same order used during encoder.fit
        raw_cat = pd.DataFrame([{col: user_raw[col] for col in categorical_columns}])

        # Encode categoricals (ohe must be the fitted encoder saved from notebook)
        encoded = ohe.transform(raw_cat[categorical_columns])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categorical_columns), index=[0])

        # Build numeric DataFrame in the same order as your model expects.
        # Try to infer model numeric feature order: model.feature_names_in_
        model_feature_names = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
        # Prepare a numeric row using common numeric names you used in training
        numeric_row = {}
        numeric_possible = ['RAM','Weight','HDD','SSD','ppi','Touchscreen','Ips']  # tweak if different
        for nm in numeric_possible:
            numeric_row[nm] = user_raw.get(nm, 0)
        numeric_df = pd.DataFrame([numeric_row])

        # Combine encoded + numeric
        final_df = pd.concat([encoded_df.reset_index(drop=True), numeric_df.reset_index(drop=True)], axis=1)

        # Reindex columns to match model expected order (very important)
        if model_feature_names is not None:
            final_df = final_df.reindex(columns=model_feature_names, fill_value=0)

        final_df = final_df.astype(float)

        # Predict (you trained on log(price))
        log_price = model.predict(final_df)[0]
        price = float(np.exp(log_price))
        st.success(f"Estimated Price: ${price:,.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# Option 2: Batch predictions via CSV upload
st.header("Batch Prediction via CSV Upload")
uploaded_file = st.file_uploader("Upload a CSV with original categorical columns (do NOT include encoded columns or Price)", type="csv")

if uploaded_file is not None:
    try:
        batch_raw = pd.read_csv(uploaded_file)

        # verify required categorical columns exist
        missing_cats = [c for c in categorical_columns if c not in batch_raw.columns]
        if missing_cats:
            st.error(f"Uploaded CSV missing required categorical columns: {missing_cats}")
        else:
            # Encode batch
            encoded_batch = ohe.transform(batch_raw[categorical_columns])
            encoded_batch_df = pd.DataFrame(encoded_batch, columns=ohe.get_feature_names_out(categorical_columns), index=batch_raw.index)

            # collect numeric cols present in upload
            numeric_cols_present = [c for c in ['RAM','Weight','HDD','SSD','ppi','Touchscreen','Ips'] if c in batch_raw.columns]
            numeric_batch = batch_raw[numeric_cols_present].reset_index(drop=True)

            final_batch = pd.concat([encoded_batch_df.reset_index(drop=True), numeric_batch], axis=1)

            model_feature_names = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else None
            if model_feature_names is not None:
                final_batch = final_batch.reindex(columns=model_feature_names, fill_value=0)

            final_batch = final_batch.astype(float)
            log_preds = model.predict(final_batch)
            preds = np.exp(log_preds)

            results = batch_raw.copy()
            results['Predicted_Price'] = preds
            st.dataframe(results)
            st.download_button("Download Predictions", results.to_csv(index=False), file_name="predictions.csv")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
