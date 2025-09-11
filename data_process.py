import numpy as np
import pandas as pd 
import joblib

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated*")

TRESTBPS_MEAN = 132.2860465116279
CHOL_MEAN = 246.83286908077994
THALCH_MEAN = 137.5456647398844
OLDPEAK_MEAN = 0.8787878787878788

ONEHOT_ENCODED_COLS = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalch', 'exang', 'oldpeak',
    'ca', 'dataset_Cleveland', 'dataset_Hungary',
    'dataset_Switzerland', 'dataset_VA Long Beach', 'cp_asymptomatic',
    'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
    'restecg_lv hypertrophy', 'restecg_normal', 'restecg_st-t abnormality',
    'slope_downsloping', 'slope_flat', 'slope_upsloping',
    'thal_fixed defect', 'thal_normal', 'thal_reversable defect']


def get_user_input():
    print("Please enter the following patient data:")
    age = int(input("Age (0-99): "))
    if age < 0 or age > 99:
        raise ValueError("Age must be between 0 and 99.")
    
    dataset = int(input("Dataset (0=Cleveland, 1=Hungary, 2=Switzerland, 3=VA Long Beach): "))
    if dataset < 0 or dataset > 3:
        raise ValueError("Dataset value must be between 0 and 3.")
    dataset = {0: 'Cleveland', 1: 'Hungary', 2: 'Switzerland', 3: 'VA Long Beach'}[dataset]
    
    sex = int(input("Sex (0=Female, 1=Male): "))
    if sex not in [0, 1]:
        raise ValueError("Sex value must be 0 or 1.")
    sex = {0: 'Female', 1: 'Male'}[sex]
    
    cp = int(input("Chest pain type (0=typical angina, 1=atypical angina, 2=non-anginal, 3=asymptomatic): "))
    if cp < 0 or cp > 3:
        raise ValueError("Chest pain type value must be between 0 and 3.")
    cp = {0: 'typical angina', 1: 'atypical angina', 2: 'non-anginal', 3: 'asymptomatic'}[cp]

    trestbps = input("Resting blood pressure (0-250 mm Hg or ''): ")
    if trestbps == '':
        trestbps = np.nan
    else:
        trestbps = int(trestbps)
        if trestbps < 0 or trestbps > 250:
            raise ValueError("Resting blood pressure must be between 0 and 250 mm Hg.")

    chol = input("Serum cholesterol (0-650 mg/dl or ''): ")
    if chol == '':
        chol = np.nan
    else:
        chol = int(chol)
        if chol < 0 or chol > 650:
            raise ValueError("Serum cholesterol must be between 0 and 650 mg/dl.")
    
    fbs = input("Fasting blood sugar > 120 mg/dl (1=True, 0=False or ''): ")
    if fbs == '':
        fbs = None
    else:
        fbs = int(fbs)
        if fbs not in [0, 1]:
            raise ValueError("Fasting blood sugar value must be 0 or 1.")
        fbs = {0: False, 1: True}[fbs]
    
    restecg = input("Resting electrocardiographic results (0=normal, 1=st-t abnormality, 2=lv hypertrophy or ''): ")
    if restecg == '':
        restecg = None
    else:
        restecg = int(restecg)
        if restecg < 0 or restecg > 2:
            raise ValueError("Resting electrocardiographic results must be between 0 and 2.")
        restecg = {0: 'normal', 1: 'st-t abnormality', 2: 'lv hypertrophy'}[restecg]
    
    thalch = input("Maximum heart rate achieved (0-250 or ''): ")
    if thalch == '':
        thalch = np.nan
    else:
        thalch = int(thalch)
        if thalch < 0 or thalch > 250:
            raise ValueError("Maximum heart rate must be between 0 and 250.")
    
    exang = input("Exercise-induced angina (1=Yes, 0=No or ''): ")
    if exang == '':
        exang = None
    else:
        exang = int(exang)
        if exang not in [0, 1]:
            raise ValueError("Exercise-induced angina value must be 0 or 1.")
        exang = {0: False, 1: True}[exang]
    
    oldpeak = input("ST depression induced by exercise relative to rest (from -3 to 7 or '', float): ")
    if oldpeak == '':
        oldpeak = np.nan
    else:
        oldpeak = float(oldpeak)
        if oldpeak < -3 or oldpeak > 7:
            raise ValueError("ST depression value must be between -3 and 7.")
    
    slope = input("Slope of the peak exercise ST segment (0=downsloping, 1=flat, 2=upsloping or ''): ")
    if slope == '':
        slope = None
    else:
        slope = int(slope)
        if slope < 0 or slope > 2:
            raise ValueError("Slope must be between 0 and 2.")
        slope = {0: 'downsloping', 1: 'flat', 2: 'upsloping'}[slope]
    
    ca = input("Number of major vessels colored by fluoroscopy (0-3 or '') : ")
    if ca == '':
        ca = np.nan
    else:
        ca = int(ca)
        if ca < 0 or ca > 3:
            raise ValueError("Number of major vessels must be between 0 and 3.")
    
    thal = input("Thalassemia (0=normal, 1=fixed defect, 2=reversible defect or ''): ")
    if thal == '':
        thal = None
    else:
        thal = int(thal)
        if thal < 0 or thal > 2:
            raise ValueError("Thalassemia value must be between 0 and 2.")
        thal = {0: 'normal', 1: 'fixed defect', 2: 'reversable defect'}[thal]

    return {
        "age": age,
        "dataset": dataset,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }


def preprocess_data(data):

    input_df = pd.DataFrame([data])
    print("User input:\n", input_df)
        
    input_df['trestbps'] = input_df['trestbps'].fillna(TRESTBPS_MEAN)
    input_df['chol'] = input_df['chol'].fillna(CHOL_MEAN)
    input_df['thalch'] = input_df['thalch'].fillna(THALCH_MEAN)
    input_df['oldpeak'] = input_df['oldpeak'].fillna(OLDPEAK_MEAN)

    input_df['ca'] = input_df['ca'].fillna(0)

    for col in ['fbs', 'exang']:
       input_df[col] = input_df[col].fillna(False)

    input_df_encoded = pd.get_dummies(input_df, columns=['dataset', 'cp', 'restecg', 'slope', 'thal'], dtype=int)

    # Align with training columns
    input_df_encoded = input_df_encoded.reindex(columns=ONEHOT_ENCODED_COLS, fill_value=0)

    le = joblib.load('checkpoints/label_encoder.pkl')

    for col in ['sex', 'fbs', 'exang']:
        input_df_encoded[col] = le.transform(input_df_encoded[col])

    print("Data was preprocessed")

    return input_df_encoded
    

