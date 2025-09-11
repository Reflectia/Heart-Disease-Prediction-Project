import numpy as np
import pandas as pd 
import joblib
from rich.console import Console
from rich.table import Table

console = Console()

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


def get_int(prompt, valid_range=None, allow_empty=False, default=None, type=int):
    '''
    Helper function to get validated integer input from the user.
    '''
    while True:
        val = input(prompt)
        if allow_empty and val == '':
            return default
        elif val == 'exit':
            console.print("Exiting the program.", style="red")
            exit(0)
        try:
            val = type(val)
            if val < valid_range[0] or val > valid_range[1]:
                print(f"Please enter a value in range {valid_range}")
            else:
                return val
        except ValueError:
            print("Please enter a valid value.")


def get_user_input():

    console.print("\n[bold cyan]💓 Heart Risk Predictor[/bold cyan]\n", justify="center")
    
    console.print("[italic]Press [dim]Enter[/dim] to skip optional fields[italic]")
    console.print("[italic]Write 'exit' if you want to leave the program[italic]\n")

    console.print("Please enter the following patient data:", style="yellow")

    console.rule("[bold blue] Patient Information (required) [/bold blue]")

    # Required inputs
    age = get_int("Age (0-99): ", valid_range=(0, 99), allow_empty=False)

    dataset = get_int("Dataset (0=Cleveland, 1=Hungary, 2=Switzerland, 3=VA Long Beach): ", valid_range=(0, 3), allow_empty=False)

    sex = get_int("Sex (0=Female, 1=Male): ", valid_range=(0, 1), allow_empty=False)

    cp = get_int("Chest pain type (0=typical angina, 1=atypical angina, 2=non-anginal, 3=asymptomatic): ", valid_range=(0, 3), allow_empty=False)

    console.rule("[bold blue] Medical Measurements (optional) [/bold blue]")

    # Optional inputs    
    trestbps = get_int("Resting blood pressure (0-250 mm Hg or skip): ", valid_range=(0, 250), allow_empty=True, default=np.nan)

    chol = get_int("Serum cholesterol (0-650 mg/dl or skip): ", valid_range=(0, 650), allow_empty=True, default=np.nan)

    fbs = get_int("Fasting blood sugar > 120 mg/dl (1=True, 0=False or skip): ", valid_range=(0, 1), allow_empty=True, default=None)

    restecg = get_int("Resting electrocardiographic results (0=normal, 1=st-t abnormality, 2=lv hypertrophy or skip): ", valid_range=(0, 2), 
                      allow_empty=True, default=None)

    thalch = get_int("Maximum heart rate achieved (0-250 or skip): ", valid_range=(0, 250), allow_empty=True, default=np.nan)

    exang = get_int("Exercise-induced angina (1=Yes, 0=No or skip): ", valid_range=(0, 1), allow_empty=True, default=None)

    oldpeak = get_int("ST depression induced by exercise relative to rest (from -3 to 7 (float) or skip): ", valid_range=(-3, 7), 
                    allow_empty=True, default=np.nan, type=float)

    slope = get_int("Slope of the peak exercise ST segment (0=downsloping, 1=flat, 2=upsloping or skip): ", valid_range=(0, 2), 
                    allow_empty=True, default=None)

    ca = get_int("Number of major vessels colored by fluoroscopy (0-3 or skip) : ", valid_range=(0, 3), allow_empty=True, default=np.nan)

    thal = get_int("Thalassemia (0=normal, 1=fixed defect, 2=reversible defect or skip): ", valid_range=(0, 2), 
                   allow_empty=True, default=None)

    # Build a nice summary table
    table = Table(title="User Input Summary", show_lines=True, header_style="bold magenta")

    data = {
        "Age": age, "Dataset": dataset, "Sex": sex, "Chest Pain": cp,
        "Resting BP": trestbps, "Cholesterol": chol, "Fasting BS": fbs,
        "Resting ECG": restecg, "Max Heart Rate": thalch,
        "Exercise Angina": exang, "ST Depression": oldpeak,
        "Slope": slope, "Vessels": ca, "Thalassemia": thal
    }

    for k in data.keys():
        table.add_column(k, style="cyan", no_wrap=True)

    table.add_row(*[str(v) for v in data.values()])

    console.print("\n")
    console.print(table)

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
        
    # Handle missing values
    input_df['trestbps'] = input_df['trestbps'].fillna(TRESTBPS_MEAN)
    input_df['chol'] = input_df['chol'].fillna(CHOL_MEAN)
    input_df['thalch'] = input_df['thalch'].fillna(THALCH_MEAN)
    input_df['oldpeak'] = input_df['oldpeak'].fillna(OLDPEAK_MEAN)

    input_df['ca'] = input_df['ca'].fillna(0)

    for col in ['fbs', 'exang']:
       input_df[col] = input_df[col].fillna(False)

    # One-hot encode categorical variables
    input_df_encoded = pd.get_dummies(input_df, columns=['dataset', 'cp', 'restecg', 'slope', 'thal'], dtype=int)

    # Align with training columns
    input_df_encoded = input_df_encoded.reindex(columns=ONEHOT_ENCODED_COLS, fill_value=0)

    # Label encode binary categorical variables
    le = joblib.load('checkpoints/label_encoder.pkl')

    for col in ['sex', 'fbs', 'exang']:
        input_df_encoded[col] = le.transform(input_df_encoded[col])

    console.print("Data was preprocessed", style="yellow")

    return input_df_encoded


