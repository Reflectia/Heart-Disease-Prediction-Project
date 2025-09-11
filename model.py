import joblib

def load_model(model_path='checkpoints/lgbm_model.pkl'):
    model = joblib.load(model_path)
    return model

def predict(model, data):
    prediction = model.predict(data)

    if prediction[0] == 0:
        print("Prediction: No heart disease")
    elif prediction[0] == 1:
        print(f"Prediction: Small Risk of Heart Disease (Probability: 25%)")
    elif prediction[0] == 2:
        print(f"Prediction: Medium Risk of Heart Disease (Probability: 50%)")  
    elif prediction[0] == 3:
        print(f"Prediction: High Risk of Heart Disease (Probability: 75%)")
    elif prediction[0] == 4:
        print(f"Prediction: Very High Risk of Heart Disease (Probability: 100%)")