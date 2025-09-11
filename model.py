import joblib
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

def load_model(model_path='checkpoints/lgbm_model.pkl'):
    model = joblib.load(model_path)
    return model


def predict(model, data):
    '''
    Make prediction using the loaded model and preprocessed data.
    Display the prediction result with appropriate styling.
    '''
    prediction = model.predict(data)

    console = Console()

    risk_map = {
        0: ("No Heart Disease", 0.0, "green", "✅"),
        1: ("Small Risk of Heart Disease", 0.25, "green", "🟢"),
        2: ("Medium Risk of Heart Disease", 0.50, "yellow", "⚠️"),
        3: ("High Risk of Heart Disease", 0.75, "red", "❌"),
        4: ("Very High Risk of Heart Disease", 1.00, "red", "🔥"),
    }

    label, prob, color, icon = risk_map[prediction[0]]

    text = Text()
    text.append(f"{icon} {label}\n", style=f"bold {color}")
    text.append(f"Probability: {int(prob * 100)}%\n")

    console.print()
    console.print(Panel(text, title="Prediction", title_align="left", border_style=color))