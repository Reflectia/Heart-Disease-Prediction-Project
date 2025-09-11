from data_process import get_user_input, preprocess_data
from model import load_model, predict


if __name__ == "__main__":
    user_data = get_user_input()
    
    preprocessed_data = preprocess_data(user_data)

    model = load_model()

    predict(model, preprocessed_data)