import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.config import MODELS_DIR

def evaluate_model(model_name):
    model_path = MODELS_DIR / model_name
    print(f"--- Evaluando {model_name} ---")
    try:
        data = joblib.load(model_path)
        modelo = data['modelo']
        features = data['features']
        
        print(f"Tipo de Modelo: {type(modelo)}")
        print(f"Parámetros: {modelo.get_params()}")
        
        # Extraer importancia de características
        importances = modelo.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print("\nTop 10 Características Más Importantes:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Simulando una predicción (Oráculo o ANS)
        print("\nEjecutando predicción de prueba (dummy data)...")
        dummy_df = pd.DataFrame(np.zeros((1, len(features))), columns=features)
        pred = modelo.predict(dummy_df)
        print(f"Resultado de la predicción dummy: {pred}")
        print("\n" + "="*50 + "\n")
        
    except Exception as e:
        print(f"Error al evaluar {model_name}: {e}")

if __name__ == "__main__":
    evaluate_model("modelo_ans_xgboost.pkl")
    evaluate_model("modelo_oraculo_xgb.pkl")
