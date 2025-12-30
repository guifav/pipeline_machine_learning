"""
Script para verificar os nomes das features esperados pelos modelos
"""
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

# Carregar classificador
with open(MODELS_DIR / "best_classifier_no_leakage.pkl", 'rb') as f:
    classifier_model = pickle.load(f)

# Carregar regressor
with open(MODELS_DIR / "best_regressor_notebook.pkl", 'rb') as f:
    regressor_model = pickle.load(f)

# Verificar feature names do classificador
if hasattr(classifier_model, 'feature_names_in_'):
    print("=" * 60)
    print("CLASSIFICADOR - Feature names esperados:")
    print("=" * 60)
    for i, name in enumerate(classifier_model.feature_names_in_):
        print(f"{i+1}. {name}")
    print(f"\nTotal: {len(classifier_model.feature_names_in_)} features")
else:
    print("Classificador não tem feature_names_in_")

print("\n")

# Verificar feature names do regressor
if hasattr(regressor_model, 'feature_names'):
    print("=" * 60)
    print("REGRESSOR - Feature names esperados:")
    print("=" * 60)
    for i, name in enumerate(regressor_model.feature_names):
        print(f"{i+1}. {name}")
    print(f"\nTotal: {len(regressor_model.feature_names)} features")
elif hasattr(regressor_model, 'feature_names_in_'):
    print("=" * 60)
    print("REGRESSOR - Feature names esperados (feature_names_in_):")
    print("=" * 60)
    for i, name in enumerate(regressor_model.feature_names_in_):
        print(f"{i+1}. {name}")
    print(f"\nTotal: {len(regressor_model.feature_names_in_)} features")
else:
    print("Regressor não tem feature_names ou feature_names_in_")
