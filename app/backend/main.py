"""
FastAPI Backend - Flight Delay Prediction API
==============================================

API RESTful para prever atrasos de voos usando modelos de Machine Learning.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Inicializar FastAPI
app = FastAPI(
    title="Flight Delay Prediction API",
    description="API para prever atrasos de voos usando Machine Learning",
    version="1.0.0"
)

# Configurar CORS para permitir requisições do frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas as origens em desenvolvimento
    allow_credentials=False,  # Deve ser False quando allow_origins é "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminhos para modelos e metadata
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
METADATA_PATH = BASE_DIR / "data" / "model_ready" / "metadata.json"

# Carregar metadata
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

# Carregar modelos ao iniciar a aplicação
print("Carregando modelos...")
with open(MODELS_DIR / "best_classifier_no_leakage.pkl", 'rb') as f:
    classifier_model = pickle.load(f)

with open(MODELS_DIR / "best_regressor_notebook.pkl", 'rb') as f:
    regressor_model = pickle.load(f)

print("Modelos carregados com sucesso!")


# Schemas Pydantic para validação de entrada/saída
class FlightInput(BaseModel):
    """Schema para entrada de dados de voo."""

    # Temporais
    month: int = Field(..., ge=1, le=12, description="Mês (1-12)")
    day: int = Field(..., ge=1, le=31, description="Dia do mês (1-31)")
    day_of_week: int = Field(..., ge=1, le=7, description="Dia da semana (1=Segunda, 7=Domingo)")
    hour: int = Field(..., ge=0, le=23, description="Hora da partida programada (0-23)")

    # Operacionais
    distance: float = Field(..., gt=0, description="Distância do voo em milhas")
    scheduled_time: float = Field(..., gt=0, description="Tempo programado de voo em minutos")

    # Taxas históricas (0-1)
    airline_delay_rate: float = Field(..., ge=0, le=1, description="Taxa histórica de atraso da companhia")
    origin_delay_rate: float = Field(..., ge=0, le=1, description="Taxa histórica de atraso do aeroporto origem")
    dest_delay_rate: float = Field(..., ge=0, le=1, description="Taxa histórica de atraso do aeroporto destino")
    route_delay_rate: float = Field(..., ge=0, le=1, description="Taxa histórica de atraso da rota")

    # Geográficas
    origin_latitude: float = Field(..., ge=-90, le=90, description="Latitude do aeroporto origem")
    origin_longitude: float = Field(..., ge=-180, le=180, description="Longitude do aeroporto origem")
    dest_latitude: float = Field(..., ge=-90, le=90, description="Latitude do aeroporto destino")
    dest_longitude: float = Field(..., ge=-180, le=180, description="Longitude do aeroporto destino")

    # Categóricas (encoded)
    origin_airport: int = Field(..., description="Código do aeroporto origem (encoded)")
    destination_airport: int = Field(..., description="Código do aeroporto destino (encoded)")
    origin_state: int = Field(..., description="Estado origem (encoded)")
    dest_state: int = Field(..., description="Estado destino (encoded)")

    # Binárias
    is_weekend: int = Field(..., ge=0, le=1, description="É fim de semana? (0=Não, 1=Sim)")
    is_morning_rush: int = Field(..., ge=0, le=1, description="É horário de pico manhã? (0=Não, 1=Sim)")
    is_evening_rush: int = Field(..., ge=0, le=1, description="É horário de pico tarde? (0=Não, 1=Sim)")
    is_domestic: int = Field(..., ge=0, le=1, description="É voo doméstico? (0=Não, 1=Sim)")

    # One-hot encoded airlines (13 companhias)
    airline_AS: int = Field(0, ge=0, le=1)
    airline_B6: int = Field(0, ge=0, le=1)
    airline_DL: int = Field(0, ge=0, le=1)
    airline_EV: int = Field(0, ge=0, le=1)
    airline_F9: int = Field(0, ge=0, le=1)
    airline_HA: int = Field(0, ge=0, le=1)
    airline_MQ: int = Field(0, ge=0, le=1)
    airline_NK: int = Field(0, ge=0, le=1)
    airline_OO: int = Field(0, ge=0, le=1)
    airline_UA: int = Field(0, ge=0, le=1)
    airline_US: int = Field(0, ge=0, le=1)
    airline_VX: int = Field(0, ge=0, le=1)
    airline_WN: int = Field(0, ge=0, le=1)

    # One-hot encoded period_of_day (3 categorias)
    period_of_day_Manha: int = Field(0, ge=0, le=1)
    period_of_day_Noite: int = Field(0, ge=0, le=1)
    period_of_day_Tarde: int = Field(0, ge=0, le=1)

    # One-hot encoded season (3 categorias)
    season_Outono: int = Field(0, ge=0, le=1)
    season_Primavera: int = Field(0, ge=0, le=1)
    season_Verao: int = Field(0, ge=0, le=1)

    class Config:
        schema_extra = {
            "example": {
                "month": 6,
                "day": 15,
                "day_of_week": 3,
                "hour": 14,
                "distance": 1500.0,
                "scheduled_time": 180.0,
                "airline_delay_rate": 0.15,
                "origin_delay_rate": 0.18,
                "dest_delay_rate": 0.20,
                "route_delay_rate": 0.17,
                "origin_latitude": 33.9425,
                "origin_longitude": -118.4081,
                "dest_latitude": 40.7769,
                "dest_longitude": -73.8740,
                "origin_airport": 150,
                "destination_airport": 200,
                "origin_state": 5,
                "dest_state": 32,
                "is_weekend": 0,
                "is_morning_rush": 0,
                "is_evening_rush": 1,
                "is_domestic": 1,
                "airline_DL": 1,
                "period_of_day_Tarde": 1,
                "season_Verao": 1
            }
        }


class PredictionResponse(BaseModel):
    """Schema para resposta de previsão."""

    will_delay: bool = Field(..., description="O voo vai atrasar?")
    delay_probability: float = Field(..., description="Probabilidade de atraso (0-1)")
    estimated_delay_minutes: Optional[float] = Field(None, description="Tempo estimado de atraso em minutos")
    confidence: str = Field(..., description="Nível de confiança da previsão")

    class Config:
        schema_extra = {
            "example": {
                "will_delay": True,
                "delay_probability": 0.73,
                "estimated_delay_minutes": 25.5,
                "confidence": "High"
            }
        }


class HealthResponse(BaseModel):
    """Schema para resposta de health check."""
    status: str
    models_loaded: bool
    classifier_type: str
    regressor_type: str


def prepare_features(flight_input: FlightInput, model_type: str = 'classifier') -> pd.DataFrame:
    """
    Prepara features na ordem exata esperada pelo modelo.

    Args:
        flight_input: Dados do voo validados
        model_type: 'classifier' ou 'regressor'

    Returns:
        DataFrame com features na ordem correta
    """
    # Converter o FlightInput para dicionário
    input_dict = flight_input.dict()

    # Escolher as features esperadas pelo modelo
    if model_type == 'classifier':
        expected_features = classifier_model.feature_names_in_
    else:
        expected_features = regressor_model.feature_names_in_

    # Criar dicionário com todas as features na ordem do modelo
    feature_dict = {}
    for feature_name in expected_features:
        # Buscar o valor no input_dict (case-insensitive)
        value = None
        for key in input_dict:
            if key.upper() == feature_name.upper():
                value = input_dict[key]
                break

        # Se não encontrou, usar 0 como padrão
        if value is None:
            value = 0

        feature_dict[feature_name] = [value]

    return pd.DataFrame(feature_dict)


@app.get("/", tags=["General"])
async def root():
    """Endpoint raiz - Informações da API."""
    return {
        "message": "Flight Delay Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "metadata": "/metadata"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check - Verifica se a API e os modelos estão funcionando.
    """
    return {
        "status": "healthy",
        "models_loaded": True,
        "classifier_type": type(classifier_model).__name__,
        "regressor_type": type(regressor_model).__name__
    }


@app.get("/metadata", tags=["General"])
async def get_metadata():
    """
    Retorna metadata do dataset e features esperadas.
    """
    return {
        "dataset_info": metadata['dataset_info'],
        "features": metadata['features'],
        "targets": metadata['targets'],
        "splits": metadata['splits']
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_delay(flight: FlightInput):
    """
    Faz previsão de atraso para um voo.

    Args:
        flight: Dados do voo para previsão

    Returns:
        Previsão de atraso com probabilidade e tempo estimado
    """
    try:
        # Preparar features para o classificador (40 features)
        X_classifier = prepare_features(flight, model_type='classifier')
        print(f"Features classificador: {X_classifier.shape}")

        # Classificação: Vai atrasar?
        will_delay = bool(classifier_model.predict(X_classifier)[0])
        delay_proba = float(classifier_model.predict_proba(X_classifier)[0, 1])

        # Regressão: Quanto tempo?
        estimated_delay = None
        if will_delay:
            # Preparar features para o regressor (41 features - inclui AIRLINE_DELAY_RATE)
            X_regressor = prepare_features(flight, model_type='regressor')
            print(f"Features regressor: {X_regressor.shape}")
            estimated_delay = float(regressor_model.predict(X_regressor)[0])
            # Garantir que seja positivo
            estimated_delay = max(0, estimated_delay)

        # Calcular nível de confiança
        if delay_proba >= 0.8 or delay_proba <= 0.2:
            confidence = "High"
        elif delay_proba >= 0.6 or delay_proba <= 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            "will_delay": will_delay,
            "delay_probability": round(delay_proba, 4),
            "estimated_delay_minutes": round(estimated_delay, 2) if estimated_delay else None,
            "confidence": confidence
        }

    except Exception as e:
        import traceback
        print(f"ERRO: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erro na previsão: {str(e)}")


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(flights: list[FlightInput]):
    """
    Faz previsão para múltiplos voos.

    Args:
        flights: Lista de voos para previsão

    Returns:
        Lista de previsões
    """
    try:
        predictions = []
        for flight in flights:
            pred = await predict_delay(flight)
            predictions.append(pred)

        return {
            "total": len(predictions),
            "predictions": predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na previsão em lote: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
