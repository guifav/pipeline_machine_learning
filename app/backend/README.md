# Backend API - Flight Delay Prediction

API RESTful construída com FastAPI para servir modelos de Machine Learning de previsão de atrasos de voos.

## Instalação

```bash
cd app/backend

# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

## Executar

```bash
# Desenvolvimento (com auto-reload)
uvicorn main:app --reload --port 8000

# Produção
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

A API estará disponível em: `http://localhost:8000`

## Documentação

Acesse a documentação interativa (Swagger UI):
- **Swagger**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### GET /health
Health check da API e modelos

```bash
curl http://localhost:8000/health
```

### GET /metadata
Informações sobre features e dataset

```bash
curl http://localhost:8000/metadata
```

### POST /predict
Fazer previsão de atraso para um voo

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "month": 6,
    "day": 15,
    "day_of_week": 3,
    "hour": 14,
    "distance": 1500,
    "scheduled_time": 180,
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
  }'
```

Resposta:
```json
{
  "will_delay": true,
  "delay_probability": 0.73,
  "estimated_delay_minutes": 25.5,
  "confidence": "High"
}
```

## Modelos

A API utiliza:
- **Classificador**: `best_classifier_no_leakage.pkl` (Random Forest) - Prevê SE vai atrasar
- **Regressor**: `best_regressor_notebook.pkl` (XGBoost) - Prevê QUANTO tempo vai atrasar

## Features Esperadas

A API espera 40 features na seguinte ordem (consulte `/metadata` para detalhes):

**Numéricas**: month, day, day_of_week, hour, distance, scheduled_time, airline_delay_rate, origin_delay_rate, dest_delay_rate, route_delay_rate, origin_latitude, origin_longitude, dest_latitude, dest_longitude

**Categóricas Encoded**: origin_airport, destination_airport, origin_state, dest_state

**Binárias**: is_weekend, is_morning_rush, is_evening_rush, is_domestic

**One-Hot Airlines**: airline_AS, airline_B6, airline_DL, airline_EV, airline_F9, airline_HA, airline_MQ, airline_NK, airline_OO, airline_UA, airline_US, airline_VX, airline_WN

**One-Hot Period**: period_of_day_Manha, period_of_day_Noite, period_of_day_Tarde

**One-Hot Season**: season_Outono, season_Primavera, season_Verao

## Métricas dos Modelos

### Classificador
- ROC-AUC: 0.75
- F1-Score: 0.44
- Recall: 0.67

### Regressor
- MAE: 17.80 minutos
- RMSE: 36.04 minutos
- R²: 0.057

## CORS

A API permite requisições de:
- `http://localhost:3000` (React CRA)
- `http://localhost:5173` (Vite)

Para adicionar outras origens, edite `main.py`:

```python
allow_origins=["http://localhost:3000", "http://seu-dominio.com"]
```
