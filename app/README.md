# Flight Delay Prediction Application

Aplicação web completa para previsão de atrasos de voos usando modelos de Machine Learning.

## Arquitetura

```
app/
├── backend/          # API FastAPI (Python)
│   ├── main.py      # Servidor API
│   ├── requirements.txt
│   └── README.md
│
└── frontend/        # Interface React (TypeScript)
    ├── src/
    │   ├── App.tsx  # Componente principal
    │   └── App.css  # Estilos
    ├── package.json
    └── README.md
```

## Tecnologias

### Backend
- **FastAPI** - Framework web Python moderno e rápido
- **Pydantic** - Validação de dados com tipos
- **Scikit-learn / XGBoost** - Modelos de ML
- **Uvicorn** - Servidor ASGI

### Frontend
- **React 18** - Biblioteca UI
- **TypeScript** - Type safety
- **Vite** - Build tool
- **CSS3** - Estilização moderna

## Início Rápido

### 1. Backend

```bash
cd app/backend

# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar API
uvicorn main:app --reload
```

API disponível em: `http://localhost:8000`
Documentação: `http://localhost:8000/docs`

### 2. Frontend

```bash
cd app/frontend

# Instalar dependências
npm install

# Executar aplicação
npm run dev
```

Aplicação disponível em: `http://localhost:5173`

## Modelos Utilizados

### Classificador
- **Arquivo**: `best_classifier_no_leakage.pkl` (301MB)
- **Algoritmo**: Random Forest
- **Objetivo**: Prever SE o voo vai atrasar (sim/não)
- **Métricas**:
  - ROC-AUC: 0.75
  - F1-Score: 0.44
  - Recall: 0.67

### Regressor
- **Arquivo**: `best_regressor_notebook.pkl` (248KB)
- **Algoritmo**: XGBoost
- **Objetivo**: Prever QUANTO tempo o voo vai atrasar (minutos)
- **Métricas**:
  - MAE: 17.80 minutos
  - RMSE: 36.04 minutos
  - R²: 0.057

## API Endpoints

### GET /health
Verifica status da API e modelos

```bash
curl http://localhost:8000/health
```

### GET /metadata
Retorna informações sobre features e dataset

```bash
curl http://localhost:8000/metadata
```

### POST /predict
Faz previsão de atraso para um voo

**Request**:
```json
{
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
  ...
}
```

**Response**:
```json
{
  "will_delay": true,
  "delay_probability": 0.73,
  "estimated_delay_minutes": 25.5,
  "confidence": "High"
}
```

## Features Esperadas

A API espera 40 features:

**Numéricas** (14):
- month, day, day_of_week, hour
- distance, scheduled_time
- airline_delay_rate, origin_delay_rate, dest_delay_rate, route_delay_rate
- origin_latitude, origin_longitude, dest_latitude, dest_longitude

**Categóricas Encoded** (4):
- origin_airport, destination_airport, origin_state, dest_state

**Binárias** (4):
- is_weekend, is_morning_rush, is_evening_rush, is_domestic

**One-Hot Airlines** (13):
- airline_AS, airline_B6, airline_DL, airline_EV, airline_F9, airline_HA, airline_MQ, airline_NK, airline_OO, airline_UA, airline_US, airline_VX, airline_WN

**One-Hot Period** (3):
- period_of_day_Manha, period_of_day_Noite, period_of_day_Tarde

**One-Hot Season** (3):
- season_Outono, season_Primavera, season_Verao

## Fluxo de Previsão

1. **Usuário preenche formulário** no frontend com dados do voo
2. **Frontend envia POST** para `/predict` com todas as 40 features
3. **Backend valida** entrada usando Pydantic schemas
4. **Backend prepara** features na ordem exata do modelo
5. **Classificador prevê** se vai atrasar (sim/não) com probabilidade
6. **Regressor prevê** quanto tempo vai atrasar (se sim)
7. **Backend calcula** nível de confiança baseado na probabilidade
8. **Frontend exibe** resultado visual com status, tempo e confiança

## Deploy

### Backend (Opções)

**Docker**:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Heroku / Render / Railway**:
- Adicionar `Procfile`: `web: uvicorn main:app --host 0.0.0.0 --port $PORT`
- Configurar build com Python 3.11+

### Frontend (Opções)

**Vercel / Netlify**:
```bash
npm run build
# Deploy pasta dist/
```

**Docker**:
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0"]
```

## Desenvolvimento

### Testar Backend Isoladamente

```bash
# Terminal 1: Rodar API
cd app/backend
uvicorn main:app --reload

# Terminal 2: Testar com curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @test_flight.json
```

### Testar Frontend Isoladamente

Mock da API no código se backend não estiver disponível:

```typescript
// App.tsx
const handlePredict = async () => {
  // Mock para desenvolvimento
  const mockResponse = {
    will_delay: true,
    delay_probability: 0.73,
    estimated_delay_minutes: 25.5,
    confidence: "High"
  };
  setPrediction(mockResponse);
};
```

## Troubleshooting

### Backend não inicia
- Verifique se os modelos existem em `../../models/`
- Confirme que metadata.json existe em `../../data/model_ready/`
- Instale todas as dependências: `pip install -r requirements.txt`

### Frontend não conecta
- Verifique se backend está rodando em `http://localhost:8000`
- Teste `/health`: `curl http://localhost:8000/health`
- Verifique CORS no console do navegador
- Confirme que `allow_origins` inclui `http://localhost:5173`

### Previsões inconsistentes
- Verifique se todas as 40 features estão sendo enviadas
- Confirme que one-hot encoding está correto (apenas 1 bit ativo por grupo)
- Valide ranges (ex: month 1-12, taxas 0-1)

## Performance

### Backend
- Carregamento de modelos: ~2-3s na inicialização
- Tempo de previsão: ~50-100ms por voo
- Memória: ~500MB (modelos carregados)

### Frontend
- Bundle size: ~150KB gzipped
- First load: ~500ms
- Interação: Instantânea

## Segurança

### Recomendações para Produção

1. **HTTPS obrigatório** em produção
2. **Rate limiting** na API (ex: 100 req/min por IP)
3. **Validação rigorosa** de todas as entradas
4. **CORS restrito** apenas para domínios confiáveis
5. **Logging** de todas as previsões
6. **Monitoramento** de drift do modelo
7. **Autenticação** (JWT) se necessário

### Exemplo de Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict_delay(request: Request, flight: FlightInput):
    ...
```

## Monitoramento

### Métricas a Acompanhar

1. **Performance**:
   - Latência média de previsão
   - Throughput (req/s)
   - Taxa de erro

2. **Modelo**:
   - Distribuição de probabilidades
   - Taxa de previsões positivas/negativas
   - Drift de features

3. **Negócio**:
   - Voos mais consultados
   - Horários de pico
   - Rotas populares

### Ferramentas Sugeridas

- **Logging**: Loguru, structlog
- **Monitoring**: Prometheus + Grafana
- **APM**: New Relic, DataDog
- **Error Tracking**: Sentry

## Contribuindo

Para adicionar novas features:

1. **Backend**: Atualizar `FlightInput` schema em `main.py`
2. **Modelo**: Re-treinar com novas features nos notebooks
3. **Frontend**: Adicionar campos no formulário
4. **Testes**: Validar end-to-end

## Licença

Projeto educacional - Livre para uso e modificação

---

**Desenvolvido com foco em boas práticas de ML em produção**
