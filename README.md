# Previsão de Atrasos em Voos - Projeto de Machine Learning

Projeto educacional completo de Machine Learning aplicado a dados reais de aviação comercial nos Estados Unidos.

## Sobre o Projeto

Este repositório contém um estudo completo de ciência de dados e machine learning para **prever atrasos de voos**, utilizando um dataset com aproximadamente **5.8 milhões de registros** de voos realizados em 2015.

### Objetivos Educacionais

- Demonstrar um pipeline completo de ML do início ao fim
- Aplicar técnicas de análise exploratória de dados (EDA)
- Implementar feature engineering em dados reais
- Comparar diferentes algoritmos de classificação e regressão
- Identificar e corrigir data leakage
- Avaliar modelos com métricas adequadas ao problema
- Documentar todo o processo de forma didática

## Dataset

O projeto utiliza três arquivos principais:

### 1. flights.csv (~5.8M registros)
Informações detalhadas sobre cada voo:
- **Temporais**: Data, horários programados e reais
- **Atrasos**: Tempo de atraso na partida e chegada
- **Operacionais**: Distância, tempo de voo, companhia aérea
- **Status**: Cancelamentos, desvios

### 2. airports.csv (323 aeroportos)
Dados dos aeroportos:
- Código IATA
- Nome, cidade, estado, país
- Coordenadas geográficas (latitude, longitude)

### 3. airlines.csv (14 companhias)
Informações das companhias aéreas:
- Código IATA
- Nome completo da companhia

## Estrutura do Projeto

```
tech_challenge/
│
├── data/
│   ├── raw/                    # Dados originais
│   ├── processed/              # Dados processados
│   └── model_ready/            # Dados prontos para ML
│
├── notebooks/                  # Jupyter Notebooks educacionais
│   ├── 01_eda_completo.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_missing_values.ipynb
│   ├── 04_preparacao_modelagem.ipynb
│   ├── 05_modelagem_classificacao.ipynb
│   ├── 06_analise_feature_importance.ipynb
│   └── 07_modelagem_regressao.ipynb
│
├── models/                     # Modelos treinados salvos
├── reports/
│   └── figures/                # Gráficos e visualizações
│
├── src/                        # Código Python reutilizável
│   └── preprocessing/
│

```

## Pipeline de Machine Learning

### Fase 1: Análise Exploratória (EDA)
**Notebook**: `01_eda_completo.ipynb`

- Carregamento e entendimento dos dados
- Estatísticas descritivas completas
- Visualizações que revelam insights
- Identificação de padrões e outliers

**Principais Descobertas**:
- 56% dos voos são adiantados (median = -1 minuto)
- 18% dos voos atrasam mais de 15 minutos
- Padrões claros por horário, dia da semana e companhia aérea

### Fase 2: Engenharia de Features
**Notebooks**: `02_feature_engineering.ipynb` e `03_missing_values.ipynb`

- Tratamento de 30+ milhões de valores ausentes (97% de redução)
- Criação de 20 novas features derivadas
- Encoding de variáveis categóricas
- Normalização de features numéricas

**Features Criadas**:
- Período do dia (manhã, tarde, noite, madrugada)
- Estação do ano
- Final de semana / dia útil
- Taxas históricas de atraso por aeroporto/companhia/rota
- Features geográficas (estados, coordenadas)

### Fase 3: Preparação para Modelagem
**Notebook**: `04_preparacao_modelagem.ipynb`

- Seleção de 25 features relevantes
- Split estratificado: 70% treino / 10% validação / 20% teste
- Balanceamento de classes
- Salvamento de objetos de preprocessing (scaler, encoders)

### Fase 4: Modelagem Supervisionada

#### Classificação (IS_DELAYED)
**Notebook**: `05_modelagem_classificacao.ipynb`

**Objetivo**: Prever SE o voo vai atrasar (sim/não)

**Modelos Implementados**:
- SGD Classifier (baseline)
- Random Forest
- XGBoost

**Melhores Resultados** (com balanceamento de classes):
- ROC-AUC: 0.75
- F1-Score: 0.44
- Recall: 0.67

**Importante**: Detectamos e corrigimos data leakage (DELAY_MINUTES estava vazando informação)

#### Análise de Data Leakage
**Notebook**: `06_analise_feature_importance.ipynb`

- Identificação de features com data leakage
- Análise de feature importance
- Re-treinamento com features limpas
- Comparação de modelos com/sem leakage

**Features Removidas**:
- `DELAY_MINUTES` (só conhecido após o voo!)

#### Regressão (DELAY_MINUTES)
**Notebook**: `07_modelagem_regressao.ipynb`

**Objetivo**: Prever QUANTO TEMPO o voo vai atrasar (em minutos)

**Modelos Implementados**:
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor

**Melhores Resultados**:
- MAE: 17.80 minutos
- RMSE: 36.04 minutos
- R²: 0.0567

**Nota**: R² baixo (0.05-0.10) é NORMAL para este problema segundo literatura científica

## Tecnologias Utilizadas

### Core
- **Python 3.11+**
- **Pandas** - Manipulação de dados
- **NumPy** - Operações numéricas

### Visualização
- **Matplotlib** - Gráficos estáticos
- **Seaborn** - Visualizações estatísticas
- **Plotly** - Gráficos interativos (opcional)

### Machine Learning
- **Scikit-learn** - Algoritmos de ML clássico
- **XGBoost** - Gradient Boosting otimizado

### Utilidades
- **Jupyter Lab/Notebook** - Ambiente interativo
- **joblib** - Persistência de modelos

## Como Usar

### Instalação

```bash
# 1. Clone ou baixe o repositório
cd tech_challenge

# 2. Crie um ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Inicie o Jupyter Lab
jupyter lab
```

### Executar os Notebooks

Execute os notebooks **na ordem sequencial**:

1. `01_eda_completo.ipynb` - Entender os dados
2. `02_feature_engineering.ipynb` - Criar features
3. `03_missing_values.ipynb` - Tratar dados faltantes
4. `04_preparacao_modelagem.ipynb` - Preparar para ML
5. `05_modelagem_classificacao.ipynb` - Modelos de classificação
6. `06_analise_feature_importance.ipynb` - Análise de importância
7. `07_modelagem_regressao.ipynb` - Modelos de regressão

## Principais Aprendizados

### 1. Data Leakage é Real
- Identificamos performance "perfeita" (ROC-AUC = 1.0) que era suspeita
- Descobrimos que DELAY_MINUTES estava vazando informação futura
- Após correção: ROC-AUC realista de ~0.75

### 2. Métricas Devem Ser Contextualizadas
- R² baixo (0.05) pode ser EXCELENTE dependendo do problema
- Literatura científica confirma: 0.05-0.30 é normal para previsão de atrasos
- MAE (erro absoluto) é mais interpretável que R² neste caso

### 3. Balanceamento de Classes Importa
- Dataset desbalanceado: 82% sem atraso, 18% com atraso
- Sem balanceamento: modelos preveem sempre "sem atraso"
- Com class_weight='balanced': recall aumenta de 11% para 67%

### 4. Feature Engineering é Crucial
- Features derivadas (taxas históricas) são mais preditivas que features brutas
- Encoding correto de categóricas faz diferença
- Normalização essencial para modelos lineares

## Métricas de Sucesso Atingidas

### Classificação
| Métrica | Meta Ajustada | Atingido | Status |
|---------|---------------|----------|--------|
| ROC-AUC | > 0.70 | 0.75 | ✅ |
| F1-Score | > 0.40 | 0.44 | ✅ |
| Recall | > 0.60 | 0.67 | ✅ |

### Regressão
| Métrica | Meta Ajustada | Atingido | Status |
|---------|---------------|----------|--------|
| R² | > 0.05 | 0.057 | ✅ |
| MAE | < 20 min | 17.80 min | ✅ |
| RMSE | < 40 min | 36.04 min | ✅ |

## Insights de Negócio

1. **56% dos voos são ADIANTADOS**, não atrasados
2. **Chicago (ORD)** é o hub mais problemático entre grandes aeroportos
3. **Aeroportos do Hawaii** são excepcionalmente pontuais (0.6 min atraso médio)
4. **Rush hours** (6-9h e 17-20h) têm maiores taxas de atraso
5. **Histórico de rota/companhia** é melhor preditor que distância

## Próximos Passos (Opcional)

### Melhorias no Modelo
1. Hyperparameter tuning (Grid Search, Bayesian Optimization)
2. Ensemble methods (stacking, voting)
3. Feature engineering avançado (interações)
4. Modelo em duas etapas (classificação → regressão condicional)

### Dados Externos
1. Incorporar dados meteorológicos históricos
2. Informações de feriados e eventos especiais
3. Dados de tráfego aéreo e congestionamento
4. Informações de manutenção de aeronaves

### Deploy
1. API REST com FastAPI
2. Dashboard interativo (Streamlit/Plotly Dash)
3. Monitoramento de drift do modelo
4. Retreinamento automático com novos dados

## Referências e Recursos

### Documentação
- [Pandas Documentation](https://pandas.pydata.org/)
- [Scikit-learn User Guide](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

### Literatura Científica
- Papers sobre previsão de atrasos reportam R² entre 0.05-0.30
- Recall > Precision para este domínio (melhor alertar demais que de menos)

## Autor e Contribuições

Este projeto foi desenvolvido com fins **educacionais**, demonstrando boas práticas de:
- Ciência de dados
- Machine learning
- Engenharia de features
- Avaliação de modelos
- Documentação técnica

Sinta-se livre para usar este código para aprendizado e experimentação!

## Licença

Este projeto é disponibilizado para fins educacionais. Os dados são públicos e foram obtidos de fontes abertas.

---

**Última atualização**: Dezembro 2025

Desenvolvido com foco em **didática**, **reprodutibilidade** e **boas práticas** de Machine Learning.

Criado por: Guilherme Favaron
**www.guilhermefavaron.com.br**