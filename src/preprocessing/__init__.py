"""
Modulo de Pre-processamento de Dados
=====================================

Este modulo contem funcoes para carregar, limpar e preparar os dados
para analise e modelagem.

Modulos disponiveis:
- data_loader: Funcoes para carregar os datasets
- feature_engineering: Criacao de novas features
"""

__version__ = "1.0.0"
__author__ = "Tech Challenge Fase 3"

# Importar funcoes principais
from .data_loader import (
    load_flights,
    load_airports,
    load_airlines,
    load_all_datasets,
    get_dataset_info
)

from .feature_engineering import (
    create_target_variables,
    create_temporal_features,
    create_geographic_features,
    create_aggregated_features,
    create_all_features,
    get_feature_names
)

from .missing_values import (
    analyze_missing_values,
    get_missing_patterns,
    handle_cancellation_reason,
    handle_categorical_delays,
    handle_flight_times,
    handle_arrival_times,
    handle_departure_times,
    handle_tail_number,
    handle_scheduled_time,
    handle_all_missing_values,
    calcular_tempo_decorrido,
    calcular_delay
)

__all__ = [
    # Data loader
    'load_flights',
    'load_airports',
    'load_airlines',
    'load_all_datasets',
    'get_dataset_info',
    # Feature engineering
    'create_target_variables',
    'create_temporal_features',
    'create_geographic_features',
    'create_aggregated_features',
    'create_all_features',
    'get_feature_names',
    # Missing values
    'analyze_missing_values',
    'get_missing_patterns',
    'handle_cancellation_reason',
    'handle_categorical_delays',
    'handle_flight_times',
    'handle_arrival_times',
    'handle_departure_times',
    'handle_tail_number',
    'handle_scheduled_time',
    'handle_all_missing_values',
    'calcular_tempo_decorrido',
    'calcular_delay'
]
