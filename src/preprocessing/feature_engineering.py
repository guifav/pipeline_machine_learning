"""
Feature Engineering - Engenharia de Features para Previsao de Atrasos
======================================================================

Este modulo contem funcoes para criar features derivadas a partir dos dados
brutos de voos, aeroportos e companhias aereas.

Autor: Tech Challenge Fase 3
Data: 2025-12-22
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime


def create_target_variables(
    df: pd.DataFrame,
    delay_threshold: int = 15,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Cria as variaveis target para classificacao e regressao.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com os dados de voos
    delay_threshold : int, default=15
        Threshold em minutos para considerar um voo atrasado
    verbose : bool, default=True
        Se True, imprime informacoes sobre as targets criadas

    Returns
    -------
    pd.DataFrame
        DataFrame com as novas colunas target adicionadas

    Examples
    --------
    >>> df = create_target_variables(df_flights, delay_threshold=15)
    """

    df = df.copy()

    # Target binario para classificacao
    df['IS_DELAYED'] = (df['DEPARTURE_DELAY'] > delay_threshold).astype(int)

    # Target continuo para regressao
    df['DELAY_MINUTES'] = df['DEPARTURE_DELAY']

    if verbose:
        print("=" * 80)
        print("VARIAVEIS TARGET CRIADAS")
        print("=" * 80)
        print(f"\n1. IS_DELAYED (Classificacao Binaria)")
        print(f"   - Threshold: {delay_threshold} minutos")
        print(f"   - Voos atrasados: {df['IS_DELAYED'].sum():,} ({df['IS_DELAYED'].mean()*100:.2f}%)")
        print(f"   - Voos no horario: {(df['IS_DELAYED']==0).sum():,} ({(df['IS_DELAYED']==0).mean()*100:.2f}%)")
        print(f"\n2. DELAY_MINUTES (Regressao)")
        print(f"   - Media: {df['DELAY_MINUTES'].mean():.2f} minutos")
        print(f"   - Mediana: {df['DELAY_MINUTES'].median():.2f} minutos\n")

    return df


def create_temporal_features(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Cria features temporais derivadas das colunas de data e horario.

    Features criadas:
    - PERIOD_OF_DAY: Periodo do dia (Madrugada, Manha, Tarde, Noite)
    - IS_WEEKEND: Se o voo e em fim de semana
    - SEASON: Estacao do ano
    - HOUR: Hora da partida programada
    - IS_MORNING_RUSH: Se e horario de pico da manha (6-9h)
    - IS_EVENING_RUSH: Se e horario de pico da noite (17-20h)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com os dados de voos
    verbose : bool, default=True
        Se True, imprime informacoes sobre as features criadas

    Returns
    -------
    pd.DataFrame
        DataFrame com as novas features temporais

    Examples
    --------
    >>> df = create_temporal_features(df_flights)
    """

    df = df.copy()

    # Extrair hora da partida programada (formato HHMM -> hora)
    df['HOUR'] = (df['SCHEDULED_DEPARTURE'] // 100).astype('Int64')

    # Periodo do dia
    def get_period_of_day(hour):
        if pd.isna(hour):
            return 'Unknown'
        elif 0 <= hour < 6:
            return 'Madrugada'
        elif 6 <= hour < 12:
            return 'Manha'
        elif 12 <= hour < 18:
            return 'Tarde'
        else:
            return 'Noite'

    df['PERIOD_OF_DAY'] = df['HOUR'].apply(get_period_of_day)
    df['PERIOD_OF_DAY'] = df['PERIOD_OF_DAY'].astype('category')

    # Fim de semana (Sabado=6, Domingo=7)
    df['IS_WEEKEND'] = (df['DAY_OF_WEEK'].isin([6, 7])).astype(int)

    # Estacao do ano
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Inverno'
        elif month in [3, 4, 5]:
            return 'Primavera'
        elif month in [6, 7, 8]:
            return 'Verao'
        else:
            return 'Outono'

    df['SEASON'] = df['MONTH'].apply(get_season)
    df['SEASON'] = df['SEASON'].astype('category')

    # Horarios de pico
    df['IS_MORNING_RUSH'] = ((df['HOUR'] >= 6) & (df['HOUR'] <= 9)).astype(int)
    df['IS_EVENING_RUSH'] = ((df['HOUR'] >= 17) & (df['HOUR'] <= 20)).astype(int)

    if verbose:
        print("=" * 80)
        print("FEATURES TEMPORAIS CRIADAS")
        print("=" * 80)
        print(f"\n1. PERIOD_OF_DAY:")
        print(df['PERIOD_OF_DAY'].value_counts().sort_index())
        print(f"\n2. IS_WEEKEND:")
        print(f"   - Fim de semana: {df['IS_WEEKEND'].sum():,} ({df['IS_WEEKEND'].mean()*100:.2f}%)")
        print(f"\n3. SEASON:")
        print(df['SEASON'].value_counts())
        print(f"\n4. Horarios de Pico:")
        print(f"   - Manha (6-9h): {df['IS_MORNING_RUSH'].sum():,} ({df['IS_MORNING_RUSH'].mean()*100:.2f}%)")
        print(f"   - Noite (17-20h): {df['IS_EVENING_RUSH'].sum():,} ({df['IS_EVENING_RUSH'].mean()*100:.2f}%)\n")

    return df


def create_geographic_features(
    df_flights: pd.DataFrame,
    df_airports: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Cria features geograficas a partir dos dados de aeroportos.

    Features criadas:
    - ORIGIN_STATE: Estado do aeroporto de origem
    - DEST_STATE: Estado do aeroporto de destino
    - IS_DOMESTIC: Se o voo e domestico (mesmo estado)
    - ORIGIN_LATITUDE: Latitude do aeroporto de origem
    - ORIGIN_LONGITUDE: Longitude do aeroporto de origem
    - DEST_LATITUDE: Latitude do aeroporto de destino
    - DEST_LONGITUDE: Longitude do aeroporto de destino

    Parameters
    ----------
    df_flights : pd.DataFrame
        DataFrame com os dados de voos
    df_airports : pd.DataFrame
        DataFrame com os dados de aeroportos
    verbose : bool, default=True
        Se True, imprime informacoes sobre as features criadas

    Returns
    -------
    pd.DataFrame
        DataFrame com as novas features geograficas

    Examples
    --------
    >>> df = create_geographic_features(df_flights, df_airports)
    """

    df = df_flights.copy()

    # Merge com dados de aeroportos de origem
    df = df.merge(
        df_airports[['IATA_CODE', 'STATE', 'LATITUDE', 'LONGITUDE']],
        left_on='ORIGIN_AIRPORT',
        right_on='IATA_CODE',
        how='left',
        suffixes=('', '_ORIGIN')
    )
    df = df.rename(columns={
        'STATE': 'ORIGIN_STATE',
        'LATITUDE': 'ORIGIN_LATITUDE',
        'LONGITUDE': 'ORIGIN_LONGITUDE'
    })
    df = df.drop(columns=['IATA_CODE'], errors='ignore')

    # Merge com dados de aeroportos de destino
    df = df.merge(
        df_airports[['IATA_CODE', 'STATE', 'LATITUDE', 'LONGITUDE']],
        left_on='DESTINATION_AIRPORT',
        right_on='IATA_CODE',
        how='left',
        suffixes=('', '_DEST')
    )
    df = df.rename(columns={
        'STATE': 'DEST_STATE',
        'LATITUDE': 'DEST_LATITUDE',
        'LONGITUDE': 'DEST_LONGITUDE'
    })
    df = df.drop(columns=['IATA_CODE'], errors='ignore')

    # Voo domestico (mesmo estado)
    df['IS_DOMESTIC'] = (df['ORIGIN_STATE'] == df['DEST_STATE']).astype(int)

    # Converter estados para category
    df['ORIGIN_STATE'] = df['ORIGIN_STATE'].astype('category')
    df['DEST_STATE'] = df['DEST_STATE'].astype('category')

    if verbose:
        print("=" * 80)
        print("FEATURES GEOGRAFICAS CRIADAS")
        print("=" * 80)
        print(f"\n1. Estados de Origem:")
        print(f"   - Total de estados: {df['ORIGIN_STATE'].nunique()}")
        print(f"\n2. IS_DOMESTIC:")
        print(f"   - Voos domesticos (mesmo estado): {df['IS_DOMESTIC'].sum():,} ({df['IS_DOMESTIC'].mean()*100:.2f}%)")
        print(f"\n3. Coordenadas geograficas adicionadas para origem e destino\n")

    return df


def create_aggregated_features(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Cria features agregadas baseadas em historico de atrasos.

    Features criadas:
    - AIRLINE_DELAY_RATE: Taxa historica de atraso por companhia
    - ORIGIN_DELAY_RATE: Taxa historica de atraso por aeroporto de origem
    - DEST_DELAY_RATE: Taxa historica de atraso por aeroporto de destino
    - ROUTE_DELAY_RATE: Taxa historica de atraso por rota

    IMPORTANTE: Estas features devem ser calculadas apenas no conjunto de treino
    para evitar data leakage!

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com os dados de voos (deve conter IS_DELAYED)
    verbose : bool, default=True
        Se True, imprime informacoes sobre as features criadas

    Returns
    -------
    pd.DataFrame
        DataFrame com as novas features agregadas

    Examples
    --------
    >>> df = create_aggregated_features(df_flights)
    """

    df = df.copy()

    # Verificar se IS_DELAYED existe
    if 'IS_DELAYED' not in df.columns:
        raise ValueError("Coluna 'IS_DELAYED' nao encontrada. Execute create_target_variables() primeiro.")

    # Taxa de atraso por companhia
    airline_delay_rate = df.groupby('AIRLINE', observed=True)['IS_DELAYED'].mean()
    df['AIRLINE_DELAY_RATE'] = df['AIRLINE'].astype(str).map(airline_delay_rate).astype(float)

    # Taxa de atraso por aeroporto de origem
    origin_delay_rate = df.groupby('ORIGIN_AIRPORT', observed=True)['IS_DELAYED'].mean()
    df['ORIGIN_DELAY_RATE'] = df['ORIGIN_AIRPORT'].astype(str).map(origin_delay_rate).astype(float)

    # Taxa de atraso por aeroporto de destino
    dest_delay_rate = df.groupby('DESTINATION_AIRPORT', observed=True)['IS_DELAYED'].mean()
    df['DEST_DELAY_RATE'] = df['DESTINATION_AIRPORT'].astype(str).map(dest_delay_rate).astype(float)

    # Criar coluna de rota
    df['ROUTE'] = df['ORIGIN_AIRPORT'].astype(str) + '-' + df['DESTINATION_AIRPORT'].astype(str)

    # Taxa de atraso por rota
    route_delay_rate = df.groupby('ROUTE', observed=True)['IS_DELAYED'].mean()
    df['ROUTE_DELAY_RATE'] = df['ROUTE'].map(route_delay_rate).astype(float)

    if verbose:
        print("=" * 80)
        print("FEATURES AGREGADAS CRIADAS")
        print("=" * 80)
        print(f"\n1. AIRLINE_DELAY_RATE:")
        print(f"   - Media: {df['AIRLINE_DELAY_RATE'].mean():.4f}")
        print(f"   - Min: {df['AIRLINE_DELAY_RATE'].min():.4f}")
        print(f"   - Max: {df['AIRLINE_DELAY_RATE'].max():.4f}")
        print(f"\n2. ORIGIN_DELAY_RATE:")
        print(f"   - Media: {df['ORIGIN_DELAY_RATE'].mean():.4f}")
        print(f"   - Min: {df['ORIGIN_DELAY_RATE'].min():.4f}")
        print(f"   - Max: {df['ORIGIN_DELAY_RATE'].max():.4f}")
        print(f"\n3. ROUTE_DELAY_RATE:")
        print(f"   - Media: {df['ROUTE_DELAY_RATE'].mean():.4f}")
        print(f"   - Rotas unicas: {df['ROUTE'].nunique():,}\n")

    return df


def create_all_features(
    df_flights: pd.DataFrame,
    df_airports: Optional[pd.DataFrame] = None,
    delay_threshold: int = 15,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Cria todas as features de uma vez.

    Esta funcao e um wrapper que executa todas as funcoes de feature engineering
    em sequencia.

    Parameters
    ----------
    df_flights : pd.DataFrame
        DataFrame com os dados de voos
    df_airports : pd.DataFrame, optional
        DataFrame com os dados de aeroportos (necessario para features geograficas)
    delay_threshold : int, default=15
        Threshold para considerar voo atrasado
    verbose : bool, default=True
        Se True, imprime informacoes sobre cada etapa

    Returns
    -------
    pd.DataFrame
        DataFrame com todas as features criadas

    Examples
    --------
    >>> df = create_all_features(df_flights, df_airports, delay_threshold=15)
    """

    if verbose:
        print("\n" + "=" * 80)
        print("INICIANDO ENGENHARIA DE FEATURES COMPLETA")
        print("=" * 80 + "\n")

    # 1. Criar targets
    df = create_target_variables(df_flights, delay_threshold=delay_threshold, verbose=verbose)

    # 2. Criar features temporais
    df = create_temporal_features(df, verbose=verbose)

    # 3. Criar features geograficas (se df_airports fornecido)
    if df_airports is not None:
        df = create_geographic_features(df, df_airports, verbose=verbose)

    # 4. Criar features agregadas
    df = create_aggregated_features(df, verbose=verbose)

    if verbose:
        print("=" * 80)
        print("ENGENHARIA DE FEATURES CONCLUIDA")
        print("=" * 80)
        print(f"\nDataset final:")
        print(f"  - Linhas: {len(df):,}")
        print(f"  - Colunas: {len(df.columns)}")
        print(f"  - Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

    return df


def get_feature_names(
    include_targets: bool = False,
    include_original: bool = False
) -> dict:
    """
    Retorna dicionario com nomes das features criadas por categoria.

    Parameters
    ----------
    include_targets : bool, default=False
        Se True, inclui variaveis target
    include_original : bool, default=False
        Se True, inclui variaveis originais relevantes

    Returns
    -------
    dict
        Dicionario com categorias e listas de features

    Examples
    --------
    >>> features = get_feature_names(include_targets=True)
    >>> print(features['temporal'])
    """

    features = {
        'target': ['IS_DELAYED', 'DELAY_MINUTES'] if include_targets else [],
        'temporal': [
            'HOUR', 'PERIOD_OF_DAY', 'IS_WEEKEND', 'SEASON',
            'IS_MORNING_RUSH', 'IS_EVENING_RUSH'
        ],
        'geographic': [
            'ORIGIN_STATE', 'DEST_STATE', 'IS_DOMESTIC',
            'ORIGIN_LATITUDE', 'ORIGIN_LONGITUDE',
            'DEST_LATITUDE', 'DEST_LONGITUDE'
        ],
        'aggregated': [
            'AIRLINE_DELAY_RATE', 'ORIGIN_DELAY_RATE',
            'DEST_DELAY_RATE', 'ROUTE_DELAY_RATE'
        ]
    }

    if include_original:
        features['original'] = [
            'YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK',
            'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
            'SCHEDULED_DEPARTURE', 'SCHEDULED_TIME', 'DISTANCE'
        ]

    return features


# Exemplo de uso
if __name__ == "__main__":
    """
    Exemplo de como usar as funcoes deste modulo.
    Execute: python feature_engineering.py
    """

    print("\nTESTE DO MODULO FEATURE_ENGINEERING\n")
    print("=" * 80)

    # Importar data_loader
    import sys
    sys.path.append('..')
    from preprocessing.data_loader import load_all_datasets

    # Carregar dados (amostra pequena para teste)
    FLIGHTS_FILE = '../../data/raw/flights.csv'
    AIRPORTS_FILE = '../../data/raw/airports.csv'
    AIRLINES_FILE = '../../data/raw/airlines.csv'

    print("\n1. Carregando dados de teste (10.000 registros)...\n")
    df_flights, df_airports, df_airlines = load_all_datasets(
        FLIGHTS_FILE,
        AIRPORTS_FILE,
        AIRLINES_FILE,
        flights_nrows=10000,
        verbose=False
    )

    print(f"Dados carregados: {len(df_flights):,} voos\n")

    # Criar todas as features
    print("2. Criando todas as features...\n")
    df_enriched = create_all_features(
        df_flights,
        df_airports,
        delay_threshold=15,
        verbose=True
    )

    # Mostrar features criadas
    print("\n3. Features criadas por categoria:\n")
    features = get_feature_names(include_targets=True, include_original=False)
    for category, feature_list in features.items():
        print(f"{category.upper()}:")
        for feat in feature_list:
            print(f"  - {feat}")
        print()

    print("=" * 80)
    print("Teste concluido com sucesso!")
    print("=" * 80)
