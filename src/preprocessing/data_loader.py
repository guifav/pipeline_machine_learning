"""
Data Loader - Funções para Carregar Datasets
=============================================

Este módulo contém funções utilitárias para carregar os três datasets principais:
- flights.csv (voos)
- airports.csv (aeroportos)
- airlines.csv (companhias aéreas)

Autor: Tech Challenge Fase 3
Data: 2025-12-22
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import os


def load_flights(
    file_path: str,
    nrows: Optional[int] = None,
    sample_frac: Optional[float] = None,
    optimize_memory: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Carrega o dataset de voos (flights.csv).

    Este dataset contém aproximadamente 5.8 milhões de registros de voos.
    Por padrão, otimiza o uso de memória convertendo tipos de dados.

    Parameters
    ----------
    file_path : str
        Caminho para o arquivo flights.csv
    nrows : int, optional
        Número de linhas a carregar (útil para testes rápidos)
        Exemplo: nrows=100000 carrega apenas as primeiras 100k linhas
    sample_frac : float, optional
        Fração do dataset a amostrar aleatoriamente (entre 0 e 1)
        Exemplo: sample_frac=0.1 carrega 10% do dataset aleatoriamente
    optimize_memory : bool, default=True
        Se True, otimiza tipos de dados para reduzir uso de memória
    verbose : bool, default=True
        Se True, imprime informações sobre o carregamento

    Returns
    -------
    pd.DataFrame
        DataFrame com os dados de voos

    Examples
    --------
    >>> # Carregar amostra de 100k registros para teste rápido
    >>> df = load_flights('flights.csv', nrows=100000)

    >>> # Carregar 10% do dataset aleatoriamente
    >>> df = load_flights('flights.csv', sample_frac=0.1)

    >>> # Carregar dataset completo otimizado
    >>> df = load_flights('flights.csv')
    """

    if verbose:
        print("=" * 80)
        print("CARREGANDO DATASET DE VOOS")
        print("=" * 80)
        if nrows:
            print(f"Modo: Amostra de {nrows:,} primeiras linhas")
        elif sample_frac:
            print(f"Modo: Amostragem aleatória de {sample_frac*100:.1f}%")
        else:
            print("Modo: Dataset completo")
        print()

    # Definir tipos de dados para otimizar memória
    dtypes = {
        'YEAR': 'int16',
        'MONTH': 'int8',
        'DAY': 'int8',
        'DAY_OF_WEEK': 'int8',
        'AIRLINE': 'category',
        'FLIGHT_NUMBER': 'int16',
        'TAIL_NUMBER': 'category',
        'ORIGIN_AIRPORT': 'category',
        'DESTINATION_AIRPORT': 'category',
        'SCHEDULED_DEPARTURE': 'float32',
        'DEPARTURE_TIME': 'float32',
        'DEPARTURE_DELAY': 'float32',
        'TAXI_OUT': 'float32',
        'WHEELS_OFF': 'float32',
        'SCHEDULED_TIME': 'float32',
        'ELAPSED_TIME': 'float32',
        'AIR_TIME': 'float32',
        'DISTANCE': 'int16',
        'WHEELS_ON': 'float32',
        'TAXI_IN': 'float32',
        'SCHEDULED_ARRIVAL': 'float32',
        'ARRIVAL_TIME': 'float32',
        'ARRIVAL_DELAY': 'float32',
        'DIVERTED': 'int8',
        'CANCELLED': 'int8',
        'CANCELLATION_REASON': 'category',
        'AIR_SYSTEM_DELAY': 'float32',
        'SECURITY_DELAY': 'float32',
        'AIRLINE_DELAY': 'float32',
        'LATE_AIRCRAFT_DELAY': 'float32',
        'WEATHER_DELAY': 'float32'
    } if optimize_memory else None

    try:
        # Carregar dataset
        if nrows:
            df = pd.read_csv(file_path, nrows=nrows, dtype=dtypes)
        else:
            df = pd.read_csv(file_path, dtype=dtypes)

            # Se sample_frac especificado, amostrar aleatoriamente
            if sample_frac:
                df = df.sample(frac=sample_frac, random_state=42)
                df = df.reset_index(drop=True)

        if verbose:
            print(f"Dataset carregado com sucesso!")
            print(f"Dimensões: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
            print(f"Memória utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print()

        return df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Arquivo não encontrado: {file_path}\n"
            f"Verifique se o arquivo existe no caminho especificado."
        )
    except Exception as e:
        raise Exception(f"Erro ao carregar arquivo: {str(e)}")


def load_airports(file_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Carrega o dataset de aeroportos (airports.csv).

    Este dataset contém informações sobre 323 aeroportos, incluindo
    códigos IATA, nomes, localizações geográficas, etc.

    Parameters
    ----------
    file_path : str
        Caminho para o arquivo airports.csv
    verbose : bool, default=True
        Se True, imprime informações sobre o carregamento

    Returns
    -------
    pd.DataFrame
        DataFrame com os dados de aeroportos

    Examples
    --------
    >>> df_airports = load_airports('airports.csv')
    """

    if verbose:
        print("Carregando dataset de aeroportos...")

    try:
        df = pd.read_csv(file_path)

        # Converter código IATA para category para economizar memória
        if 'IATA_CODE' in df.columns:
            df['IATA_CODE'] = df['IATA_CODE'].astype('category')

        if verbose:
            print(f"{len(df)} aeroportos carregados!")
            print(f"Memória: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            print()

        return df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Arquivo não encontrado: {file_path}\n"
            f"Verifique se o arquivo existe no caminho especificado."
        )
    except Exception as e:
        raise Exception(f"Erro ao carregar arquivo: {str(e)}")


def load_airlines(file_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Carrega o dataset de companhias aéreas (airlines.csv).

    Este dataset contém informações sobre 14 companhias aéreas,
    incluindo códigos IATA e nomes completos.

    Parameters
    ----------
    file_path : str
        Caminho para o arquivo airlines.csv
    verbose : bool, default=True
        Se True, imprime informações sobre o carregamento

    Returns
    -------
    pd.DataFrame
        DataFrame com os dados de companhias aéreas

    Examples
    --------
    >>> df_airlines = load_airlines('airlines.csv')
    """

    if verbose:
        print("Carregando dataset de companhias aéreas...")

    try:
        df = pd.read_csv(file_path)

        # Converter código IATA para category
        if 'IATA_CODE' in df.columns:
            df['IATA_CODE'] = df['IATA_CODE'].astype('category')

        if verbose:
            print(f"{len(df)} companhias aéreas carregadas!")
            print(f"Memória: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            print()

        return df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Arquivo não encontrado: {file_path}\n"
            f"Verifique se o arquivo existe no caminho especificado."
        )
    except Exception as e:
        raise Exception(f"Erro ao carregar arquivo: {str(e)}")


def load_all_datasets(
    flights_path: str,
    airports_path: str,
    airlines_path: str,
    flights_nrows: Optional[int] = None,
    flights_sample_frac: Optional[float] = None,
    optimize_memory: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega todos os três datasets de uma vez.

    Função de conveniência que carrega flights, airports e airlines
    em uma única chamada.

    Parameters
    ----------
    flights_path : str
        Caminho para flights.csv
    airports_path : str
        Caminho para airports.csv
    airlines_path : str
        Caminho para airlines.csv
    flights_nrows : int, optional
        Número de linhas do dataset de voos a carregar
    flights_sample_frac : float, optional
        Fração do dataset de voos a amostrar
    optimize_memory : bool, default=True
        Se True, otimiza uso de memória
    verbose : bool, default=True
        Se True, imprime informações

    Returns
    -------
    tuple of pd.DataFrame
        (df_flights, df_airports, df_airlines)

    Examples
    --------
    >>> # Carregar todos com amostra de 100k voos
    >>> flights, airports, airlines = load_all_datasets(
    ...     'flights.csv',
    ...     'airports.csv',
    ...     'airlines.csv',
    ...     flights_nrows=100000
    ... )

    >>> # Carregar todos completos
    >>> flights, airports, airlines = load_all_datasets(
    ...     'flights.csv',
    ...     'airports.csv',
    ...     'airlines.csv'
    ... )
    """

    if verbose:
        print("\n" + "=" * 80)
        print("CARREGANDO TODOS OS DATASETS")
        print("=" * 80 + "\n")

    # Carregar voos
    df_flights = load_flights(
        flights_path,
        nrows=flights_nrows,
        sample_frac=flights_sample_frac,
        optimize_memory=optimize_memory,
        verbose=verbose
    )

    # Carregar aeroportos
    df_airports = load_airports(airports_path, verbose=verbose)

    # Carregar companhias
    df_airlines = load_airlines(airlines_path, verbose=verbose)

    if verbose:
        total_memory = (
            df_flights.memory_usage(deep=True).sum() +
            df_airports.memory_usage(deep=True).sum() +
            df_airlines.memory_usage(deep=True).sum()
        ) / 1024**2

        print("=" * 80)
        print("TODOS OS DATASETS CARREGADOS COM SUCESSO!")
        print("=" * 80)
        print(f"Memória total utilizada: {total_memory:.2f} MB")
        print()

    return df_flights, df_airports, df_airlines


def get_dataset_info(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Imprime informações resumidas sobre um dataset.

    Mostra dimensões, tipos de dados, valores ausentes e uso de memória.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a analisar
    name : str, default="Dataset"
        Nome do dataset para exibição

    Examples
    --------
    >>> get_dataset_info(df_flights, "Voos")
    """

    print("=" * 80)
    print(f"INFORMAÇÕES DO DATASET: {name}")
    print("=" * 80)
    print(f"\nDimensões: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    print(f"Memória: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nTipos de dados:")
    print(df.dtypes.value_counts())
    print(f"\nValores ausentes:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        for col, count in missing.items():
            pct = count / len(df) * 100
            print(f"  - {col}: {count:,} ({pct:.2f}%)")
    else:
        print("  Nenhum valor ausente!")
    print()


# Exemplo de uso
if __name__ == "__main__":
    """
    Exemplo de como usar as funções deste módulo.
    Execute este arquivo diretamente para testar:

    python data_loader.py
    """

    # Definir caminhos (ajuste conforme sua estrutura)
    FLIGHTS_FILE = '../../data/raw/flights.csv'
    AIRPORTS_FILE = '../../data/raw/airports.csv'
    AIRLINES_FILE = '../../data/raw/airlines.csv'

    print("\nTESTE DO MÓDULO DATA_LOADER\n")

    # Teste 1: Carregar amostra pequena para teste
    print("TESTE 1: Carregando amostra de 10.000 registros\n")
    df_flights_sample = load_flights(FLIGHTS_FILE, nrows=10000)
    get_dataset_info(df_flights_sample, "Voos (Amostra)")

    # Teste 2: Carregar todos os datasets auxiliares
    print("\nTESTE 2: Carregando datasets auxiliares\n")
    df_airports = load_airports(AIRPORTS_FILE)
    df_airlines = load_airlines(AIRLINES_FILE)

    # Teste 3: Carregar todos de uma vez
    print("\nTESTE 3: Carregando todos os datasets\n")
    flights, airports, airlines = load_all_datasets(
        FLIGHTS_FILE,
        AIRPORTS_FILE,
        AIRLINES_FILE,
        flights_nrows=10000
    )

    print("Todos os testes concluídos com sucesso!")
