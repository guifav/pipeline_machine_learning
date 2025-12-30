"""
Missing Values Treatment - Estrategias para Tratamento de Valores Ausentes
============================================================================

Este modulo contem funcoes para analisar e tratar valores ausentes no dataset
de voos, aplicando diferentes estrategias conforme o tipo e significado de
cada variavel.

IMPORTANTE: Valores ausentes neste dataset NAO sao aleatorios (MNAR).
Eles possuem significado semantico relacionado ao status e contexto do voo.

Autor: Tech Challenge Fase 3
Data: 2025-12-22
Versao: 2.0 - Estrategias corrigidas
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================================================
# FUNCOES AUXILIARES
# ============================================================================

def calcular_tempo_decorrido(departure_time: pd.Series, arrival_time: pd.Series) -> pd.Series:
    """
    Calcula tempo decorrido entre partida e chegada em minutos.

    Considera mudanca de dia (ex: partida 2350, chegada 0010 = 20 minutos).

    Parameters
    ----------
    departure_time : pd.Series
        Horario de partida no formato HHMM (ex: 1430 = 14:30)
    arrival_time : pd.Series
        Horario de chegada no formato HHMM

    Returns
    -------
    pd.Series
        Tempo decorrido em minutos
    """
    # Converter HHMM para minutos desde meia-noite
    dep_minutes = (departure_time // 100) * 60 + (departure_time % 100)
    arr_minutes = (arrival_time // 100) * 60 + (arrival_time % 100)

    # Calcular diferenca
    elapsed = arr_minutes - dep_minutes

    # Se negativo, houve mudanca de dia (adicionar 24h = 1440 minutos)
    elapsed = elapsed.where(elapsed >= 0, elapsed + 1440)

    return elapsed


def calcular_delay(actual_time: pd.Series, scheduled_time: pd.Series) -> pd.Series:
    """
    Calcula atraso em minutos entre horario programado e real.

    Considera mudanca de dia.

    Parameters
    ----------
    actual_time : pd.Series
        Horario real no formato HHMM
    scheduled_time : pd.Series
        Horario programado no formato HHMM

    Returns
    -------
    pd.Series
        Atraso em minutos (negativo = adiantado, positivo = atrasado)
    """
    # Converter para minutos
    actual_min = (actual_time // 100) * 60 + (actual_time % 100)
    scheduled_min = (scheduled_time // 100) * 60 + (scheduled_time % 100)

    # Calcular diferenca
    delay = actual_min - scheduled_min

    # Ajustar mudanca de dia (se diferenca > 12h, provavelmente houve mudanca)
    delay = delay.where(delay.abs() < 720, delay - np.sign(delay) * 1440)

    return delay


# ============================================================================
# ANALISE DE MISSING VALUES
# ============================================================================

def analyze_missing_values(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Analisa valores ausentes no dataset.

    Retorna um DataFrame com informacoes detalhadas sobre missing values
    em cada coluna.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a analisar
    verbose : bool, default=True
        Se True, imprime relatorio detalhado

    Returns
    -------
    pd.DataFrame
        DataFrame com analise de missing values por coluna
    """

    if verbose:
        print("=" * 80)
        print("ANALISE DE VALORES AUSENTES")
        print("=" * 80)

    # Calcular estatisticas de missing values
    missing_stats = pd.DataFrame({
        'column': df.columns,
        'total_missing': df.isnull().sum().values,
        'percent_missing': (df.isnull().sum() / len(df) * 100).values,
        'dtype': df.dtypes.values,
        'non_null_count': df.count().values,
        'total_count': len(df)
    })

    # Filtrar apenas colunas com missing values
    missing_stats = missing_stats[missing_stats['total_missing'] > 0]
    missing_stats = missing_stats.sort_values('total_missing', ascending=False)
    missing_stats = missing_stats.reset_index(drop=True)

    if verbose:
        if len(missing_stats) > 0:
            print(f"\nTotal de colunas com valores ausentes: {len(missing_stats)}\n")
            print(missing_stats.to_string())
            print()
        else:
            print("\nNenhum valor ausente encontrado!")
            print()

    return missing_stats


def get_missing_patterns(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Identifica padroes de valores ausentes entre colunas.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a analisar
    columns : list of str, optional
        Lista de colunas a analisar

    Returns
    -------
    pd.DataFrame
        Matriz de correlacao de missing values entre colunas
    """

    if columns is None:
        columns = df.columns[df.isnull().any()].tolist()

    if len(columns) == 0:
        return pd.DataFrame()

    missing_matrix = df[columns].isnull()
    missing_corr = missing_matrix.corr()

    return missing_corr


# ============================================================================
# GRUPO 1: CANCELLATION_REASON
# ============================================================================

def handle_cancellation_reason(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Trata valores ausentes em CANCELLATION_REASON.

    Estrategia: Preencher com "NO_CANCELLATION" quando CANCELLED = 0

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de voos
    verbose : bool
        Se True, imprime informacoes

    Returns
    -------
    pd.DataFrame
        Dataset com CANCELLATION_REASON tratado
    """
    df = df.copy()

    if 'CANCELLATION_REASON' not in df.columns:
        return df

    if verbose:
        print("GRUPO 1: Tratando CANCELLATION_REASON...")
        missing_before = df['CANCELLATION_REASON'].isnull().sum()

    # Adicionar nova categoria se coluna for categorical
    if df['CANCELLATION_REASON'].dtype.name == 'category':
        if 'NO_CANCELLATION' not in df['CANCELLATION_REASON'].cat.categories:
            df['CANCELLATION_REASON'] = df['CANCELLATION_REASON'].cat.add_categories(['NO_CANCELLATION'])

    # Preencher com "NO_CANCELLATION" quando nao houve cancelamento
    mask = (df['CANCELLED'] == 0) & df['CANCELLATION_REASON'].isnull()
    df.loc[mask, 'CANCELLATION_REASON'] = 'NO_CANCELLATION'

    if verbose:
        filled = mask.sum()
        print(f"  - {filled:,} valores preenchidos com 'NO_CANCELLATION'")
        print(f"  - Missing antes: {missing_before:,}, depois: {df['CANCELLATION_REASON'].isnull().sum():,}\n")

    return df


# ============================================================================
# GRUPO 2: DELAYS CATEGORICOS
# ============================================================================

def handle_categorical_delays(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Trata valores ausentes em delays categoricos.

    Estrategia: Preencher com 0 (ausente = nao houve atraso por essa causa)

    Colunas tratadas:
    - WEATHER_DELAY
    - LATE_AIRCRAFT_DELAY
    - AIRLINE_DELAY
    - SECURITY_DELAY
    - AIR_SYSTEM_DELAY

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de voos
    verbose : bool
        Se True, imprime informacoes

    Returns
    -------
    pd.DataFrame
        Dataset com delays categoricos tratados
    """
    df = df.copy()

    delay_cols = ['WEATHER_DELAY', 'LATE_AIRCRAFT_DELAY', 'AIRLINE_DELAY',
                  'SECURITY_DELAY', 'AIR_SYSTEM_DELAY']

    if verbose:
        print("GRUPO 2: Tratando delays categoricos...")

    for col in delay_cols:
        if col in df.columns:
            missing_before = df[col].isnull().sum()
            df[col] = df[col].fillna(0)

            if verbose:
                print(f"  - {col}: {missing_before:,} valores preenchidos com 0")

    if verbose:
        print()

    return df


# ============================================================================
# GRUPO 3: TEMPOS DE VOO
# ============================================================================

def handle_flight_times(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Trata valores ausentes em tempos de voo.

    Estrategia: Calcular a partir de outras colunas

    Colunas tratadas:
    - ELAPSED_TIME: calcular de DEPARTURE_TIME e ARRIVAL_TIME
    - AIR_TIME: calcular de ELAPSED_TIME - TAXI_OUT - TAXI_IN
    - ARRIVAL_DELAY: calcular de ARRIVAL_TIME - SCHEDULED_ARRIVAL

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de voos
    verbose : bool
        Se True, imprime informacoes

    Returns
    -------
    pd.DataFrame
        Dataset com tempos de voo tratados
    """
    df = df.copy()

    if verbose:
        print("GRUPO 3: Tratando tempos de voo...")

    # ELAPSED_TIME
    if 'ELAPSED_TIME' in df.columns:
        missing_before = df['ELAPSED_TIME'].isnull().sum()

        if all(col in df.columns for col in ['DEPARTURE_TIME', 'ARRIVAL_TIME']):
            mask = (df['ELAPSED_TIME'].isnull() &
                   df['DEPARTURE_TIME'].notna() &
                   df['ARRIVAL_TIME'].notna())

            df.loc[mask, 'ELAPSED_TIME'] = calcular_tempo_decorrido(
                df.loc[mask, 'DEPARTURE_TIME'],
                df.loc[mask, 'ARRIVAL_TIME']
            )

            if verbose:
                calculated = mask.sum()
                print(f"  - ELAPSED_TIME: {calculated:,} valores calculados")

    # AIR_TIME
    if 'AIR_TIME' in df.columns:
        missing_before = df['AIR_TIME'].isnull().sum()

        if all(col in df.columns for col in ['ELAPSED_TIME', 'TAXI_OUT', 'TAXI_IN']):
            mask = (df['AIR_TIME'].isnull() &
                   df['ELAPSED_TIME'].notna() &
                   df['TAXI_OUT'].notna() &
                   df['TAXI_IN'].notna())

            df.loc[mask, 'AIR_TIME'] = (df.loc[mask, 'ELAPSED_TIME'] -
                                        df.loc[mask, 'TAXI_OUT'] -
                                        df.loc[mask, 'TAXI_IN'])

            if verbose:
                calculated = mask.sum()
                print(f"  - AIR_TIME: {calculated:,} valores calculados")

    # ARRIVAL_DELAY
    if 'ARRIVAL_DELAY' in df.columns:
        missing_before = df['ARRIVAL_DELAY'].isnull().sum()

        if all(col in df.columns for col in ['ARRIVAL_TIME', 'SCHEDULED_ARRIVAL']):
            mask = (df['ARRIVAL_DELAY'].isnull() &
                   df['ARRIVAL_TIME'].notna() &
                   df['SCHEDULED_ARRIVAL'].notna())

            df.loc[mask, 'ARRIVAL_DELAY'] = calcular_delay(
                df.loc[mask, 'ARRIVAL_TIME'],
                df.loc[mask, 'SCHEDULED_ARRIVAL']
            )

            if verbose:
                calculated = mask.sum()
                print(f"  - ARRIVAL_DELAY: {calculated:,} valores calculados\n")

    return df


# ============================================================================
# GRUPO 4: HORARIOS DE CHEGADA
# ============================================================================

def handle_arrival_times(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Trata valores ausentes em horarios de chegada.

    Estrategia:
    - Voos cancelados: manter NaN
    - Voos que partiram mas sem chegada: imputar com SCHEDULED_ARRIVAL + DEPARTURE_DELAY

    Colunas tratadas:
    - ARRIVAL_TIME
    - TAXI_IN
    - WHEELS_ON

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de voos
    verbose : bool
        Se True, imprime informacoes

    Returns
    -------
    pd.DataFrame
        Dataset com horarios de chegada tratados
    """
    df = df.copy()

    if verbose:
        print("GRUPO 4: Tratando horarios de chegada...")

    # ARRIVAL_TIME: imputar para voos que partiram mas nao tem chegada
    if 'ARRIVAL_TIME' in df.columns:
        missing_before = df['ARRIVAL_TIME'].isnull().sum()

        # Voos que partiram mas sem chegada registrada
        mask = ((df['CANCELLED'] == 0) &
               (df['DEPARTURE_TIME'].notna()) &
               (df['ARRIVAL_TIME'].isnull()))

        if 'SCHEDULED_ARRIVAL' in df.columns and 'DEPARTURE_DELAY' in df.columns:
            # Estimar: horario programado + atraso na partida
            df.loc[mask, 'ARRIVAL_TIME'] = (df.loc[mask, 'SCHEDULED_ARRIVAL'] +
                                            df.loc[mask, 'DEPARTURE_DELAY'].fillna(0))

            if verbose:
                imputed = mask.sum()
                print(f"  - ARRIVAL_TIME: {imputed:,} valores imputados")

    # TAXI_IN e WHEELS_ON: manter NaN se ARRIVAL_TIME ausente
    if verbose:
        for col in ['TAXI_IN', 'WHEELS_ON']:
            if col in df.columns:
                missing = df[col].isnull().sum()
                print(f"  - {col}: {missing:,} valores ausentes (mantidos)")
        print()

    return df


# ============================================================================
# GRUPO 5: HORARIOS DE PARTIDA
# ============================================================================

def handle_departure_times(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Trata valores ausentes em horarios de partida.

    Estrategia:
    - Voos cancelados: manter NaN
    - Voos que decolaram: imputar com mediana por rota + dia da semana

    Colunas tratadas:
    - DEPARTURE_TIME
    - DEPARTURE_DELAY (recalcular apos imputacao)
    - WHEELS_OFF
    - TAXI_OUT

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de voos
    verbose : bool
        Se True, imprime informacoes

    Returns
    -------
    pd.DataFrame
        Dataset com horarios de partida tratados
    """
    df = df.copy()

    if verbose:
        print("GRUPO 5: Tratando horarios de partida...")

    # DEPARTURE_TIME: imputar com mediana por rota/dia
    if 'DEPARTURE_TIME' in df.columns:
        missing_before = df['DEPARTURE_TIME'].isnull().sum()

        # Voos NAO cancelados com DEPARTURE_TIME ausente
        mask_impute = (df['CANCELLED'] == 0) & df['DEPARTURE_TIME'].isnull()

        if mask_impute.sum() > 0:
            # Calcular mediana por grupo
            group_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DAY_OF_WEEK']

            if all(col in df.columns for col in group_cols):
                medians = df[df['DEPARTURE_TIME'].notna()].groupby(group_cols, observed=True)['DEPARTURE_TIME'].median()

                # Aplicar imputacao
                for idx in df[mask_impute].index:
                    row = df.loc[idx]
                    key = (row['AIRLINE'], row['ORIGIN_AIRPORT'],
                          row['DESTINATION_AIRPORT'], row['DAY_OF_WEEK'])

                    if key in medians.index:
                        df.loc[idx, 'DEPARTURE_TIME'] = medians[key]
                    elif 'SCHEDULED_DEPARTURE' in df.columns:
                        # Fallback: usar horario programado
                        df.loc[idx, 'DEPARTURE_TIME'] = row['SCHEDULED_DEPARTURE']

                if verbose:
                    imputed = missing_before - df['DEPARTURE_TIME'].isnull().sum()
                    print(f"  - DEPARTURE_TIME: {imputed:,} valores imputados (mediana por rota/dia)")

    # DEPARTURE_DELAY: recalcular apos imputacao de DEPARTURE_TIME
    if 'DEPARTURE_DELAY' in df.columns:
        if all(col in df.columns for col in ['DEPARTURE_TIME', 'SCHEDULED_DEPARTURE']):
            mask = df['DEPARTURE_DELAY'].isnull() & df['DEPARTURE_TIME'].notna()

            df.loc[mask, 'DEPARTURE_DELAY'] = calcular_delay(
                df.loc[mask, 'DEPARTURE_TIME'],
                df.loc[mask, 'SCHEDULED_DEPARTURE']
            )

            if verbose:
                calculated = mask.sum()
                print(f"  - DEPARTURE_DELAY: {calculated:,} valores recalculados")

    # WHEELS_OFF e TAXI_OUT: manter NaN se DEPARTURE_TIME ausente
    if verbose:
        for col in ['WHEELS_OFF', 'TAXI_OUT']:
            if col in df.columns:
                missing = df[col].isnull().sum()
                print(f"  - {col}: {missing:,} valores ausentes (mantidos)")
        print()

    return df


# ============================================================================
# GRUPO 6: TAIL_NUMBER
# ============================================================================

def handle_tail_number(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Trata valores ausentes em TAIL_NUMBER.

    Estrategia: Preencher com "UNKNOWN"

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de voos
    verbose : bool
        Se True, imprime informacoes

    Returns
    -------
    pd.DataFrame
        Dataset com TAIL_NUMBER tratado
    """
    df = df.copy()

    if 'TAIL_NUMBER' not in df.columns:
        return df

    if verbose:
        print("GRUPO 6: Tratando TAIL_NUMBER...")
        missing_before = df['TAIL_NUMBER'].isnull().sum()

    # Adicionar nova categoria se coluna for categorical
    if df['TAIL_NUMBER'].dtype.name == 'category':
        if 'UNKNOWN' not in df['TAIL_NUMBER'].cat.categories:
            df['TAIL_NUMBER'] = df['TAIL_NUMBER'].cat.add_categories(['UNKNOWN'])

    df['TAIL_NUMBER'] = df['TAIL_NUMBER'].fillna('UNKNOWN')

    if verbose:
        print(f"  - {missing_before:,} valores preenchidos com 'UNKNOWN'\n")

    return df


# ============================================================================
# GRUPO 7: SCHEDULED_TIME
# ============================================================================

def handle_scheduled_time(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Trata valores ausentes em SCHEDULED_TIME.

    Estrategia: Remover as linhas (apenas 6 valores, 0.0001% do dataset)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de voos
    verbose : bool
        Se True, imprime informacoes

    Returns
    -------
    pd.DataFrame
        Dataset sem linhas com SCHEDULED_TIME ausente
    """
    df = df.copy()

    if 'SCHEDULED_TIME' not in df.columns:
        return df

    if verbose:
        print("GRUPO 7: Tratando SCHEDULED_TIME...")
        missing_before = df['SCHEDULED_TIME'].isnull().sum()
        rows_before = len(df)

    df = df[df['SCHEDULED_TIME'].notna()].copy()

    if verbose:
        rows_after = len(df)
        removed = rows_before - rows_after
        print(f"  - {removed} linhas removidas (0.0001% do dataset)\n")

    return df


# ============================================================================
# PIPELINE COMPLETO
# ============================================================================

def handle_all_missing_values(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Aplica todas as estrategias de tratamento de missing values.

    Pipeline completo (ordem importa!):
    1. GRUPO 7: Remover 6 linhas com SCHEDULED_TIME ausente
    2. GRUPO 1: Preencher CANCELLATION_REASON
    3. GRUPO 2: Preencher delays categoricos com 0
    4. GRUPO 6: Preencher TAIL_NUMBER com "UNKNOWN"
    5. GRUPO 5: Imputar horarios de partida
    6. GRUPO 4: Imputar horarios de chegada
    7. GRUPO 3: Calcular tempos de voo

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original de voos
    verbose : bool, default=True
        Se True, imprime informacoes sobre cada etapa

    Returns
    -------
    pd.DataFrame
        Dataset com todos os tratamentos aplicados
    """

    if verbose:
        print("=" * 80)
        print("PIPELINE COMPLETO DE TRATAMENTO DE MISSING VALUES")
        print("=" * 80)
        print()

        missing_before = df.isnull().sum().sum()
        print(f"Total de valores ausentes no inicio: {missing_before:,}\n")

    # Ordem de execucao conforme ESTRATEGIA_MISSING_VALUES.md
    df = handle_scheduled_time(df, verbose=verbose)           # GRUPO 7
    df = handle_cancellation_reason(df, verbose=verbose)      # GRUPO 1
    df = handle_categorical_delays(df, verbose=verbose)       # GRUPO 2
    df = handle_tail_number(df, verbose=verbose)              # GRUPO 6
    df = handle_departure_times(df, verbose=verbose)          # GRUPO 5
    df = handle_arrival_times(df, verbose=verbose)            # GRUPO 4
    df = handle_flight_times(df, verbose=verbose)             # GRUPO 3

    if verbose:
        missing_after = df.isnull().sum().sum()
        reduction = missing_before - missing_after
        reduction_pct = (reduction / missing_before * 100) if missing_before > 0 else 0

        print("=" * 80)
        print("RESUMO DO TRATAMENTO")
        print("=" * 80)
        print(f"Valores ausentes antes:  {missing_before:,}")
        print(f"Valores ausentes depois: {missing_after:,}")
        print(f"Reducao:                 {reduction:,} ({reduction_pct:.2f}%)")
        print()

    return df


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Exemplo de como usar as funcoes deste modulo.
    """

    print("\nTESTE DO MODULO MISSING_VALUES (VERSAO 2.0)\n")

    # Criar dataset de exemplo
    np.random.seed(42)
    df_test = pd.DataFrame({
        'CANCELLED': [0, 0, 1, 0, 1],
        'DIVERTED': [0, 0, 0, 1, 0],
        'AIRLINE': ['AA', 'AA', 'UA', 'AA', 'UA'],
        'ORIGIN_AIRPORT': ['JFK', 'JFK', 'LAX', 'JFK', 'LAX'],
        'DESTINATION_AIRPORT': ['LAX', 'LAX', 'JFK', 'LAX', 'JFK'],
        'DAY_OF_WEEK': [1, 1, 2, 1, 2],
        'DEPARTURE_TIME': [900, np.nan, np.nan, 1200, np.nan],
        'SCHEDULED_DEPARTURE': [900, 1000, 1100, 1200, 1300],
        'ARRIVAL_TIME': [1100, np.nan, np.nan, np.nan, np.nan],
        'SCHEDULED_ARRIVAL': [1100, 1200, 1300, 1400, 1500],
        'DEPARTURE_DELAY': [0, np.nan, np.nan, 5, np.nan],
        'ARRIVAL_DELAY': [0, np.nan, np.nan, np.nan, np.nan],
        'WEATHER_DELAY': [np.nan, np.nan, np.nan, 5, np.nan],
        'TAIL_NUMBER': ['N123', np.nan, 'N456', 'N789', np.nan],
        'CANCELLATION_REASON': [np.nan, np.nan, 'A', np.nan, 'B'],
        'SCHEDULED_TIME': [120, 120, 120, 120, 120]
    })

    print("Dataset de teste:")
    print(df_test)
    print()

    # Testar analise
    missing_report = analyze_missing_values(df_test)

    # Testar pipeline completo
    df_clean = handle_all_missing_values(df_test)

    print("Dataset apos tratamento:")
    print(df_clean)
    print()

    print("Teste concluido com sucesso!")
