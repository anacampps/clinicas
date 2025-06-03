"""
Módulo para cálculo de estatísticas descritivas sobre os dados do CADE.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Union

def calcular_estatisticas_multas(df: pd.DataFrame, coluna_percentual: str = 'percentual_multa') -> Dict[str, Any]:
    """
    Calcula estatísticas descritivas sobre percentuais de multa.
    
    Args:
        df (DataFrame): DataFrame com coluna de percentuais de multa
        coluna_percentual (str): Nome da coluna que contém os percentuais
        
    Returns:
        dict: Dicionário com estatísticas calculadas
    """
    # Filtrar apenas valores não nulos
    percentuais = df[coluna_percentual].dropna()
    
    if len(percentuais) == 0:
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'std': None,
            'min': None,
            'max': None,
            'quartiles': None
        }
    
    return {
        'count': len(percentuais),
        'mean': percentuais.mean(),
        'median': percentuais.median(),
        'std': percentuais.std(),
        'min': percentuais.min(),
        'max': percentuais.max(),
        'quartiles': [percentuais.quantile(0.25), percentuais.quantile(0.5), percentuais.quantile(0.75)]
    }

def calcular_estatisticas_por_ano(df: pd.DataFrame, 
                                 coluna_percentual: str = 'percentual_multa',
                                 coluna_ano: str = 'ano_documento') -> pd.DataFrame:
    """
    Calcula estatísticas de percentuais de multa agrupadas por ano.
    
    Args:
        df (DataFrame): DataFrame com colunas de percentuais e ano
        coluna_percentual (str): Nome da coluna que contém os percentuais
        coluna_ano (str): Nome da coluna que contém o ano
        
    Returns:
        DataFrame: DataFrame com estatísticas por ano
    """
    # Filtrar apenas linhas com percentuais não nulos
    df_filtrado = df.dropna(subset=[coluna_percentual])
    
    if len(df_filtrado) == 0:
        return pd.DataFrame()
    
    # Agrupar por ano e calcular estatísticas
    stats_por_ano = df_filtrado.groupby(coluna_ano)[coluna_percentual].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).reset_index()
    
    return stats_por_ano

def calcular_estatisticas_por_tipo_documento(df: pd.DataFrame,
                                           coluna_percentual: str = 'percentual_multa',
                                           coluna_tipo: str = 'descricao_tipo_documento') -> pd.DataFrame:
    """
    Calcula estatísticas de percentuais de multa agrupadas por tipo de documento.
    
    Args:
        df (DataFrame): DataFrame com colunas de percentuais e tipo de documento
        coluna_percentual (str): Nome da coluna que contém os percentuais
        coluna_tipo (str): Nome da coluna que contém o tipo de documento
        
    Returns:
        DataFrame: DataFrame com estatísticas por tipo de documento
    """
    # Filtrar apenas linhas com percentuais não nulos
    df_filtrado = df.dropna(subset=[coluna_percentual])
    
    if len(df_filtrado) == 0:
        return pd.DataFrame()
    
    # Agrupar por tipo de documento e calcular estatísticas
    stats_por_tipo = df_filtrado.groupby(coluna_tipo)[coluna_percentual].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).reset_index()
    
    return stats_por_tipo

def calcular_distribuicao_percentuais(df: pd.DataFrame, 
                                     coluna_percentual: str = 'percentual_multa',
                                     bins: int = 10) -> Dict[str, List[float]]:
    """
    Calcula a distribuição de frequência dos percentuais de multa.
    
    Args:
        df (DataFrame): DataFrame com coluna de percentuais
        coluna_percentual (str): Nome da coluna que contém os percentuais
        bins (int): Número de intervalos para a distribuição
        
    Returns:
        dict: Dicionário com intervalos e contagens
    """
    # Filtrar apenas valores não nulos
    percentuais = df[coluna_percentual].dropna()
    
    if len(percentuais) == 0:
        return {'intervalos': [], 'contagens': []}
    
    # Calcular histograma
    contagens, intervalos = np.histogram(percentuais, bins=bins)
    
    # Criar rótulos para os intervalos
    rotulos_intervalos = [f"{intervalos[i]:.2f}-{intervalos[i+1]:.2f}%" for i in range(len(intervalos)-1)]
    
    return {
        'intervalos': rotulos_intervalos,
        'contagens': contagens.tolist()
    }

def calcular_correlacao_dosimetria_multa(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula correlações entre elementos de dosimetria e percentuais de multa.
    
    Args:
        df (DataFrame): DataFrame com colunas de dosimetria e percentuais
        
    Returns:
        dict: Dicionário com correlações calculadas
    """
    # Colunas de dosimetria numéricas ou booleanas
    colunas_dosimetria = [col for col in df.columns if col.startswith('dosimetria_')]
    
    # Filtrar apenas colunas numéricas ou booleanas
    colunas_validas = []
    for col in colunas_dosimetria:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            colunas_validas.append(col)
    
    if not colunas_validas or 'percentual_multa' not in df.columns:
        return {}
    
    # Calcular correlações
    correlacoes = {}
    for col in colunas_validas:
        # Converter booleanos para inteiros (0/1)
        if pd.api.types.is_bool_dtype(df[col]):
            serie_temp = df[col].astype(int)
        else:
            serie_temp = df[col]
        
        # Calcular correlação apenas para valores não nulos
        mask = (~df['percentual_multa'].isna()) & (~serie_temp.isna())
        if mask.sum() > 1:  # Precisa de pelo menos 2 pontos para correlação
            correlacao = df.loc[mask, 'percentual_multa'].corr(serie_temp[mask])
            correlacoes[col] = correlacao
    
    return correlacoes

def gerar_relatorio_estatistico(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Gera um relatório estatístico completo sobre os dados.
    
    Args:
        df (DataFrame): DataFrame processado com dados do CADE
        
    Returns:
        dict: Dicionário com todas as estatísticas calculadas
    """
    relatorio = {}
    
    # Estatísticas gerais sobre percentuais de multa
    relatorio['estatisticas_gerais'] = calcular_estatisticas_multas(df)
    
    # Estatísticas por ano
    relatorio['estatisticas_por_ano'] = calcular_estatisticas_por_ano(df).to_dict('records')
    
    # Estatísticas por tipo de documento
    relatorio['estatisticas_por_tipo'] = calcular_estatisticas_por_tipo_documento(df).to_dict('records')
    
    # Distribuição de percentuais
    relatorio['distribuicao_percentuais'] = calcular_distribuicao_percentuais(df)
    
    # Correlações com dosimetria
    relatorio['correlacoes_dosimetria'] = calcular_correlacao_dosimetria_multa(df)
    
    # Contagem de condenações vs. arquivamentos
    if 'decisao_tribunal' in df.columns:
        decisoes = df['decisao_tribunal'].apply(lambda x: str(x).lower())
        condenacoes = decisoes.str.contains('condenação').sum()
        arquivamentos = decisoes.str.contains('arquivamento').sum()
        relatorio['contagem_decisoes'] = {
            'condenacoes': condenacoes,
            'arquivamentos': arquivamentos,
            'outros': len(df) - condenacoes - arquivamentos
        }
    
    return relatorio
