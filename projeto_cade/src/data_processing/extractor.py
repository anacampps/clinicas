"""
Módulo para extração de informações específicas dos documentos do CADE.
"""

import re
import pandas as pd
import numpy as np

def extrair_percentual_multa(texto):
    """
    Extrai percentuais de faturamento mencionados como multa no texto.
    
    Args:
        texto (str): Texto do documento
        
    Returns:
        list: Lista de percentuais encontrados
    """
    if not isinstance(texto, str):
        return None
    
    texto = texto.lower()
    
    # Padrões para identificar percentuais de multa
    padroes = [
        r'multa\s+de\s+(\d+[.,]?\d*)%\s+(?:do|de|sobre)?\s+(?:seu)?\s+faturamento',
        r'(\d+[.,]?\d*)%\s+(?:do|de|sobre)?\s+(?:seu)?\s+faturamento\s+(?:bruto|líquido)?',
        r'percentual\s+de\s+(\d+[.,]?\d*)%\s+(?:do|de|sobre)?\s+(?:seu)?\s+faturamento',
        r'pena\s+pecuniária\s+(?:de|no\s+valor\s+de)\s+(\d+[.,]?\d*)%\s+(?:do|de|sobre)?\s+(?:seu)?\s+faturamento',
        r'multa\s+(?:de|no\s+valor\s+de)\s+(\d+[.,]?\d*)%\s+(?:do|de|sobre)?\s+(?:seu)?\s+faturamento',
        r'aplicação\s+de\s+multa\s+de\s+(\d+[.,]?\d*)%\s+(?:do|de|sobre)?\s+(?:seu)?\s+faturamento',
        r'condenação\s+(?:de|ao\s+pagamento\s+de)\s+multa\s+de\s+(\d+[.,]?\d*)%\s+(?:do|de|sobre)?\s+(?:seu)?\s+faturamento',
        r'condenação\s+(?:de|ao\s+pagamento\s+de)\s+(\d+[.,]?\d*)%\s+(?:do|de|sobre)?\s+(?:seu)?\s+faturamento'
    ]
    
    resultados = []
    for padrao in padroes:
        matches = re.findall(padrao, texto)
        for match in matches:
            try:
                # Substituir vírgula por ponto e converter para float
                valor = float(match.replace(',', '.'))
                # Filtrar valores absurdos (acima de 100%)
                if valor <= 100:
                    resultados.append(valor)
            except (ValueError, TypeError):
                continue
    
    return resultados if resultados else None

def extrair_valor_multa_reais(texto):
    """
    Extrai valores monetários de multas em reais.
    
    Args:
        texto (str): Texto do documento
        
    Returns:
        list: Lista de valores em reais encontrados
    """
    if not isinstance(texto, str):
        return None
    
    texto = texto.lower()
    
    # Padrões para identificar valores monetários
    padroes = [
        r'multa\s+de\s+r\$\s*(\d+[.,]?\d*(?:\.\d+)*)',
        r'multa\s+no\s+valor\s+de\s+r\$\s*(\d+[.,]?\d*(?:\.\d+)*)',
        r'pena\s+pecuniária\s+de\s+r\$\s*(\d+[.,]?\d*(?:\.\d+)*)',
        r'condenação\s+(?:de|ao\s+pagamento\s+de)\s+r\$\s*(\d+[.,]?\d*(?:\.\d+)*)'
    ]
    
    resultados = []
    for padrao in padroes:
        matches = re.findall(padrao, texto)
        for match in matches:
            try:
                # Remover pontos de separação de milhar e substituir vírgula por ponto
                valor_str = match.replace('.', '')
                valor_str = valor_str.replace(',', '.')
                valor = float(valor_str)
                resultados.append(valor)
            except (ValueError, TypeError):
                continue
    
    return resultados if resultados else None

def extrair_elementos_dosimetria(texto):
    """
    Extrai elementos relacionados à dosimetria da pena.
    
    Args:
        texto (str): Texto do documento
        
    Returns:
        dict: Dicionário com elementos de dosimetria encontrados
    """
    if not isinstance(texto, str):
        return {}
    
    texto = texto.lower()
    
    elementos = {
        'reincidencia': False,
        'boa_fe': False,
        'ma_fe': False,
        'cooperacao': False,
        'gravidade': None,
        'duracao_conduta': None
    }
    
    # Verificar reincidência
    if re.search(r'reincid[êe]ncia', texto) or re.search(r'reincidente', texto):
        elementos['reincidencia'] = True
    
    # Verificar boa-fé
    if re.search(r'boa[- ]f[ée]', texto):
        elementos['boa_fe'] = True
    
    # Verificar má-fé
    if re.search(r'm[áa][- ]f[ée]', texto):
        elementos['ma_fe'] = True
    
    # Verificar cooperação
    if re.search(r'cooper[ao][çc][ãa]o', texto) or re.search(r'colabor[ao][çc][ãa]o', texto):
        elementos['cooperacao'] = True
    
    # Extrair gravidade
    gravidade_match = re.search(r'(alta|elevada|grave|baixa|leve|média|moderada)\s+gravidade', texto)
    if gravidade_match:
        elementos['gravidade'] = gravidade_match.group(1)
    
    # Extrair duração da conduta
    duracao_match = re.search(r'conduta\s+(?:por|durante)\s+(\d+)\s+(anos?|meses?|dias?)', texto)
    if duracao_match:
        valor = int(duracao_match.group(1))
        unidade = duracao_match.group(2)
        
        # Normalizar para meses
        if 'ano' in unidade:
            valor *= 12
        elif 'dia' in unidade:
            valor = valor / 30  # Aproximação
        
        elementos['duracao_conduta'] = valor
    
    return elementos

def aplicar_extracao_ao_dataframe(df, coluna_texto='texto_completo'):
    """
    Aplica funções de extração ao DataFrame.
    
    Args:
        df (DataFrame): DataFrame com coluna de texto
        coluna_texto (str): Nome da coluna que contém o texto para análise
        
    Returns:
        DataFrame: DataFrame com colunas adicionais de extração
    """
    # Criar cópia para não modificar o original
    df_resultado = df.copy()
    
    # Extrair percentuais de multa
    df_resultado['percentuais_multa'] = df_resultado[coluna_texto].apply(extrair_percentual_multa)
    
    # Extrair o primeiro percentual (geralmente o mais relevante)
    df_resultado['percentual_multa'] = df_resultado['percentuais_multa'].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
    )
    
    # Extrair valores monetários
    df_resultado['valores_multa_reais'] = df_resultado[coluna_texto].apply(extrair_valor_multa_reais)
    
    # Extrair o primeiro valor monetário
    df_resultado['valor_multa_reais'] = df_resultado['valores_multa_reais'].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
    )
    
    # Extrair elementos de dosimetria
    df_resultado['elementos_dosimetria'] = df_resultado[coluna_texto].apply(extrair_elementos_dosimetria)
    
    # Expandir elementos de dosimetria em colunas separadas
    for elemento in ['reincidencia', 'boa_fe', 'ma_fe', 'cooperacao', 'gravidade', 'duracao_conduta']:
        df_resultado[f'dosimetria_{elemento}'] = df_resultado['elementos_dosimetria'].apply(
            lambda x: x.get(elemento) if isinstance(x, dict) else None
        )
    # Filtrar apenas os documentos com condenação
    df_condenados = df_resultado[
    (df_resultado['percentual_multa'].notnull()) | 
    (df_resultado['valor_multa_reais'].notnull())
]

    return df_condenados
