"""
Módulo para visualização de dados do CADE.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any, Union
import os

def configurar_estilo_visualizacoes():
    """
    Configura o estilo padrão para as visualizações.
    """
    # Configurar estilo seaborn
    sns.set_style("whitegrid")
    
    # Configurar tamanho padrão das figuras
    plt.rcParams["figure.figsize"] = (12, 8)
    
    # Configurar fonte para suportar caracteres especiais
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Configurar tamanho dos textos
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

def gerar_histograma_multas(df: pd.DataFrame, 
                           coluna_percentual: str = 'percentual_multa',
                           output_path: str = None) -> str:
    """
    Gera histograma de percentuais de multa.
    
    Args:
        df (DataFrame): DataFrame com coluna de percentuais
        coluna_percentual (str): Nome da coluna que contém os percentuais
        output_path (str): Caminho para salvar o gráfico
        
    Returns:
        str: Caminho do arquivo salvo
    """
    # Configurar estilo
    configurar_estilo_visualizacoes()
    
    # Filtrar apenas valores não nulos
    percentuais = df[coluna_percentual].dropna()
    
    if len(percentuais) == 0:
        return None
    
    # Criar figura
    plt.figure()
    
    # Gerar histograma com KDE
    sns.histplot(percentuais, kde=True, bins=15)
    
    # Configurar rótulos
    plt.title('Distribuição de Percentuais de Multa sobre Faturamento')
    plt.xlabel('Percentual de Multa (%)')
    plt.ylabel('Frequência')
    plt.grid(True, alpha=0.3)
    
    # Adicionar linha vertical com a média
    media = percentuais.mean()
    plt.axvline(media, color='red', linestyle='--', 
                label=f'Média: {media:.2f}%')
    
    # Adicionar linha vertical com a mediana
    mediana = percentuais.median()
    plt.axvline(mediana, color='green', linestyle='-', 
                label=f'Mediana: {mediana:.2f}%')
    
    plt.legend()
    
    # Salvar figura se caminho for especificado
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    else:
        plt.show()
        plt.close()
        return None

def gerar_grafico_evolucao_temporal(df: pd.DataFrame,
                                   coluna_percentual: str = 'percentual_multa',
                                   coluna_ano: str = 'ano_documento',
                                   output_path: str = None) -> str:
    """
    Gera gráfico de evolução temporal dos percentuais de multa.
    
    Args:
        df (DataFrame): DataFrame com colunas de percentuais e ano
        coluna_percentual (str): Nome da coluna que contém os percentuais
        coluna_ano (str): Nome da coluna que contém o ano
        output_path (str): Caminho para salvar o gráfico
        
    Returns:
        str: Caminho do arquivo salvo
    """
    # Configurar estilo
    configurar_estilo_visualizacoes()
    
    # Filtrar apenas valores não nulos
    df_filtrado = df.dropna(subset=[coluna_percentual, coluna_ano])
    
    if len(df_filtrado) == 0:
        return None
    
    # Agrupar por ano e calcular estatísticas
    stats_por_ano = df_filtrado.groupby(coluna_ano)[coluna_percentual].agg([
        'mean', 'median', 'count'
    ]).reset_index()
    
    # Criar figura
    plt.figure()
    
    # Plotar média e mediana
    plt.plot(stats_por_ano[coluna_ano], stats_por_ano['mean'], 
             marker='o', linestyle='-', color='blue', 
             label='Média')
    
    plt.plot(stats_por_ano[coluna_ano], stats_por_ano['median'], 
             marker='s', linestyle='--', color='green', 
             label='Mediana')
    
    # Adicionar contagem como tamanho dos pontos
    tamanhos = stats_por_ano['count'] * 10  # Multiplicador para melhor visualização
    plt.scatter(stats_por_ano[coluna_ano], stats_por_ano['mean'], 
                s=tamanhos, alpha=0.5, color='blue')
    
    # Configurar rótulos
    plt.title('Evolução Temporal dos Percentuais de Multa')
    plt.xlabel('Ano')
    plt.ylabel('Percentual de Multa (%)')
    plt.grid(True, alpha=0.3)
    
    # Configurar eixo x para mostrar todos os anos
    plt.xticks(stats_por_ano[coluna_ano])
    
    # Adicionar anotações com contagem
    for i, row in stats_por_ano.iterrows():
        plt.annotate(f"n={int(row['count'])}", 
                    (row[coluna_ano], row['mean']),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
    
    plt.legend()
    
    # Salvar figura se caminho for especificado
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    else:
        plt.show()
        plt.close()
        return None

def gerar_boxplot_por_tipo_documento(df: pd.DataFrame,
                                    coluna_percentual: str = 'percentual_multa',
                                    coluna_tipo: str = 'descricao_tipo_documento',
                                    output_path: str = None) -> str:
    """
    Gera boxplot de percentuais de multa por tipo de documento.
    
    Args:
        df (DataFrame): DataFrame com colunas de percentuais e tipo de documento
        coluna_percentual (str): Nome da coluna que contém os percentuais
        coluna_tipo (str): Nome da coluna que contém o tipo de documento
        output_path (str): Caminho para salvar o gráfico
        
    Returns:
        str: Caminho do arquivo salvo
    """
    # Configurar estilo
    configurar_estilo_visualizacoes()
    
    # Filtrar apenas valores não nulos
    df_filtrado = df.dropna(subset=[coluna_percentual, coluna_tipo])
    
    if len(df_filtrado) == 0:
        return None
    
    # Criar figura
    plt.figure()
    
    # Gerar boxplot
    sns.boxplot(x=coluna_tipo, y=coluna_percentual, data=df_filtrado)
    
    # Adicionar pontos individuais
    sns.stripplot(x=coluna_tipo, y=coluna_percentual, data=df_filtrado, 
                 size=4, color=".3", alpha=0.6)
    
    # Configurar rótulos
    plt.title('Distribuição de Percentuais de Multa por Tipo de Documento')
    plt.xlabel('Tipo de Documento')
    plt.ylabel('Percentual de Multa (%)')
    plt.grid(True, alpha=0.3)
    
    # Rotacionar rótulos do eixo x para melhor legibilidade
    plt.xticks(rotation=45, ha='right')
    
    # Salvar figura se caminho for especificado
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    else:
        plt.show()
        plt.close()
        return None

def gerar_grafico_correlacao_dosimetria(df: pd.DataFrame,
                                       output_path: str = None) -> str:
    """
    Gera gráfico de correlação entre elementos de dosimetria e percentuais de multa.
    
    Args:
        df (DataFrame): DataFrame com colunas de dosimetria e percentuais
        output_path (str): Caminho para salvar o gráfico
        
    Returns:
        str: Caminho do arquivo salvo
    """
    # Configurar estilo
    configurar_estilo_visualizacoes()
    
    # Colunas de dosimetria numéricas ou booleanas
    colunas_dosimetria = [col for col in df.columns if col.startswith('dosimetria_')]
    
    # Filtrar apenas colunas numéricas ou booleanas
    colunas_validas = []
    for col in colunas_dosimetria:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            colunas_validas.append(col)
    
    if not colunas_validas or 'percentual_multa' not in df.columns:
        return None
    
    # Criar DataFrame para correlação
    df_corr = df[colunas_validas + ['percentual_multa']].copy()
    
    # Converter booleanos para inteiros (0/1)
    for col in colunas_validas:
        if pd.api.types.is_bool_dtype(df_corr[col]):
            df_corr[col] = df_corr[col].astype(int)
    
    # Calcular matriz de correlação
    corr_matrix = df_corr.corr()
    
    # Criar figura
    plt.figure(figsize=(10, 8))
    
    # Gerar heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
               fmt='.2f', linewidths=0.5)
    
    # Configurar rótulos
    plt.title('Correlação entre Elementos de Dosimetria e Percentuais de Multa')
    
    # Salvar figura se caminho for especificado
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    else:
        plt.show()
        plt.close()
        return None

def gerar_todas_visualizacoes(df: pd.DataFrame, diretorio_saida: str) -> Dict[str, str]:
    """
    Gera todas as visualizações e salva em um diretório.
    
    Args:
        df (DataFrame): DataFrame processado com dados do CADE
        diretorio_saida (str): Diretório para salvar as visualizações
        
    Returns:
        dict: Dicionário com caminhos dos arquivos salvos
    """
    # Criar diretório se não existir
    os.makedirs(diretorio_saida, exist_ok=True)
    
    caminhos = {}
    
    # Gerar histograma de multas
    caminho_histograma = os.path.join(diretorio_saida, 'histograma_multas.png')
    caminhos['histograma'] = gerar_histograma_multas(df, output_path=caminho_histograma)
    
    # Gerar gráfico de evolução temporal
    caminho_evolucao = os.path.join(diretorio_saida, 'evolucao_temporal.png')
    caminhos['evolucao_temporal'] = gerar_grafico_evolucao_temporal(df, output_path=caminho_evolucao)
    
    # Gerar boxplot por tipo de documento
    caminho_boxplot = os.path.join(diretorio_saida, 'boxplot_tipo_documento.png')
    caminhos['boxplot'] = gerar_boxplot_por_tipo_documento(df, output_path=caminho_boxplot)
    
    # Gerar gráfico de correlação
    caminho_correlacao = os.path.join(diretorio_saida, 'correlacao_dosimetria.png')
    caminhos['correlacao'] = gerar_grafico_correlacao_dosimetria(df, output_path=caminho_correlacao)
    
    return caminhos
