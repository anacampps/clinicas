"""
Módulo para criação de modelo inferencial para análise de dados do CADE.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List, Optional
import os

def preparar_dados_para_modelo(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara os dados para o modelo inferencial.
    
    Args:
        df (DataFrame): DataFrame processado com dados do CADE
        
    Returns:
        tuple: (X, y) onde X são as features e y é a variável alvo
    """
    # Criar cópia para não modificar o original
    df_modelo = df.copy()
    
    # Criar variável alvo: condenação (1) ou não (0)
    if 'decisao_tribunal' in df_modelo.columns:
        df_modelo['condenacao'] = df_modelo['decisao_tribunal'].apply(
            lambda x: 1 if isinstance(x, list) and any('condenação' in str(item).lower() for item in x) else 0
        )
    else:
        # Se não houver coluna de decisão, não é possível criar o modelo
        return None, None
    
    # Selecionar features potencialmente relevantes
    features_numericas = [
        'ano_documento',
        'dosimetria_duracao_conduta'
    ]
    
    features_categoricas = [
        'descricao_tipo_documento',
        'dosimetria_gravidade'
    ]
    
    features_binarias = [
        'dosimetria_reincidencia',
        'dosimetria_boa_fe',
        'dosimetria_ma_fe',
        'dosimetria_cooperacao'
    ]
    
    # Filtrar apenas features que existem no DataFrame
    features_numericas = [col for col in features_numericas if col in df_modelo.columns]
    features_categoricas = [col for col in features_categoricas if col in df_modelo.columns]
    features_binarias = [col for col in features_binarias if col in df_modelo.columns]
    
    # Converter features binárias para inteiros (0/1)
    for col in features_binarias:
        df_modelo[col] = df_modelo[col].astype(float)
    
    # Combinar todas as features
    todas_features = features_numericas + features_categoricas + features_binarias
    
    if not todas_features:
        return None, None
    
    # Selecionar apenas linhas com variável alvo não nula
    mask = ~df_modelo['condenacao'].isna()
    X = df_modelo.loc[mask, todas_features]
    y = df_modelo.loc[mask, 'condenacao']
    
    return X, y

def criar_pipeline_modelo(features_numericas: List[str], 
                         features_categoricas: List[str]) -> Pipeline:
    """
    Cria pipeline de pré-processamento e modelo.
    
    Args:
        features_numericas (list): Lista de colunas numéricas
        features_categoricas (list): Lista de colunas categóricas
        
    Returns:
        Pipeline: Pipeline scikit-learn com pré-processamento e modelo
    """
    # Preprocessadores para diferentes tipos de features
    preprocessador_numerico = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessador_categorico = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combinar preprocessadores
    preprocessador = ColumnTransformer(transformers=[
        ('num', preprocessador_numerico, features_numericas),
        ('cat', preprocessador_categorico, features_categoricas)
    ])
    
    # Criar pipeline completo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessador),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    return pipeline

def treinar_modelo_inferencial(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Treina um modelo inferencial para identificar fatores associados a condenações.
    
    Args:
        df (DataFrame): DataFrame processado com dados do CADE
        
    Returns:
        dict: Dicionário com modelo treinado e métricas de avaliação
    """
    # Preparar dados
    X, y = preparar_dados_para_modelo(df)
    
    if X is None or y is None or len(X) < 10:  # Verificar se há dados suficientes
        return {
            'status': 'erro',
            'mensagem': 'Dados insuficientes para treinar o modelo'
        }
    
    # Separar features numéricas e categóricas
    features_numericas = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    features_categoricas = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Criar pipeline
    pipeline = criar_pipeline_modelo(features_numericas, features_categoricas)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Treinar modelo
    pipeline.fit(X_train, y_train)
    
    # Avaliar modelo
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Extrair importância das features
    modelo = pipeline.named_steps['classifier']
    
    # Obter nomes das features após one-hot encoding
    preprocessador = pipeline.named_steps['preprocessor']
    feature_names = []
    
    # Para features numéricas, os nomes permanecem os mesmos
    if features_numericas:
        feature_names.extend(features_numericas)
    
    # Para features categóricas, precisamos obter os nomes após one-hot encoding
    if features_categoricas:
        encoder = preprocessador.transformers_[1][1].named_steps['onehot']
        categorias = encoder.categories_
        for i, cat in enumerate(categorias):
            for categoria in cat:
                feature_names.append(f"{features_categoricas[i]}_{categoria}")
    
    # Verificar se o número de features corresponde à importância
    if len(feature_names) == len(modelo.feature_importances_):
        importancias = dict(zip(feature_names, modelo.feature_importances_))
    else:
        importancias = {f"feature_{i}": imp for i, imp in enumerate(modelo.feature_importances_)}
    
    # Ordenar importâncias
    importancias = dict(sorted(importancias.items(), key=lambda x: x[1], reverse=True))
    
    return {
        'status': 'sucesso',
        'pipeline': pipeline,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'feature_importance': importancias
    }

def visualizar_importancia_features(resultados_modelo: Dict[str, Any], 
                                   output_path: str = None) -> str:
    """
    Visualiza a importância das features do modelo.
    
    Args:
        resultados_modelo (dict): Dicionário com resultados do modelo
        output_path (str): Caminho para salvar o gráfico
        
    Returns:
        str: Caminho do arquivo salvo
    """
    if resultados_modelo['status'] != 'sucesso':
        return None
    
    # Extrair importância das features
    importancias = resultados_modelo['feature_importance']
    
    # Converter para DataFrame para facilitar a visualização
    df_importancias = pd.DataFrame({
        'Feature': list(importancias.keys()),
        'Importância': list(importancias.values())
    }).sort_values('Importância', ascending=False).head(15)  # Mostrar apenas as 15 mais importantes
    
    # Criar figura
    plt.figure(figsize=(12, 8))
    
    # Gerar gráfico de barras horizontais
    sns.barplot(x='Importância', y='Feature', data=df_importancias)
    
    # Configurar rótulos
    plt.title('Importância das Features para Condenações')
    plt.xlabel('Importância Relativa')
    plt.ylabel('Feature')
    plt.grid(True, alpha=0.3)
    
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

def visualizar_matriz_confusao(resultados_modelo: Dict[str, Any],
                              output_path: str = None) -> str:
    """
    Visualiza a matriz de confusão do modelo.
    
    Args:
        resultados_modelo (dict): Dicionário com resultados do modelo
        output_path (str): Caminho para salvar o gráfico
        
    Returns:
        str: Caminho do arquivo salvo
    """
    if resultados_modelo['status'] != 'sucesso':
        return None
    
    # Extrair matriz de confusão
    conf_matrix = np.array(resultados_modelo['confusion_matrix'])
    
    # Criar figura
    plt.figure(figsize=(8, 6))
    
    # Gerar heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Não Condenação', 'Condenação'],
               yticklabels=['Não Condenação', 'Condenação'])
    
    # Configurar rótulos
    plt.title('Matriz de Confusão')
    plt.xlabel('Predição')
    plt.ylabel('Valor Real')
    
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
