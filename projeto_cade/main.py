"""
Script principal para processamento e análise de documentos do CADE.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_processing.extractor import aplicar_extracao_ao_dataframe
from src.data_processing.descriptive import gerar_relatorio_estatistico
from src.data_processing.visualization import gerar_todas_visualizacoes
from src.data_processing.inference import treinar_modelo_inferencial, visualizar_importancia_features, visualizar_matriz_confusao

def processar_dados_cade(caminho_arquivo, diretorio_saida='output'):
    """
    Processa os dados do CADE e gera análises.
    
    Args:
        caminho_arquivo (str): Caminho para o arquivo Parquet
        diretorio_saida (str): Diretório para salvar os resultados
    
    Returns:
        dict: Dicionário com resultados do processamento
    """
    print(f"Carregando dados do arquivo: {caminho_arquivo}")
    # Carregar dados
    df = pd.read_parquet(caminho_arquivo)
    print(f"Dados carregados com sucesso. Shape: {df.shape}")
    
    # Criar diretório de saída se não existir
    os.makedirs(diretorio_saida, exist_ok=True)
    
    # Passo 1: Preparação inicial dos dados
    print("Preparando dados iniciais...")
    df['descricao_tipo_processo'] = df.get('descricao_tipo_processo', '').fillna('')
    df['ano_documento'] = pd.to_numeric(df.get('ano_documento', 0), errors='coerce')
    
    # Passo 2: Filtrar processos administrativos
    print("Filtrando processos administrativos...")
    df_admin = df[
        df['descricao_tipo_processo'].str.lower().str.contains('processo administrativo') &
        (df['ano_documento'] > 2012)
    ].copy()
    print(f"Processos administrativos filtrados: {df_admin.shape[0]}")
    
    # Passo 3: Filtrar documentos de voto
    print("Filtrando documentos de voto...")
    df_votos = df_admin[df_admin['descricao_tipo_documento'].str.strip().str.lower().isin([
        "voto",
        "voto processo administrativo",
        "voto embargos de declaração"
    ])].copy()
    print(f"Documentos de voto filtrados: {df_votos.shape[0]}")
    
    # Passo 4: Verificar e concatenar colunas de texto
    print("Concatenando colunas de texto...")
    colunas_texto_candidatas = [
        'corpo_texto_formatado',
        'corpo_texto',
        'conteudo',
        'descricao_titulo_documento',
        'descricao_tipo_documento'
    ]
    colunas_existentes = [col for col in colunas_texto_candidatas if col in df_votos.columns]
    
    if not colunas_existentes:
        raise ValueError("Nenhuma das colunas textuais candidatas existe no DataFrame.")
    
    df_votos['texto_completo'] = df_votos[colunas_existentes].fillna('').astype(str).agg(' '.join, axis=1)
    
    # Passo 5: Identificar pessoa jurídica
    print("Identificando pessoas jurídicas...")
    
    # Passo 6: Extrair percentuais de multa e elementos de dosimetria
    print("Extraindo percentuais de multa e elementos de dosimetria...")
    df_processado = aplicar_extracao_ao_dataframe(df_votos)
    print(f"Extração concluída. Shape: {df_processado.shape}")
    
    # Salvar DataFrame processado
    caminho_df_processado = os.path.join(diretorio_saida, 'df_processado.csv')
    df_processado.to_csv(caminho_df_processado, index=False)
    print(f"DataFrame processado salvo em: {caminho_df_processado}")
    
    # Passo 7: Gerar estatísticas descritivas
    print("Gerando estatísticas descritivas...")
    estatisticas = gerar_relatorio_estatistico(df_processado)
    
    # Salvar estatísticas em formato CSV
    caminho_estatisticas_gerais = os.path.join(diretorio_saida, 'estatisticas_gerais.csv')
    pd.DataFrame([estatisticas['estatisticas_gerais']]).to_csv(caminho_estatisticas_gerais, index=False)
    
    caminho_estatisticas_ano = os.path.join(diretorio_saida, 'estatisticas_por_ano.csv')
    pd.DataFrame(estatisticas['estatisticas_por_ano']).to_csv(caminho_estatisticas_ano, index=False)
    
    caminho_estatisticas_tipo = os.path.join(diretorio_saida, 'estatisticas_por_tipo.csv')
    pd.DataFrame(estatisticas['estatisticas_por_tipo']).to_csv(caminho_estatisticas_tipo, index=False)
    
    print(f"Estatísticas salvas em: {diretorio_saida}")
    
    # Passo 8: Gerar visualizações
    print("Gerando visualizações...")
    diretorio_visualizacoes = os.path.join(diretorio_saida, 'visualizacoes')
    caminhos_visualizacoes = gerar_todas_visualizacoes(df_processado, diretorio_visualizacoes)
    print(f"Visualizações salvas em: {diretorio_visualizacoes}")
    
    # Passo 9: Treinar modelo inferencial
    print("Treinando modelo inferencial...")
    resultados_modelo = treinar_modelo_inferencial(df_processado)
    
    # Salvar resultados do modelo
    if resultados_modelo['status'] == 'sucesso':
        # Visualizar importância das features
        caminho_importancia = os.path.join(diretorio_visualizacoes, 'importancia_features.png')
        visualizar_importancia_features(resultados_modelo, caminho_importancia)
        
        # Visualizar matriz de confusão
        caminho_matriz = os.path.join(diretorio_visualizacoes, 'matriz_confusao.png')
        visualizar_matriz_confusao(resultados_modelo, caminho_matriz)
        
        # Salvar relatório de classificação
        caminho_report = os.path.join(diretorio_saida, 'relatorio_classificacao.csv')
        pd.DataFrame(resultados_modelo['report']).transpose().to_csv(caminho_report)
        
        print(f"Resultados do modelo salvos em: {diretorio_saida}")
        print(f"Acurácia do modelo: {resultados_modelo['accuracy']:.4f}")
    else:
        print(f"Erro ao treinar modelo: {resultados_modelo['mensagem']}")
    
    return {
        'df_processado': df_processado,
        'estatisticas': estatisticas,
        'caminhos_visualizacoes': caminhos_visualizacoes,
        'resultados_modelo': resultados_modelo
    }

if __name__ == "__main__":
    # Definir caminhos
    caminho_arquivo = "data/cade_clinicas_2.parquet"  # Arquivo Parquet na pasta data
    diretorio_saida = "output"
    
    # Processar dados
    resultados = processar_dados_cade(caminho_arquivo, diretorio_saida)
    
    print("\nProcessamento concluído com sucesso!")
