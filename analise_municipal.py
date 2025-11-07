#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISE MUNICIPAL: VOTO EM LULA E VOTAÇÃO EM DEPUTADAS FEDERAIS (2022)

Autor: Isabela de Almeida
Data: Novembro 2025
Disciplina: [Nome da disciplina]
Professora: Beatriz Sanchez

Descrição:
Este script analisa se municípios que votaram majoritariamente em Lula 
no 2º turno de 2022 (>50%) também direcionaram maior votação para 
candidatas mulheres nas eleições para Deputado Federal.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configurações de exibição
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.2f}'.format)

print("="*80)
print("ANÁLISE: MUNICÍPIOS LULA E VOTAÇÃO EM DEPUTADAS FEDERAIS")
print("="*80)

# ==============================================================================
# PARTE 1: PROCESSAR DADOS PRESIDENCIAIS (2º TURNO)
# ==============================================================================

print("\n[1] Carregando dados presidenciais (2º turno)...")

# Carregar dados
# IMPORTANTE: Você precisa baixar este arquivo antes:
# https://cdn.tse.jus.br/estatistica/sead/odsele/votacao_secao/votacao_secao_2022_BR.zip
df_pres = pd.read_csv('votacao_secao_2022_BR.csv', 
                       encoding='latin1', 
                       sep=';',
                       dtype={'NR_VOTAVEL': str})  # Importante: tratar como string

# Filtrar 2º turno e cargo de Presidente
df_pres_2t = df_pres[
    (df_pres['NR_TURNO'] == 2) & 
    (df_pres['CD_CARGO_PERGUNTA'] == 1)
].copy()

print(f"   Total de registros (2º turno Presidente): {len(df_pres_2t):,}")

# Agregar por município e candidato
votos_mun = df_pres_2t.groupby(
    ['SG_UF', 'CD_MUNICIPIO', 'NM_MUNICIPIO', 'NR_VOTAVEL']
)['QT_VOTOS'].sum().reset_index()

# Pivotar para ter Lula e Bolsonaro em colunas
votos_pivot = votos_mun.pivot_table(
    index=['SG_UF', 'CD_MUNICIPIO', 'NM_MUNICIPIO'],
    columns='NR_VOTAVEL',
    values='QT_VOTOS',
    fill_value=0
).reset_index()

# Renomear colunas
votos_pivot.columns.name = None
votos_pivot = votos_pivot.rename(columns={
    '13': 'votos_lula',
    '22': 'votos_bolsonaro'
})

# Calcular percentuais
votos_pivot['votos_validos'] = (
    votos_pivot['votos_lula'] + votos_pivot['votos_bolsonaro']
)
votos_pivot['perc_lula'] = (
    votos_pivot['votos_lula'] / votos_pivot['votos_validos'] * 100
)

# Classificar municípios
votos_pivot['grupo_lula'] = (votos_pivot['perc_lula'] > 50).astype(int)

print(f"   Municípios processados: {len(votos_pivot):,}")
print(f"   Municípios Lula >50%: {votos_pivot['grupo_lula'].sum():,}")
print(f"   Municípios Lula <=50%: {(1-votos_pivot['grupo_lula']).sum():,}")

# ==============================================================================
# PARTE 2: PROCESSAR DADOS DE CANDIDATOS (GÊNERO)
# ==============================================================================

print("\n[2] Carregando dados de candidatos...")

# Baixar: https://cdn.tse.jus.br/estatistica/sead/odsele/consulta_cand/consulta_cand_2022.zip
df_cand = pd.read_csv('consulta_cand_2022_BRASIL.csv', 
                      encoding='latin1', 
                      sep=';',
                      dtype={'NR_CANDIDATO': str})

# Filtrar Deputado Federal
df_dep_fed = df_cand[
    (df_cand['CD_CARGO'] == 6) & 
    (df_cand['NR_TURNO'] == 1)
].copy()

# Criar indicador de mulher
df_dep_fed['eh_mulher'] = (df_dep_fed['DS_GENERO'] == 'FEMININO').astype(int)

# Mapeamento candidato -> gênero
map_genero = df_dep_fed[['SG_UF', 'NR_CANDIDATO', 'eh_mulher']].copy()

print(f"   Total de candidatos a Dep. Federal: {len(map_genero):,}")
print(f"   Candidatas mulheres: {map_genero['eh_mulher'].sum():,}")
print(f"   Candidatos homens: {(1-map_genero['eh_mulher']).sum():,}")

# ==============================================================================
# PARTE 3: PROCESSAR VOTAÇÃO DEPUTADO FEDERAL POR UF
# ==============================================================================

print("\n[3] Carregando votação em Deputados Federais por UF...")

ufs = ['AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 
       'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 
       'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO']

lista_dfs = []

for uf in ufs:
    try:
        print(f"   Processando {uf}...", end=' ')
        
        # Baixar arquivos em:
        # https://cdn.tse.jus.br/estatistica/sead/odsele/votacao_secao/votacao_secao_2022_{UF}.zip
        df_uf = pd.read_csv(f'votacao_secao_2022_{uf}.csv', 
                           encoding='latin1', 
                           sep=';',
                           dtype={'NR_VOTAVEL': str})
        
        # Filtrar Deputado Federal, 1º turno
        df_uf_dep = df_uf[
            (df_uf['NR_TURNO'] == 1) & 
            (df_uf['CD_CARGO_PERGUNTA'] == 6)
        ].copy()
        
        # Agregar por município e candidato
        votos_dep_mun = df_uf_dep.groupby(
            ['SG_UF', 'CD_MUNICIPIO', 'NM_MUNICIPIO', 'NR_VOTAVEL']
        )['QT_VOTOS'].sum().reset_index()
        
        lista_dfs.append(votos_dep_mun)
        print(f"OK ({len(votos_dep_mun):,} registros)")
        
    except FileNotFoundError:
        print(f"ARQUIVO NÃO ENCONTRADO")
        continue

# Concatenar todos os dados
df_dep_todos = pd.concat(lista_dfs, ignore_index=True)
print(f"\n   Total de registros de votação: {len(df_dep_todos):,}")

# Merge com gênero dos candidatos
df_dep_todos = df_dep_todos.merge(
    map_genero,
    left_on=['SG_UF', 'NR_VOTAVEL'],
    right_on=['SG_UF', 'NR_CANDIDATO'],
    how='left'
)

# Preencher NaN (votos de legenda, brancos, nulos) como 0
df_dep_todos['eh_mulher'] = df_dep_todos['eh_mulher'].fillna(0)

# ==============================================================================
# PARTE 4: AGREGAR VOTAÇÃO EM MULHERES POR MUNICÍPIO
# ==============================================================================

print("\n[4] Agregando votação em mulheres por município...")

# Calcular votos em mulheres por município
df_dep_todos['votos_mulher'] = df_dep_todos['eh_mulher'] * df_dep_todos['QT_VOTOS']

votos_genero = df_dep_todos.groupby(
    ['SG_UF', 'CD_MUNICIPIO', 'NM_MUNICIPIO']
).agg({
    'QT_VOTOS': 'sum',
    'votos_mulher': 'sum'
}).reset_index()

votos_genero.columns = ['SG_UF', 'CD_MUNICIPIO', 'NM_MUNICIPIO', 
                        'total_votos_dep', 'votos_em_mulheres']

# Calcular percentual
votos_genero['perc_votos_mulheres'] = (
    votos_genero['votos_em_mulheres'] / votos_genero['total_votos_dep'] * 100
)

print(f"   Municípios com dados de deputadas: {len(votos_genero):,}")

# ==============================================================================
# PARTE 5: JUNTAR TUDO E ANALISAR
# ==============================================================================

print("\n[5] Juntando dados e preparando análise...")

# Merge
df_final = votos_pivot.merge(
    votos_genero,
    on=['SG_UF', 'CD_MUNICIPIO', 'NM_MUNICIPIO'],
    how='inner'
)

print(f"   Dataset final: {len(df_final):,} municípios")

# Separar grupos
grupo_lula = df_final[df_final['grupo_lula'] == 1]['perc_votos_mulheres']
grupo_nao_lula = df_final[df_final['grupo_lula'] == 0]['perc_votos_mulheres']

# ==============================================================================
# PARTE 6: ESTATÍSTICAS DESCRITIVAS
# ==============================================================================

print("\n" + "="*80)
print("RESULTADOS DA ANÁLISE")
print("="*80)

print(f"\nTAMANHO DOS GRUPOS:")
print(f"  Municípios Lula >50%: {len(grupo_lula):,}")
print(f"  Municípios Lula <=50%: {len(grupo_nao_lula):,}")

print(f"\nMÉDIA DE VOTOS EM DEPUTADAS FEDERAIS:")
print(f"  Grupo Lula >50%: {grupo_lula.mean():.2f}%")
print(f"  Grupo Lula <=50%: {grupo_nao_lula.mean():.2f}%")
print(f"  Diferença: {grupo_lula.mean() - grupo_nao_lula.mean():.2f} pontos percentuais")

print(f"\nMEDIANA DE VOTOS EM DEPUTADAS FEDERAIS:")
print(f"  Grupo Lula >50%: {grupo_lula.median():.2f}%")
print(f"  Grupo Lula <=50%: {grupo_nao_lula.median():.2f}%")

print(f"\nDESVIO PADRÃO:")
print(f"  Grupo Lula >50%: {grupo_lula.std():.2f}%")
print(f"  Grupo Lula <=50%: {grupo_nao_lula.std():.2f}%")

# ==============================================================================
# PARTE 7: TESTES ESTATÍSTICOS
# ==============================================================================

print("\n" + "-"*80)
print("TESTES ESTATÍSTICOS")
print("-"*80)

# Teste t
t_stat, p_value = stats.ttest_ind(grupo_lula, grupo_nao_lula)

print(f"\nTESTE T DE STUDENT (independente):")
print(f"  Estatística t: {t_stat:.4f}")
print(f"  P-valor: {p_value:.6f}")

if p_value < 0.05:
    print(f"  *** RESULTADO: Diferença ESTATISTICAMENTE SIGNIFICATIVA (p < 0.05) ***")
    if grupo_lula.mean() > grupo_nao_lula.mean():
        print(f"  Municípios Lula votaram MAIS em deputadas federais")
    else:
        print(f"  Municípios Lula votaram MENOS em deputadas federais")
else:
    print(f"  Resultado: Diferença NÃO estatisticamente significativa (p >= 0.05)")

# Teste de Mann-Whitney (não-paramétrico)
u_stat, p_value_mw = stats.mannwhitneyu(grupo_lula, grupo_nao_lula, alternative='two-sided')

print(f"\nTESTE DE MANN-WHITNEY (não-paramétrico):")
print(f"  Estatística U: {u_stat:.4f}")
print(f"  P-valor: {p_value_mw:.6f}")

# Tamanho do efeito (Cohen's d)
mean_diff = grupo_lula.mean() - grupo_nao_lula.mean()
pooled_std = np.sqrt((grupo_lula.std()**2 + grupo_nao_lula.std()**2) / 2)
cohens_d = mean_diff / pooled_std

print(f"\nTAMANHO DO EFEITO (Cohen's d):")
print(f"  d = {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    print(f"  Interpretação: Efeito PEQUENO")
elif abs(cohens_d) < 0.5:
    print(f"  Interpretação: Efeito MÉDIO")
else:
    print(f"  Interpretação: Efeito GRANDE")

# ==============================================================================
# PARTE 8: SALVAR RESULTADOS
# ==============================================================================

print("\n" + "="*80)
print("SALVANDO RESULTADOS")
print("="*80)

# Salvar dataset completo
df_final.to_csv('analise_municipal_lula_deputadas_2022.csv', 
               index=False, encoding='utf-8-sig')
print("\n[✓] Dataset completo salvo: analise_municipal_lula_deputadas_2022.csv")

# Criar tabela resumo
resumo = pd.DataFrame({
    'Grupo': ['Municípios Lula >50%', 'Municípios Lula <=50%'],
    'N': [len(grupo_lula), len(grupo_nao_lula)],
    'Média': [grupo_lula.mean(), grupo_nao_lula.mean()],
    'Mediana': [grupo_lula.median(), grupo_nao_lula.median()],
    'Desvio_Padrão': [grupo_lula.std(), grupo_nao_lula.std()],
    'Mínimo': [grupo_lula.min(), grupo_nao_lula.min()],
    'Máximo': [grupo_lula.max(), grupo_nao_lula.max()]
})

resumo.to_csv('resumo_estatistico.csv', index=False, encoding='utf-8-sig')
print("[✓] Resumo estatístico salvo: resumo_estatistico.csv")

# Salvar resultados dos testes
testes = pd.DataFrame({
    'Teste': ['Teste t de Student', 'Mann-Whitney U', 'Cohen\'s d'],
    'Estatística': [t_stat, u_stat, cohens_d],
    'P-valor': [p_value, p_value_mw, np.nan]
})

testes.to_csv('resultados_testes.csv', index=False, encoding='utf-8-sig')
print("[✓] Resultados dos testes salvos: resultados_testes.csv")

print("\n" + "="*80)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print("="*80)
