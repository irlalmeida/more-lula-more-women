#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
import gc
import os
import glob

warnings.filterwarnings('ignore')

print("="*80)
print("ANÁLISE: LULA E VOTAÇÃO EM DEPUTADAS FEDERAIS E ESTADUAIS POR MUNICÍPIO")
print("="*80)

# Caminhos base
DATA_DIR = '/home/otdsp/more-lula-more-women-?/data'
CAND_DIR = os.path.join(DATA_DIR, 'consulta_cand_2022')

# ==============================================================================
# PARTE 1: DADOS PRESIDENCIAIS (2º TURNO)
# ==============================================================================

print("\n[1/5] Processando dados presidenciais...")

arquivo_pres = os.path.join(DATA_DIR, 'votacao_secao_2022_BR.csv')

pres_chunks = pd.read_csv(
    arquivo_pres,
    sep=';',
    encoding='latin1',
    chunksize=500000,
    usecols=['NR_TURNO', 'CD_CARGO', 'SG_UF', 'CD_MUNICIPIO', 'NM_MUNICIPIO', 
             'NR_VOTAVEL', 'QT_VOTOS']
)

pres_mun = []
for chunk in pres_chunks:
    chunk = chunk[(chunk['NR_TURNO'] == 2) & (chunk['CD_CARGO'] == 1)]
    agg = chunk.groupby(['SG_UF', 'CD_MUNICIPIO', 'NM_MUNICIPIO', 'NR_VOTAVEL'], 
                        as_index=False)['QT_VOTOS'].sum()
    pres_mun.append(agg)

pres_mun = pd.concat(pres_mun, ignore_index=True)
gc.collect()

pres_pivot = pres_mun.pivot_table(
    index=['SG_UF', 'CD_MUNICIPIO', 'NM_MUNICIPIO'],
    columns='NR_VOTAVEL',
    values='QT_VOTOS',
    fill_value=0
).reset_index()

pres_pivot.columns.name = None
if 13 in pres_pivot.columns:
    pres_pivot = pres_pivot.rename(columns={13: 'votos_lula', 22: 'votos_bolsonaro'})
else:
    pres_pivot['votos_lula'] = 0
    pres_pivot['votos_bolsonaro'] = 0

pres_pivot['total_validos_pres'] = pres_pivot['votos_lula'] + pres_pivot['votos_bolsonaro']
pres_pivot['perc_lula'] = (pres_pivot['votos_lula'] / pres_pivot['total_validos_pres'] * 100).round(2)
pres_pivot['grupo_lula'] = pres_pivot['perc_lula'].apply(
    lambda x: 'menos_50_lula' if x < 50 else 'mais_50_lula'
)

print(f"   Municípios processados: {len(pres_pivot)}")

# ==============================================================================
# PARTE 2: IDENTIFICAR GÊNERO - DEPUTADOS FEDERAIS E ESTADUAIS
# ==============================================================================

print("\n[2/5] Identificando gênero dos candidatos...")

arquivo_cand = os.path.join(CAND_DIR, 'consulta_cand_2022_BRASIL.csv')

print(f"\n   Lendo arquivo: {arquivo_cand}")

# Verificar estrutura primeiro
amostra = pd.read_csv(arquivo_cand, sep=';', encoding='latin1', nrows=10)
print(f"   Primeiros cargos encontrados: {amostra['CD_CARGO'].unique()}")
print(f"   Descrição dos cargos: {amostra['DS_CARGO'].unique()}")

# Ler todos os candidatos
cands = pd.read_csv(
    arquivo_cand,
    sep=';',
    encoding='latin1',
    usecols=['SG_UF', 'CD_CARGO', 'DS_CARGO', 'NR_TURNO', 'NR_CANDIDATO', 'DS_GENERO']
)

print(f"\n   Total de candidatos no arquivo: {len(cands)}")
print(f"   Cargos únicos: {sorted(cands['CD_CARGO'].unique())}")

# Verificar descrições de cargo
cargo_desc = cands[['CD_CARGO', 'DS_CARGO']].drop_duplicates().sort_values('CD_CARGO')
print("\n   Mapeamento de cargos:")
for _, row in cargo_desc.iterrows():
    print(f"      {row['CD_CARGO']}: {row['DS_CARGO']}")

# Filtrar deputados (federal e estadual), 1º turno
cands_dep = cands[(cands['NR_TURNO'] == 1) & (cands['CD_CARGO'].isin([6, 7]))].copy()

print(f"\n   Candidatos a deputado (federal + estadual) no 1º turno: {len(cands_dep)}")

# Separar por cargo
cands_fed = cands_dep[cands_dep['CD_CARGO'] == 6].copy()
cands_est = cands_dep[cands_dep['CD_CARGO'] == 7].copy()

print(f"\n   DEPUTADO FEDERAL (cargo 6):")
print(f"      Total: {len(cands_fed)}")
if len(cands_fed) > 0:
    cands_fed['eh_mulher'] = (cands_fed['DS_GENERO'].str.strip() == 'FEMININO').astype(int)
    print(f"      Mulheres: {cands_fed['eh_mulher'].sum()}")
    print(f"      Homens: {len(cands_fed) - cands_fed['eh_mulher'].sum()}")
    genero_map_fed = cands_fed.set_index(['SG_UF', 'NR_CANDIDATO'])['eh_mulher'].to_dict()
else:
    print("      AVISO: Nenhum candidato federal encontrado!")
    genero_map_fed = {}

print(f"\n   DEPUTADO ESTADUAL (cargo 7):")
print(f"      Total: {len(cands_est)}")
if len(cands_est) > 0:
    cands_est['eh_mulher'] = (cands_est['DS_GENERO'].str.strip() == 'FEMININO').astype(int)
    print(f"      Mulheres: {cands_est['eh_mulher'].sum()}")
    print(f"      Homens: {len(cands_est) - cands_est['eh_mulher'].sum()}")
    genero_map_est = cands_est.set_index(['SG_UF', 'NR_CANDIDATO'])['eh_mulher'].to_dict()
else:
    print("      AVISO: Nenhum candidato estadual encontrado!")
    genero_map_est = {}

del cands, cands_dep, cands_fed, cands_est
gc.collect()

if len(genero_map_fed) == 0 and len(genero_map_est) == 0:
    print("\n   ERRO: Nenhum candidato foi mapeado!")
    exit(1)

# ==============================================================================
# PARTE 3: PROCESSAR VOTAÇÃO EM DEPUTADOS FEDERAIS
# ==============================================================================

print("\n[3/5] Processando votação em DEPUTADOS FEDERAIS por UF...")

arquivos_uf = sorted(glob.glob(os.path.join(DATA_DIR, 'votacao_secao_2022_*.csv')))
arquivos_uf = [f for f in arquivos_uf if not f.endswith('BR.csv')]

estados_grandes = ['SP', 'MG', 'BA', 'MA', 'RJ', 'RS', 'PR']

votos_dep_fed_list = []

for arq in arquivos_uf:
    uf = os.path.basename(arq).split('_')[-1].replace('.csv', '')
    print(f"   Processando {uf}...", end=' ')

    chunk_size = 300000 if uf in estados_grandes else 500000

    chunks = pd.read_csv(
        arq,
        sep=';',
        encoding='latin1',
        chunksize=chunk_size,
        usecols=['NR_TURNO', 'CD_CARGO', 'SG_UF', 'CD_MUNICIPIO', 
                 'NR_VOTAVEL', 'QT_VOTOS'],
        dtype={'NR_TURNO': 'int8', 'CD_CARGO': 'int8', 'NR_VOTAVEL': 'int32',
               'QT_VOTOS': 'int32', 'CD_MUNICIPIO': 'int32'}
    )

    uf_agg_list = []
    for chunk in chunks:
        chunk = chunk[(chunk['NR_TURNO'] == 1) & (chunk['CD_CARGO'] == 6)]
        agg = chunk.groupby(['SG_UF', 'CD_MUNICIPIO', 'NR_VOTAVEL'], 
                           as_index=False)['QT_VOTOS'].sum()
        uf_agg_list.append(agg)

    if uf_agg_list:
        uf_agg = pd.concat(uf_agg_list, ignore_index=True)
        uf_agg = uf_agg.groupby(['SG_UF', 'CD_MUNICIPIO', 'NR_VOTAVEL'], 
                               as_index=False)['QT_VOTOS'].sum()
        votos_dep_fed_list.append(uf_agg)
        print(f"{len(uf_agg)} registros")
    else:
        print("0 registros")

    gc.collect()

if votos_dep_fed_list:
    votos_dep_fed = pd.concat(votos_dep_fed_list, ignore_index=True)
    del votos_dep_fed_list
    gc.collect()
    print(f"\n   Total de registros: {len(votos_dep_fed)}")

    # Adicionar gênero
    votos_dep_fed['eh_mulher'] = votos_dep_fed.apply(
        lambda r: genero_map_fed.get((r['SG_UF'], r['NR_VOTAVEL']), 0), 
        axis=1
    )
else:
    print("\n   AVISO: Nenhum voto em deputado federal encontrado!")
    votos_dep_fed = pd.DataFrame(columns=['SG_UF', 'CD_MUNICIPIO', 'NR_VOTAVEL', 'QT_VOTOS', 'eh_mulher'])

# ==============================================================================
# PARTE 4: PROCESSAR VOTAÇÃO EM DEPUTADOS ESTADUAIS
# ==============================================================================

print("\n[4/5] Processando votação em DEPUTADOS ESTADUAIS por UF...")

votos_dep_est_list = []

for arq in arquivos_uf:
    uf = os.path.basename(arq).split('_')[-1].replace('.csv', '')
    print(f"   Processando {uf}...", end=' ')

    chunk_size = 300000 if uf in estados_grandes else 500000

    chunks = pd.read_csv(
        arq,
        sep=';',
        encoding='latin1',
        chunksize=chunk_size,
        usecols=['NR_TURNO', 'CD_CARGO', 'SG_UF', 'CD_MUNICIPIO', 
                 'NR_VOTAVEL', 'QT_VOTOS'],
        dtype={'NR_TURNO': 'int8', 'CD_CARGO': 'int8', 'NR_VOTAVEL': 'int32',
               'QT_VOTOS': 'int32', 'CD_MUNICIPIO': 'int32'}
    )

    uf_agg_list = []
    for chunk in chunks:
        chunk = chunk[(chunk['NR_TURNO'] == 1) & (chunk['CD_CARGO'] == 7)]
        agg = chunk.groupby(['SG_UF', 'CD_MUNICIPIO', 'NR_VOTAVEL'], 
                           as_index=False)['QT_VOTOS'].sum()
        uf_agg_list.append(agg)

    if uf_agg_list:
        uf_agg = pd.concat(uf_agg_list, ignore_index=True)
        uf_agg = uf_agg.groupby(['SG_UF', 'CD_MUNICIPIO', 'NR_VOTAVEL'], 
                               as_index=False)['QT_VOTOS'].sum()
        votos_dep_est_list.append(uf_agg)
        print(f"{len(uf_agg)} registros")
    else:
        print("0 registros")

    gc.collect()

if votos_dep_est_list:
    votos_dep_est = pd.concat(votos_dep_est_list, ignore_index=True)
    del votos_dep_est_list
    gc.collect()
    print(f"\n   Total de registros: {len(votos_dep_est)}")

    # Adicionar gênero
    votos_dep_est['eh_mulher'] = votos_dep_est.apply(
        lambda r: genero_map_est.get((r['SG_UF'], r['NR_VOTAVEL']), 0), 
        axis=1
    )
else:
    print("\n   AVISO: Nenhum voto em deputado estadual encontrado!")
    votos_dep_est = pd.DataFrame(columns=['SG_UF', 'CD_MUNICIPIO', 'NR_VOTAVEL', 'QT_VOTOS', 'eh_mulher'])

# ==============================================================================
# PARTE 5: AGREGAR E CRIAR TABELAS FINAIS
# ==============================================================================

print("\n[5/5] Agregando dados e criando tabelas finais...")

# Agregar deputados federais por município
if len(votos_dep_fed) > 0:
    votos_fed_mun = votos_dep_fed.groupby(['SG_UF', 'CD_MUNICIPIO'], as_index=False).agg({
        'QT_VOTOS': 'sum'
    })
    votos_fed_mun = votos_fed_mun.rename(columns={'QT_VOTOS': 'total_votos_dep_fed'})

    votos_fed_mulheres = votos_dep_fed[votos_dep_fed['eh_mulher'] == 1].groupby(
        ['SG_UF', 'CD_MUNICIPIO'], as_index=False
    )['QT_VOTOS'].sum()
    votos_fed_mulheres = votos_fed_mulheres.rename(columns={'QT_VOTOS': 'votos_mulheres_fed'})

    votos_fed_mun = votos_fed_mun.merge(votos_fed_mulheres, on=['SG_UF', 'CD_MUNICIPIO'], how='left')
    votos_fed_mun['votos_mulheres_fed'] = votos_fed_mun['votos_mulheres_fed'].fillna(0)
    votos_fed_mun['perc_votos_mulheres_fed'] = (
        votos_fed_mun['votos_mulheres_fed'] / votos_fed_mun['total_votos_dep_fed'] * 100
    ).round(2)
else:
    votos_fed_mun = pres_pivot[['SG_UF', 'CD_MUNICIPIO']].copy()
    votos_fed_mun['total_votos_dep_fed'] = 0
    votos_fed_mun['votos_mulheres_fed'] = 0
    votos_fed_mun['perc_votos_mulheres_fed'] = 0.0

# Agregar deputados estaduais por município
if len(votos_dep_est) > 0:
    votos_est_mun = votos_dep_est.groupby(['SG_UF', 'CD_MUNICIPIO'], as_index=False).agg({
        'QT_VOTOS': 'sum'
    })
    votos_est_mun = votos_est_mun.rename(columns={'QT_VOTOS': 'total_votos_dep_est'})

    votos_est_mulheres = votos_dep_est[votos_dep_est['eh_mulher'] == 1].groupby(
        ['SG_UF', 'CD_MUNICIPIO'], as_index=False
    )['QT_VOTOS'].sum()
    votos_est_mulheres = votos_est_mulheres.rename(columns={'QT_VOTOS': 'votos_mulheres_est'})

    votos_est_mun = votos_est_mun.merge(votos_est_mulheres, on=['SG_UF', 'CD_MUNICIPIO'], how='left')
    votos_est_mun['votos_mulheres_est'] = votos_est_mun['votos_mulheres_est'].fillna(0)
    votos_est_mun['perc_votos_mulheres_est'] = (
        votos_est_mun['votos_mulheres_est'] / votos_est_mun['total_votos_dep_est'] * 100
    ).round(2)
else:
    votos_est_mun = pres_pivot[['SG_UF', 'CD_MUNICIPIO']].copy()
    votos_est_mun['total_votos_dep_est'] = 0
    votos_est_mun['votos_mulheres_est'] = 0
    votos_est_mun['perc_votos_mulheres_est'] = 0.0

# Merge tudo
df_final = pres_pivot.merge(votos_fed_mun, on=['SG_UF', 'CD_MUNICIPIO'], how='left')
df_final = df_final.merge(votos_est_mun, on=['SG_UF', 'CD_MUNICIPIO'], how='left')

# Preencher NaN com 0
for col in ['total_votos_dep_fed', 'votos_mulheres_fed', 'perc_votos_mulheres_fed',
            'total_votos_dep_est', 'votos_mulheres_est', 'perc_votos_mulheres_est']:
    df_final[col] = df_final[col].fillna(0)

# Calcular totais combinados
df_final['total_votos_deputados'] = df_final['total_votos_dep_fed'] + df_final['total_votos_dep_est']
df_final['votos_mulheres_total'] = df_final['votos_mulheres_fed'] + df_final['votos_mulheres_est']

# Evitar divisão por zero
df_final['perc_votos_mulheres_total'] = 0.0
mask = df_final['total_votos_deputados'] > 0
df_final.loc[mask, 'perc_votos_mulheres_total'] = (
    df_final.loc[mask, 'votos_mulheres_total'] / df_final.loc[mask, 'total_votos_deputados'] * 100
).round(2)

# Selecionar colunas
df_final = df_final[[
    'SG_UF', 'NM_MUNICIPIO', 'total_votos_deputados', 'perc_lula',
    'perc_votos_mulheres_fed', 'perc_votos_mulheres_est', 'perc_votos_mulheres_total',
    'grupo_lula'
]]

df_final = df_final.rename(columns={'total_votos_deputados': 'num_eleitores'})

# Separar em duas tabelas
df_menos_50 = df_final[df_final['grupo_lula'] == 'menos_50_lula'].copy()
df_mais_50 = df_final[df_final['grupo_lula'] == 'mais_50_lula'].copy()

# Ordenar por percentual total de votos em mulheres
df_menos_50 = df_menos_50.sort_values('perc_votos_mulheres_total', ascending=False)
df_mais_50 = df_mais_50.sort_values('perc_votos_mulheres_total', ascending=False)

# Remover coluna grupo_lula
df_menos_50 = df_menos_50.drop(columns=['grupo_lula'])
df_mais_50 = df_mais_50.drop(columns=['grupo_lula'])

# Salvar
df_menos_50.to_csv('municipios_menos_50_lula.csv', index=False)
df_mais_50.to_csv('municipios_mais_50_lula.csv', index=False)

print(f"\n   Tabela 1 (< 50% Lula): {len(df_menos_50)} municípios")
print(f"   Tabela 2 (≥ 50% Lula): {len(df_mais_50)} municípios")

# ==============================================================================
# ESTATÍSTICAS DESCRITIVAS
# ==============================================================================

print("\n" + "="*80)
print("ESTATÍSTICAS DESCRITIVAS")
print("="*80)

def calc_stats(df, nome):
    stats_dict = {
        'grupo': nome,
        'n_municipios': len(df),
        'media_perc_mulheres_fed': df['perc_votos_mulheres_fed'].mean().round(2),
        'mediana_perc_mulheres_fed': df['perc_votos_mulheres_fed'].median().round(2),
        'media_perc_mulheres_est': df['perc_votos_mulheres_est'].mean().round(2),
        'mediana_perc_mulheres_est': df['perc_votos_mulheres_est'].median().round(2),
        'media_perc_mulheres_total': df['perc_votos_mulheres_total'].mean().round(2),
        'mediana_perc_mulheres_total': df['perc_votos_mulheres_total'].median().round(2),
        'moda_perc_mulheres_total': df['perc_votos_mulheres_total'].mode().values[0] if len(df['perc_votos_mulheres_total'].mode()) > 0 else np.nan,
        'desvio_padrao_total': df['perc_votos_mulheres_total'].std().round(2)
    }
    return stats_dict

stats_menos = calc_stats(df_menos_50, 'Menos_50%_Lula')
stats_mais = calc_stats(df_mais_50, 'Mais_50%_Lula')

df_stats = pd.DataFrame([stats_menos, stats_mais])
df_stats.to_csv('estatisticas_descritivas.csv', index=False)

print("\nEstatísticas salvas em 'estatisticas_descritivas.csv'")
print("\n" + df_stats.to_string(index=False))

print("\n" + "="*80)
print("ANÁLISE CONCLUÍDA!")
print("="*80)
print("\nArquivos gerados:")
print("  1. municipios_menos_50_lula.csv")
print("  2. municipios_mais_50_lula.csv")
print("  3. estatisticas_descritivas.csv")
print("\nColunas nas tabelas de municípios:")
print("  - SG_UF: UF do município")
print("  - NM_MUNICIPIO: Nome do município")
print("  - num_eleitores: Total de votos válidos em deputados (federal + estadual)")
print("  - perc_lula: % de votos em Lula no 2º turno presidencial")
print("  - perc_votos_mulheres_fed: % de votos em deputadas FEDERAIS")
print("  - perc_votos_mulheres_est: % de votos em deputadas ESTADUAIS")
print("  - perc_votos_mulheres_total: % de votos em deputadas (fed + est)")

del votos_dep_fed, votos_dep_est, votos_fed_mun, votos_est_mun
gc.collect()
