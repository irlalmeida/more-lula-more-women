#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import stats
import warnings
import gc  # Garbage collector manual

warnings.filterwarnings('ignore')

print("="*80)
print("ANÁLISE: MUNICÍPIOS LULA E VOTAÇÃO EM DEPUTADAS FEDERAIS")
print("="*80)

# ------------------------------------------------------------
# Helper para compatibilizar nomes de colunas
# ------------------------------------------------------------
def pick(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"Nenhuma coluna encontrada entre: {cands}")

def ensure_str(df, col):
    if col in df.columns:
        df[col] = df[col].astype(str)
    return df

def std_rename(df, mapping):
    for *alts, std in mapping:
        for a in alts:
            if a in df.columns:
                if std not in df.columns:
                    df = df.rename(columns={a: std})
                break
    return df

# ==============================================================================
# PARTE 1: PRESIDENCIAL 2º TURNO (BR)
# ==============================================================================
print("\n[1] Carregando dados presidenciais (2º turno)...")

# Usar chunksize para arquivo grande
chunksize = 500000
chunks_pres = []

for chunk in pd.read_csv(
    './data/votacao_secao_2022_BR.csv',
    encoding='latin1',
    sep=';',
    low_memory=False,
    chunksize=chunksize
):
    chunk = std_rename(chunk, [
        ('NR_TURNO','NRTURNO','NR_TURNO'),
        ('CD_CARGO_PERGUNTA','CDCARGOPERGUNTA','CD_CARGO_PERGUNTA'),
        ('CD_CARGO','CD_CARGO'),
        ('SG_UF','SGUF','SG_UF'),
        ('CD_MUNICIPIO','CDMUNICIPIO','CD_MUNICIPIO'),
        ('NM_MUNICIPIO','NMMUNICIPIO','NM_MUNICIPIO'),
        ('NR_VOTAVEL','NRVOTAVEL','NR_VOTAVEL'),
        ('QT_VOTOS','QTVOTOS','QT_VOTOS'),
    ])
    
    TURNO_P = pick(chunk, 'NR_TURNO')
    CARGO_P = 'CD_CARGO_PERGUNTA' if 'CD_CARGO_PERGUNTA' in chunk.columns else pick(chunk, 'CD_CARGO')
    
    # Filtrar já no chunk
    chunk_filtrado = chunk[(chunk[TURNO_P] == 2) & (chunk[CARGO_P] == 1)].copy()
    
    if len(chunk_filtrado) > 0:
        chunks_pres.append(chunk_filtrado)
    
    del chunk, chunk_filtrado
    gc.collect()

df_pres = pd.concat(chunks_pres, ignore_index=True)
del chunks_pres
gc.collect()

UF_P = pick(df_pres, 'SG_UF')
CDM_P = pick(df_pres, 'CD_MUNICIPIO')
NMM_P = pick(df_pres, 'NM_MUNICIPIO')
NRV_P = pick(df_pres, 'NR_VOTAVEL')
QTV_P = pick(df_pres, 'QT_VOTOS')

df_pres = ensure_str(df_pres, NRV_P)
df_pres_2t = df_pres.copy()
del df_pres
gc.collect()

print(f"  Total de registros (2º turno Presidente): {len(df_pres_2t):,}")

# Agregar por município e candidato
votos_mun = df_pres_2t.groupby([UF_P, CDM_P, NMM_P, NRV_P])[QTV_P].sum().reset_index()
del df_pres_2t
gc.collect()

# Pivotar
votos_pivot = votos_mun.pivot_table(
    index=[UF_P, CDM_P, NMM_P],
    columns=NRV_P,
    values=QTV_P,
    fill_value=0
).reset_index()
votos_pivot.columns.name = None
del votos_mun
gc.collect()

# Renomear colunas para Lula e Bolsonaro
for lula_key in ('13', 13):
    if lula_key in votos_pivot.columns:
        votos_pivot = votos_pivot.rename(columns={lula_key: 'votos_lula'})
for bolso_key in ('22', 22):
    if bolso_key in votos_pivot.columns:
        votos_pivot = votos_pivot.rename(columns={bolso_key: 'votos_bolsonaro'})

if 'votos_lula' not in votos_pivot.columns:
    votos_pivot['votos_lula'] = 0
if 'votos_bolsonaro' not in votos_pivot.columns:
    votos_pivot['votos_bolsonaro'] = 0

votos_pivot['votos_validos'] = votos_pivot['votos_lula'] + votos_pivot['votos_bolsonaro']
votos_pivot['perc_lula'] = np.where(
    votos_pivot['votos_validos'] > 0,
    votos_pivot['votos_lula'] / votos_pivot['votos_validos'] * 100,
    0.0
)

votos_pivot['grupo_lula'] = (votos_pivot['perc_lula'] > 50).astype(int)

votos_pivot = votos_pivot.rename(columns={
    UF_P: 'SG_UF', CDM_P: 'CD_MUNICIPIO', NMM_P: 'NM_MUNICIPIO'
})

print(f"  Municípios processados: {len(votos_pivot):,}")
print(f"  Municípios Lula >50%: {votos_pivot['grupo_lula'].sum():,}")
print(f"  Municípios Lula <=50%: {len(votos_pivot) - votos_pivot['grupo_lula'].sum():,}")

# ==============================================================================
# PARTE 2: CANDIDATOS (GÊNERO)
# ==============================================================================
print("\n[2] Carregando dados de candidatos...")

df_cand = pd.read_csv(
    '/home/otdsp/more-lula-more-women-?/data/consulta_cand_2022/consulta_cand_2022_BRASIL.csv',
    encoding='latin1',
    sep=';',
    low_memory=False
)

df_cand = std_rename(df_cand, [
    ('CD_CARGO','CD_CARGO'),
    ('NR_TURNO','NRTURNO','NR_TURNO'),
    ('DS_GENERO','DS_GENERO'),
    ('SG_UF','SGUF','SG_UF'),
    ('NR_CANDIDATO','NRCANDIDATO','NR_CANDIDATO'),
])

CARGO_C = pick(df_cand, 'CD_CARGO')
UF_C = pick(df_cand, 'SG_UF')
NR_CAND = pick(df_cand, 'NR_CANDIDATO')
GEN_C = pick(df_cand, 'DS_GENERO')
TURNO_C = 'NR_TURNO' if 'NR_TURNO' in df_cand.columns else None

mask = (df_cand[CARGO_C] == 6)
if TURNO_C:
    mask &= (df_cand[TURNO_C] == 1)

df_dep_fed = df_cand[mask].copy()
del df_cand
gc.collect()

df_dep_fed = ensure_str(df_dep_fed, NR_CAND)
df_dep_fed['eh_mulher'] = (df_dep_fed[GEN_C] == 'FEMININO').astype(int)

map_genero = df_dep_fed[[UF_C, NR_CAND, 'eh_mulher']].copy()
map_genero = map_genero.rename(columns={UF_C: 'SG_UF', NR_CAND: 'NR_CANDIDATO'})
del df_dep_fed
gc.collect()

print(f"  Total de candidatos a Dep. Federal: {len(map_genero):,}")
print(f"  Candidatas mulheres: {map_genero['eh_mulher'].sum():,}")
print(f"  Candidatos homens: {len(map_genero) - map_genero['eh_mulher'].sum():,}")

# ==============================================================================
# PARTE 3: VOTAÇÃO DEP. FEDERAL (1º TURNO) POR UF - OTIMIZADO
# ==============================================================================
print("\n[3] Carregando votação em Deputados Federais por UF...")

ufs = ['AC','AL','AM','AP','BA','CE','DF','ES','GO','MA',
       'MG','MS','MT','PA','PB','PE','PI','PR','RJ','RN',
       'RO','RR','RS','SC','SE','SP','TO']

# MUDANÇA CRÍTICA: Processar e salvar intermediários, não acumular tudo
arquivo_temp = './temp_votos_dep_agregados.csv'
primeiro_uf = True

for uf in ufs:
    try:
        print(f"  Processando {uf}...", end=' ', flush=True)
        
        # OTIMIZAÇÃO 1: Usar chunksize para estados grandes
        chunksize_uf = 300000 if uf in ['SP','MG','BA','MA','RJ','RS','PR'] else 500000
        
        chunks_uf = []
        for chunk in pd.read_csv(
            f'./data/votacao_secao_2022_{uf}.csv',
            encoding='latin1',
            sep=';',
            low_memory=False,
            chunksize=chunksize_uf,
            # OTIMIZAÇÃO 2: Especificar dtypes para reduzir memória
            dtype={
                'QT_VOTOS': 'int32',  # Em vez de int64
                'CD_MUNICIPIO': 'int32',
                'NR_TURNO': 'int8',
                'CD_CARGO_PERGUNTA': 'int8'
            }
        ):
            chunk = std_rename(chunk, [
                ('NR_TURNO','NRTURNO','NR_TURNO'),
                ('CD_CARGO_PERGUNTA','CDCARGOPERGUNTA','CD_CARGO_PERGUNTA'),
                ('CD_CARGO','CD_CARGO'),
                ('SG_UF','SGUF','SG_UF'),
                ('CD_MUNICIPIO','CDMUNICIPIO','CD_MUNICIPIO'),
                ('NM_MUNICIPIO','NMMUNICIPIO','NM_MUNICIPIO'),
                ('NR_VOTAVEL','NRVOTAVEL','NR_VOTAVEL'),
                ('QT_VOTOS','QTVOTOS','QT_VOTOS'),
            ])
            
            TURNO_D = pick(chunk, 'NR_TURNO')
            CARGO_D = 'CD_CARGO_PERGUNTA' if 'CD_CARGO_PERGUNTA' in chunk.columns else pick(chunk, 'CD_CARGO')
            
            # Filtrar já no chunk
            chunk_dep = chunk[(chunk[TURNO_D] == 1) & (chunk[CARGO_D] == 6)].copy()
            
            if len(chunk_dep) > 0:
                chunks_uf.append(chunk_dep)
            
            del chunk, chunk_dep
            gc.collect()
        
        if len(chunks_uf) == 0:
            print("SEM DADOS")
            continue
        
        df_uf = pd.concat(chunks_uf, ignore_index=True)
        del chunks_uf
        gc.collect()
        
        UF_D = pick(df_uf, 'SG_UF')
        CDM_D = pick(df_uf, 'CD_MUNICIPIO')
        NMM_D = pick(df_uf, 'NM_MUNICIPIO')
        NRV_D = pick(df_uf, 'NR_VOTAVEL')
        QTV_D = pick(df_uf, 'QT_VOTOS')
        
        df_uf = ensure_str(df_uf, NRV_D)
        
        # OTIMIZAÇÃO 3: Agregar imediatamente, não guardar dados brutos
        votos_dep_mun = df_uf.groupby([UF_D, CDM_D, NMM_D, NRV_D])[QTV_D].sum().reset_index()
        del df_uf
        gc.collect()
        
        votos_dep_mun = votos_dep_mun.rename(columns={
            UF_D: 'SG_UF', CDM_D: 'CD_MUNICIPIO', NMM_D: 'NM_MUNICIPIO',
            NRV_D: 'NR_VOTAVEL', QTV_D: 'QT_VOTOS'
        })
        
        # MUDANÇA CRÍTICA: Salvar incrementalmente em vez de acumular na memória
        if primeiro_uf:
            votos_dep_mun.to_csv(arquivo_temp, index=False, mode='w', header=True)
            primeiro_uf = False
        else:
            votos_dep_mun.to_csv(arquivo_temp, index=False, mode='a', header=False)
        
        print(f"OK ({len(votos_dep_mun):,} linhas agregadas)")
        
        del votos_dep_mun
        gc.collect()
        
    except FileNotFoundError:
        print("ARQUIVO NÃO ENCONTRADO")
        continue
    except Exception as e:
        print(f"ERRO: {e}")
        continue

# Carregar dados consolidados do arquivo temporário
print("\n  Carregando dados consolidados...")
df_dep_todos = pd.read_csv(arquivo_temp, low_memory=False)
print(f"  Total de linhas: {len(df_dep_todos):,}")

# ==============================================================================
# PARTE 4: AGREGAR VOTOS EM MULHERES POR MUNICÍPIO
# ==============================================================================
print("\n[4] Agregando votação em mulheres por município...")

# CORREÇÃO: Garantir que NR_VOTAVEL seja string antes do merge
df_dep_todos['NR_VOTAVEL'] = df_dep_todos['NR_VOTAVEL'].astype(str)

# CORREÇÃO: Garantir que NR_CANDIDATO também seja string
map_genero['NR_CANDIDATO'] = map_genero['NR_CANDIDATO'].astype(str)

df_dep_todos = df_dep_todos.merge(
    map_genero.rename(columns={'NR_CANDIDATO':'NR_VOTAVEL'}),
    on=['SG_UF','NR_VOTAVEL'],
    how='left'
)

df_dep_todos['eh_mulher'] = df_dep_todos['eh_mulher'].fillna(0)
df_dep_todos['votos_mulher'] = df_dep_todos['eh_mulher'] * df_dep_todos['QT_VOTOS']

votos_genero = df_dep_todos.groupby(['SG_UF','CD_MUNICIPIO','NM_MUNICIPIO']).agg({
    'QT_VOTOS':'sum',
    'votos_mulher':'sum'
}).reset_index()

del df_dep_todos
gc.collect()

votos_genero = votos_genero.rename(columns={
    'QT_VOTOS':'total_votos_dep',
    'votos_mulher':'votos_em_mulheres'
})

votos_genero['perc_votos_mulheres'] = np.where(
    votos_genero['total_votos_dep'] > 0,
    votos_genero['votos_em_mulheres'] / votos_genero['total_votos_dep'] * 100,
    0.0
)

print(f"  Municípios com dados de deputadas: {len(votos_genero):,}")


# ==============================================================================
# PARTE 5: JUNTAR E ANALISAR
# ==============================================================================
print("\n[5] Juntando dados e preparando análise...")

df_final = votos_pivot.merge(
    votos_genero,
    on=['SG_UF','CD_MUNICIPIO','NM_MUNICIPIO'],
    how='inner'
)

del votos_pivot, votos_genero
gc.collect()

print(f"  Dataset final: {len(df_final):,} municípios")

grupo_lula = df_final[df_final['grupo_lula'] == 1]['perc_votos_mulheres']
grupo_nao = df_final[df_final['grupo_lula'] == 0]['perc_votos_mulheres']

print("\n" + "="*80)
print("RESULTADOS DA ANÁLISE")
print("="*80)

print("\nTAMANHO DOS GRUPOS:")
print(f"  Municípios Lula >50%: {len(grupo_lula):,}")
print(f"  Municípios Lula <=50%: {len(grupo_nao):,}")

print("\nMÉDIA DE VOTOS EM DEPUTADAS FEDERAIS:")
print(f"  Grupo Lula >50%: {grupo_lula.mean():.2f}%")
print(f"  Grupo Lula <=50%: {grupo_nao.mean():.2f}%")
print(f"  Diferença: {grupo_lula.mean() - grupo_nao.mean():.2f} pp")

print("\nMEDIANA DE VOTOS EM DEPUTADAS FEDERAIS:")
print(f"  Grupo Lula >50%: {grupo_lula.median():.2f}%")
print(f"  Grupo Lula <=50%: {grupo_nao.median():.2f}%")

print("\nDESVIO PADRÃO:")
print(f"  Grupo Lula >50%: {grupo_lula.std():.2f}%")
print(f"  Grupo Lula <=50%: {grupo_nao.std():.2f}%")

print("\n" + "-"*80)
print("TESTES ESTATÍSTICOS")
print("-"*80)

t_stat, p_value = stats.ttest_ind(grupo_lula, grupo_nao)
print("\nTESTE T DE STUDENT (independente):")
print(f"  Estatística t: {t_stat:.4f}")
print(f"  P-valor: {p_value:.6f}")

from scipy.stats import mannwhitneyu
u_stat, p_value_mw = mannwhitneyu(grupo_lula, grupo_nao, alternative='two-sided')
print("\nTESTE DE MANN-WHITNEY (não-paramétrico):")
print(f"  Estatística U: {u_stat:.4f}")
print(f"  P-valor: {p_value_mw:.6f}")

mean_diff = grupo_lula.mean() - grupo_nao.mean()
pooled_std = np.sqrt((grupo_lula.std()**2 + grupo_nao.std()**2) / 2)
cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

print("\nTAMANHO DO EFEITO (Cohen's d):")
print(f"  d = {cohens_d:.4f}")

print("\n" + "="*80)
print("SALVANDO RESULTADOS")
print("="*80)

df_final.to_csv('analise_municipal_lula_deputadas_2022.csv', index=False, encoding='utf-8-sig')

resumo = pd.DataFrame({
    'Grupo': ['Municípios Lula >50%','Municípios Lula <=50%'],
    'N': [len(grupo_lula), len(grupo_nao)],
    'Média': [grupo_lula.mean(), grupo_nao.mean()],
    'Mediana': [grupo_lula.median(), grupo_nao.median()],
    'Desvio_Padrão': [grupo_lula.std(), grupo_nao.std()],
    'Mínimo': [grupo_lula.min(), grupo_nao.min()],
    'Máximo': [grupo_lula.max(), grupo_nao.max()]
})
resumo.to_csv('resumo_estatistico.csv', index=False, encoding='utf-8-sig')

testes = pd.DataFrame({
    'Teste': ['Teste t de Student','Mann-Whitney U','Cohen\'s d'],
    'Estatística': [t_stat, u_stat, cohens_d],
    'P-valor': [p_value, p_value_mw, np.nan]
})
testes.to_csv('resultados_testes.csv', index=False, encoding='utf-8-sig')

print("\n[✓] Arquivos salvos: analise_municipal_lula_deputadas_2022.csv, resumo_estatistico.csv, resultados_testes.csv")
print("="*80)

# Limpar arquivo temporário
import os
if os.path.exists(arquivo_temp):
    os.remove(arquivo_temp)
    print(f"\n[✓] Arquivo temporário {arquivo_temp} removido")
