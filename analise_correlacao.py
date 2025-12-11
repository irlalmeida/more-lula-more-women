#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
ANÁLISE COMBINADA: T-TEST + CORRELAÇÃO DE PEARSON

PARTE 1: TESTE T-STUDENT
  Compara MÉDIAS de votos em mulheres entre:
  - Cidades < 50% Lula (Anti-Lula)
  - Cidades ≥ 50% Lula (Pró-Lula)
  
  H0: Não há diferença significativa nas médias
  H1: Há diferença significativa nas médias

PARTE 2: CORRELAÇÃO DE PEARSON
  Mede a relação LINEAR entre % Lula e % Votos em Mulheres
  
  r = -1.0: Correlação NEGATIVA PERFEITA
  r = 0.0:  SEM CORRELAÇÃO
  r = +1.0: Correlação POSITIVA PERFEITA
  
VISUALIZAÇÕES:
  - Box Plots: Distribuições e medianas
  - Histogramas: Densidades dos dados
  - Scatter Plots: Relação entre Lula e Mulheres
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

METRICAS_MULHERES = [
    'perc_votos_mulheres_fed',
    'perc_votos_mulheres_est',
    'perc_votos_mulheres_total'
]

METRICAS_LABELS = {
    'perc_votos_mulheres_fed': 'Deputadas Federais (%)',
    'perc_votos_mulheres_est': 'Deputadas Estaduais/Distritais (%)',
    'perc_votos_mulheres_total': 'Deputadas Total (%)'
}

GRUPOS_LABELS = {
    'menos_50_lula': '< 50% Lula',
    'mais_50_lula': '≥ 50% Lula'
}

# ==============================================================================
# CARREGAMENTO DOS DADOS
# ==============================================================================

print("="*80)
print("ANÁLISE COMBINADA: T-TEST + CORRELAÇÃO DE PEARSON")
print("="*80)

print("\n[1/4] Carregando dados...")

df_menos_50 = pd.read_csv('municipios_menos_50_lula.csv')
df_mais_50 = pd.read_csv('municipios_mais_50_lula.csv')

dados = {
    'menos_50_lula': df_menos_50,
    'mais_50_lula': df_mais_50
}

print(f" Municípios < 50% Lula: {len(df_menos_50)}")
print(f" Municípios ≥ 50% Lula: {len(df_mais_50)}")
print(f" Total: {len(df_menos_50) + len(df_mais_50)}")

# ==============================================================================
# FUNÇÃO 1: TESTE T-STUDENT
# ==============================================================================

def realizar_ttest(grupo1, grupo2, metrica):
    """Realiza T-test"""
    dados1 = grupo1[metrica].dropna().values
    dados2 = grupo2[metrica].dropna().values
    
    if len(dados1) < 2 or len(dados2) < 2:
        return None
    
    media1 = np.mean(dados1)
    media2 = np.mean(dados2)
    mediana1 = np.median(dados1)
    mediana2 = np.median(dados2)
    std1 = np.std(dados1)
    std2 = np.std(dados2)
    diferenca_media = media2 - media1
    
    t_stat, p_value = ttest_ind(dados1, dados2)
    
    return {
        'n_grupo1': len(dados1),
        'n_grupo2': len(dados2),
        'media_grupo1': media1,
        'media_grupo2': media2,
        'mediana_grupo1': mediana1,
        'mediana_grupo2': mediana2,
        'std_grupo1': std1,
        'std_grupo2': std2,
        'diferenca_media': diferenca_media,
        'p_value': p_value,
        't_stat': t_stat,
        'dados_grupo1': dados1,
        'dados_grupo2': dados2
    }

# ==============================================================================
# FUNÇÃO 2: CORRELAÇÃO DE PEARSON
# ==============================================================================

def calcular_pearson(df, metrica_mulheres, metrica_lula='perc_lula'):
    """Calcula correlação de Pearson entre Lula e Mulheres"""
    dados = df[[metrica_lula, metrica_mulheres]].dropna()
    
    if len(dados) < 3:
        return None
    
    X = dados[metrica_lula].values
    y = dados[metrica_mulheres].values
    
    r, p_value = pearsonr(X, y)
    
    return {
        'r': r,
        'p_value': p_value,
        'X': X,
        'y': y,
        'n': len(dados)
    }

# ==============================================================================
# EXECUTAR ANÁLISES
# ==============================================================================

print("\n[2/4] Realizando testes T-Student e correlações de Pearson...")

resultados_ttest = []
resultados_pearson = []

for metrica_mulheres in METRICAS_MULHERES:
    print(f"\n Analisando: {METRICAS_LABELS[metrica_mulheres]}")
    
    # T-TEST
    teste = realizar_ttest(dados['menos_50_lula'], dados['mais_50_lula'], metrica_mulheres)
    
    if teste is not None:
        print(f"  T-TEST:")
        print(f"    < 50% Lula: {teste['media_grupo1']:.2f}% (±{teste['std_grupo1']:.2f})")
        print(f"    ≥ 50% Lula: {teste['media_grupo2']:.2f}% (±{teste['std_grupo2']:.2f})")
        print(f"    Diferença: {teste['diferenca_media']:.2f}%")
        print(f"    p-value: {teste['p_value']:.6f} {'✓' if teste['p_value'] < 0.05 else '✗'}")
        
        resultados_ttest.append({
            'metrica': METRICAS_LABELS[metrica_mulheres],
            'media_menos_50': round(teste['media_grupo1'], 2),
            'media_mais_50': round(teste['media_grupo2'], 2),
            'diferenca_media': round(teste['diferenca_media'], 2),
            't_statistic': round(teste['t_stat'], 4),
            'p_value_ttest': round(teste['p_value'], 6),
            'significancia_ttest': 'SIM' if teste['p_value'] < 0.05 else 'NÃO'
        })
    
    # CORRELAÇÃO DE PEARSON (para cada grupo)
    for grupo_nome, df_grupo in dados.items():
        pearson = calcular_pearson(df_grupo, metrica_mulheres)
        
        if pearson is not None:
            grupo_label = GRUPOS_LABELS[grupo_nome]
            print(f"  PEARSON ({grupo_label}):")
            print(f"    r = {pearson['r']:.4f}")
            print(f"    p-value: {pearson['p_value']:.6f} {'✓' if pearson['p_value'] < 0.05 else '✗'}")
            
            resultados_pearson.append({
                'metrica': METRICAS_LABELS[metrica_mulheres],
                'grupo': grupo_label,
                'r_pearson': round(pearson['r'], 4),
                'p_value_pearson': round(pearson['p_value'], 6),
                'significancia_pearson': 'SIM' if pearson['p_value'] < 0.05 else 'NÃO',
                'n': pearson['n']
            })

# Salvar resultados
df_ttest = pd.DataFrame(resultados_ttest)
df_ttest.to_csv('analise_ttest_resultados.csv', index=False)

df_pearson = pd.DataFrame(resultados_pearson)
df_pearson.to_csv('analise_pearson_resultados.csv', index=False)

print(f"\n Resultados T-test salvos em: analise_ttest_resultados.csv")
print(f" Resultados Pearson salvos em: analise_pearson_resultados.csv")

# ==============================================================================
# VISUALIZAÇÕES
# ==============================================================================

print("\n[3/4] Gerando visualizações...")

# ===== FIGURA 1: BOX PLOTS E HISTOGRAMAS (T-TEST) =====

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('T-Test: Votos em Mulheres - Cidades Pró-Lula vs Anti-Lula', 
             fontsize=16, fontweight='bold')

for idx, metrica_mulheres in enumerate(METRICAS_MULHERES):
    
    teste = realizar_ttest(dados['menos_50_lula'], dados['mais_50_lula'], metrica_mulheres)
    
    if teste is None:
        continue
    
    # BOX PLOT
    ax_box = axes[0, idx]
    
    dados_plot = [teste['dados_grupo1'], teste['dados_grupo2']]
    bp = ax_box.boxplot(dados_plot, labels=['< 50% Lula', '≥ 50% Lula'],
                        patch_artist=True, widths=0.6)
    
    colors = ['#FF9999', '#66B2FF']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for median in bp['medians']:
        median.set_linewidth(2)
        median.set_color('darkred')
    
    ax_box.set_ylabel(METRICAS_LABELS[metrica_mulheres], fontweight='bold', fontsize=11)
    ax_box.set_title(METRICAS_LABELS[metrica_mulheres], fontweight='bold', fontsize=12)
    ax_box.grid(True, alpha=0.3, axis='y')
    
    sig_text = "p < 0.05 ✓" if teste['p_value'] < 0.05 else "p ≥ 0.05 ✗"
    ax_box.text(0.5, 0.98, f"T-Test\n{sig_text}\np={teste['p_value']:.4f}", 
                transform=ax_box.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # HISTOGRAMA
    ax_hist = axes[1, idx]
    
    ax_hist.hist(teste['dados_grupo1'], alpha=0.6, bins=25, label='< 50% Lula', 
                 color='#FF9999', edgecolor='black', linewidth=0.5)
    ax_hist.hist(teste['dados_grupo2'], alpha=0.6, bins=25, label='≥ 50% Lula', 
                 color='#66B2FF', edgecolor='black', linewidth=0.5)
    
    ax_hist.axvline(teste['media_grupo1'], color='red', linestyle='--', linewidth=2)
    ax_hist.axvline(teste['media_grupo2'], color='blue', linestyle='--', linewidth=2)
    
    ax_hist.set_xlabel(METRICAS_LABELS[metrica_mulheres], fontweight='bold')
    ax_hist.set_ylabel('Frequência', fontweight='bold')
    ax_hist.legend(loc='upper right', fontsize=9)
    ax_hist.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('analise_ttest_boxplot_histograma.png', dpi=300, bbox_inches='tight')
print(f" Gráfico T-test salvo em: analise_ttest_boxplot_histograma.png")
plt.close()

# ===== FIGURA 2: SCATTER PLOTS (CORRELAÇÃO DE PEARSON) =====

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Correlação de Pearson: % Lula vs % Votos em Mulheres', 
             fontsize=16, fontweight='bold')

plot_idx = 0

for grupo_idx, (grupo_nome, df_grupo) in enumerate(dados.items()):
    
    for metrica_idx, metrica_mulheres in enumerate(METRICAS_MULHERES):
        
        ax = axes[grupo_idx, metrica_idx]
        
        pearson = calcular_pearson(df_grupo, metrica_mulheres)
        
        if pearson is None:
            continue
        
        X = pearson['X']
        y = pearson['y']
        r = pearson['r']
        p_val = pearson['p_value']
        
        # Scatter plot
        ax.scatter(X, y, alpha=0.5, s=30, color=['#FF9999', '#66B2FF'][grupo_idx])
        
        # Linha de regressão
        z = np.polyfit(X, y, 1)
        p = np.poly1d(z)
        X_sorted = np.sort(X)
        ax.plot(X_sorted, p(X_sorted), "r-", linewidth=2, label='Regressão linear')
        
        ax.set_xlabel('Votos em Lula (%)', fontweight='bold')
        ax.set_ylabel(METRICAS_LABELS[metrica_mulheres], fontweight='bold')
        ax.set_title(f"{GRUPOS_LABELS[grupo_nome]} - {METRICAS_LABELS[metrica_mulheres]}", 
                     fontweight='bold')
        
        sig_text = "p < 0.05 ✓" if p_val < 0.05 else "p ≥ 0.05 ✗"
        
        stats_text = (
            f"r = {r:.4f}\n"
            f"{sig_text}\n"
            f"p = {p_val:.4f}\n"
            f"n = {pearson['n']}"
        )
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                family='monospace', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('analise_pearson_scatter.png', dpi=300, bbox_inches='tight')
print(f" Gráfico Pearson salvo em: analise_pearson_scatter.png")
plt.close()

# ==============================================================================
# RESUMO DOS RESULTADOS
# ==============================================================================

print("\n[4/4] Finalizando análise...")
print("\n" + "="*80)
print("RESUMO: TESTE T-STUDENT")
print("="*80)
print("\n" + df_ttest.to_string(index=False))

print("\n" + "="*80)
print("RESUMO: CORRELAÇÃO DE PEARSON")
print("="*80)
print("\n" + df_pearson.to_string(index=False))

# ==============================================================================
# INTERPRETAÇÃO
# ==============================================================================

print("\n" + "="*80)
print("INTERPRETAÇÃO DOS RESULTADOS")
print("="*80)

print("""
PARTE 1: TESTE T-STUDENT
───────────────────────
O que testa: Se a MÉDIA de votos em mulheres é diferente entre os grupos

Resultado:
  • Se p < 0.05: Há diferença significativa entre as médias ✓
  • Se p ≥ 0.05: Não há diferença significativa entre as médias ✗

Significado:
  ✓ Diferença encontrada: Cidades pró e anti-Lula votam DIFERENTE em mulheres
  ✗ Sem diferença: O percentual de votos em mulheres é IGUAL em ambos os grupos

───────────────────────────────────────────────────────────────────────────────

PARTE 2: CORRELAÇÃO DE PEARSON
───────────────────────────────
O que testa: Se há relação LINEAR entre % Lula e % Votos em Mulheres

Interpretação do r:
  r = +1.0:  Correlação positiva perfeita (aumenta junto)
  r = +0.5:  Correlação positiva moderada
  r = 0.0:   Sem correlação linear
  r = -0.5:  Correlação negativa moderada
  r = -1.0:  Correlação negativa perfeita (aumenta/diminui inverso)

Resultado:
  • Se p < 0.05: Correlação é significativa (não é acaso) ✓
  • Se p ≥ 0.05: Correlação não é significativa (pode ser acaso) ✗

Significado:
  ✓ r positivo: Cidades que votam mais em Lula votam MAIS em mulheres
  ✓ r negativo: Cidades que votam mais em Lula votam MENOS em mulheres
  ✗ r ≈ 0: Voto em Lula e voto em mulheres SÃO INDEPENDENTES

───────────────────────────────────────────────────────────────────────────────

COMO USAR AMBOS OS TESTES JUNTOS:
──────────────────────────────────

T-TEST diz: "As MÉDIAS são diferentes?"
  → Resposta simples: SIM ou NÃO

PEARSON diz: "Há uma relação de TENDÊNCIA?"
  → Mais informação: Como (positivo/negativo) e quanto (força da relação)

EXEMPLO DE INTERPRETAÇÃO COMBINADA:
───────────────────────────────────

Cenário 1:
  T-test: p = 0.001 ✓ (diferença significativa)
  Pearson r = -0.25, p = 0.005 ✓ (correlação negativa)
  
  Conclusão: "À medida que % Lula aumenta, % mulheres DIMINUI significativamente"

Cenário 2:
  T-test: p = 0.03 ✓ (diferença significativa)
  Pearson r = 0.05, p = 0.45 ✗ (sem correlação)
  
  Conclusão: "Há diferença de média, mas sem relação linear clara"

Cenário 3:
  T-test: p = 0.12 ✗ (sem diferença significativa)
  Pearson r = 0.8, p = 0.0001 ✓ (correlação forte)
  
  Conclusão: "Não há diferença de média simples, mas há forte relação linear"
""")

print("\n" + "="*80)
print("ARQUIVOS GERADOS:")
print("="*80)
print("✓ analise_ttest_resultados.csv - Resultados T-test")
print("✓ analise_pearson_resultados.csv - Resultados Pearson")
print("✓ analise_ttest_boxplot_histograma.png - Box plots e histogramas")
print("✓ analise_pearson_scatter.png - Scatter plots com regressão linear")
print("="*80)
