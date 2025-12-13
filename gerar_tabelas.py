import pandas as pd
import matplotlib.pyplot as plt

# Ler os arquivos CSV gerados anteriormente
estatisticas_descritivas = pd.read_csv('estatisticas_descritivas.csv')
analise_ttest_resultados = pd.read_csv('analise_ttest_resultados.csv')
analise_pearson_resultados = pd.read_csv('analise_pearson_resultados.csv')

# TABELA 1: Resumo da Amostra
n_total = estatisticas_descritivas['n_municipios'].sum()
n_menos_50 = estatisticas_descritivas.loc[estatisticas_descritivas['grupo'] == 'Menos_50%_Lula', 'n_municipios'].values[0]
n_mais_50 = estatisticas_descritivas.loc[estatisticas_descritivas['grupo'] == 'Mais_50%_Lula', 'n_municipios'].values[0]

tabela_amostra_data = {
    'Grupo': ['Anti-Lula (< 50% Lula)', 'Pró-Lula (≥ 50% Lula)', 'Total'],
    'Número de Municípios': [n_menos_50, n_mais_50, n_total],
    'Percentual da Amostra': [f'{(n_menos_50/n_total)*100:.1f}%', 
                             f'{(n_mais_50/n_total)*100:.1f}%', 
                             '100.0%']
}

tabela_amostra = pd.DataFrame(tabela_amostra_data)

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('tight')
ax.axis('off')

table_data = [tabela_amostra.columns.tolist()] + tabela_amostra.values.tolist()
table = ax.table(cellText=table_data, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Colorir apenas cabeçalho em cinza
for i in range(len(tabela_amostra.columns)):
    table[(0, i)].set_facecolor('#D3D3D3')
    table[(0, i)].set_text_props(weight='bold')

plt.title('TABELA 1: TAMANHO DA AMOSTRA POR GRUPO', 
          fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('tabela_amostra.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ tabela_amostra.png gerada com sucesso")

# TABELA 2: Teste T - Sem arredondamentos automáticos
# Ler dados brutos e formatar manualmente
tabela_ttest = analise_ttest_resultados.copy()
tabela_ttest_display = pd.DataFrame({
    'Tipo de Eleição': tabela_ttest['metrica'].values,
    'Cidades Anti-Lula (%)': tabela_ttest['media_menos_50'].values,
    'Cidades Pró-Lula (%)': tabela_ttest['media_mais_50'].values,
    'Diferença (pp)': tabela_ttest['diferenca_media'].values,
    'P-valor': [f'< 0.0001' if x < 0.0001 else f'{x:.4f}' for x in tabela_ttest['p_value_ttest'].values],
    'Significativo?': tabela_ttest['significancia_ttest'].values
})

fig, ax = plt.subplots(figsize=(14, 3))
ax.axis('tight')
ax.axis('off')

table_data = [tabela_ttest_display.columns.tolist()] + tabela_ttest_display.values.tolist()
table = ax.table(cellText=table_data, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Colorir apenas cabeçalho em cinza
for i in range(len(tabela_ttest_display.columns)):
    table[(0, i)].set_facecolor('#D3D3D3')
    table[(0, i)].set_text_props(weight='bold')

plt.title('TABELA 2: VOTAÇÃO EM MULHERES POR TIPO DE ELEIÇÃO E GRUPO MUNICIPAL', 
          fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('tabela_ttest.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ tabela_ttest.png gerada com sucesso")

# TABELA 3: Estatísticas Descritivas - Sem arredondamentos automáticos
tabela_descritivas = estatisticas_descritivas.copy()
tabela_descritivas_display = pd.DataFrame({
    'Grupo': tabela_descritivas['grupo'].values,
    'Média Fed (%)': tabela_descritivas['media_perc_mulheres_fed'].values,
    'Mediana Fed (%)': tabela_descritivas['mediana_perc_mulheres_fed'].values,
    'Média Est (%)': tabela_descritivas['media_perc_mulheres_est'].values,
    'Mediana Est (%)': tabela_descritivas['mediana_perc_mulheres_est'].values,
    'Média Total (%)': tabela_descritivas['media_perc_mulheres_total'].values,
    'Mediana Total (%)': tabela_descritivas['mediana_perc_mulheres_total'].values,
    'Desvio Padrão (%)': tabela_descritivas['desvio_padrao_total'].values
})

fig, ax = plt.subplots(figsize=(14, 3))
ax.axis('tight')
ax.axis('off')

table_data = [tabela_descritivas_display.columns.tolist()] + tabela_descritivas_display.values.tolist()
table = ax.table(cellText=table_data, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Colorir apenas cabeçalho em cinza
for i in range(len(tabela_descritivas_display.columns)):
    table[(0, i)].set_facecolor('#D3D3D3')
    table[(0, i)].set_text_props(weight='bold')

plt.title('TABELA 3: ESTATÍSTICAS DESCRITIVAS DE VOTAÇÃO EM MULHERES', 
          fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('tabela_descritivas.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ tabela_descritivas.png gerada com sucesso")

# TABELA 4: Correlação de Pearson - Sem arredondamentos automáticos
tabela_pearson = analise_pearson_resultados.copy()
tabela_pearson_display = pd.DataFrame({
    'Tipo de Eleição': tabela_pearson['metrica'].values,
    'Grupo': tabela_pearson['grupo'].values,
    'Correlação (r)': tabela_pearson['r_pearson'].values,
    'P-valor': [f'< 0.0001' if x < 0.0001 else f'{x:.4f}' for x in tabela_pearson['p_value_pearson'].values],
    'N (Municípios)': tabela_pearson['n'].values,
    'Significativo?': tabela_pearson['significancia_pearson'].values
})

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.axis('tight')
ax.axis('off')

table_data = [tabela_pearson_display.columns.tolist()] + tabela_pearson_display.values.tolist()
table = ax.table(cellText=table_data, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Colorir apenas cabeçalho em cinza
for i in range(len(tabela_pearson_display.columns)):
    table[(0, i)].set_facecolor('#D3D3D3')
    table[(0, i)].set_text_props(weight='bold')

plt.title('TABELA 4: CORRELAÇÃO ENTRE VOTOS EM LULA E VOTOS EM MULHERES', 
          fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('tabela_pearson.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ tabela_pearson.png gerada com sucesso")

print("\n" + "="*60)
print("TODAS AS TABELAS FORAM GERADAS COM SUCESSO!")
print("="*60)
print("\nArquivos criados:")
print("  • tabela_amostra.png (Tabela 1)")
print("  • tabela_ttest.png (Tabela 2)")
print("  • tabela_descritivas.png (Tabela 3)")
print("  • tabela_pearson.png (Tabela 4)")
print("\nUse nos seus textos:")
print("  [INSERIR tabela_amostra.png AQUI]")
print("  [INSERIR tabela_ttest.png AQUI]")
print("  [INSERIR tabela_descritivas.png AQUI]")
print("  [INSERIR tabela_pearson.png AQUI]")
