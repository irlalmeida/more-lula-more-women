import pandas as pd
import matplotlib.pyplot as plt

# Ler os arquivos CSV gerados anteriormente
estatisticas_descritivas = pd.read_csv('estatisticas_descritivas.csv')
analise_ttest_resultados = pd.read_csv('analise_ttest_resultados.csv')
analise_pearson_resultados = pd.read_csv('analise_pearson_resultados.csv')

# TABELA 1: Teste T - Reformatar para visualização
tabela_ttest = analise_ttest_resultados[['metrica', 'media_menos_50', 'media_mais_50', 
                                          'diferenca_media', 'p_value_ttest', 'significancia_ttest']]
tabela_ttest.columns = ['Tipo de Eleição', 'Cidades Anti-Lula (%)', 'Cidades Pró-Lula (%)', 
                        'Diferença (pp)', 'P-valor', 'Significativo?']
tabela_ttest = tabela_ttest.round(4)

fig, ax = plt.subplots(figsize=(14, 3))
ax.axis('tight')
ax.axis('off')

table_data = [tabela_ttest.columns.tolist()] + tabela_ttest.values.tolist()
table = ax.table(cellText=table_data, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Colorir apenas cabeçalho em cinza
for i in range(len(tabela_ttest.columns)):
    table[(0, i)].set_facecolor('#D3D3D3')
    table[(0, i)].set_text_props(weight='bold')

plt.title('TABELA 1: VOTAÇÃO EM MULHERES POR TIPO DE ELEIÇÃO E GRUPO MUNICIPAL', 
          fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('tabela_ttest.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ tabela_ttest.png gerada com sucesso")

# TABELA 2: Estatísticas Descritivas
tabela_descritivas = estatisticas_descritivas[['grupo', 'media_perc_mulheres_fed', 
                                                'mediana_perc_mulheres_fed', 'media_perc_mulheres_est',
                                                'mediana_perc_mulheres_est', 'media_perc_mulheres_total',
                                                'mediana_perc_mulheres_total', 'desvio_padrao_total']]
tabela_descritivas.columns = ['Grupo', 'Média Fed (%)', 'Mediana Fed (%)', 
                              'Média Est (%)', 'Mediana Est (%)', 
                              'Média Total (%)', 'Mediana Total (%)', 'Desvio Padrão (%)']
tabela_descritivas = tabela_descritivas.round(2)

fig, ax = plt.subplots(figsize=(14, 3))
ax.axis('tight')
ax.axis('off')

table_data = [tabela_descritivas.columns.tolist()] + tabela_descritivas.values.tolist()
table = ax.table(cellText=table_data, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Colorir apenas cabeçalho em cinza
for i in range(len(tabela_descritivas.columns)):
    table[(0, i)].set_facecolor('#D3D3D3')
    table[(0, i)].set_text_props(weight='bold')

plt.title('TABELA 2: ESTATÍSTICAS DESCRITIVAS DE VOTAÇÃO EM MULHERES', 
          fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('tabela_descritivas.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ tabela_descritivas.png gerada com sucesso")

# TABELA 3: Correlação de Pearson
tabela_pearson = analise_pearson_resultados[['metrica', 'grupo', 'r_pearson', 
                                              'p_value_pearson', 'n', 'significancia_pearson']]
tabela_pearson.columns = ['Tipo de Eleição', 'Grupo', 'Correlação (r)', 
                          'P-valor', 'N (Municípios)', 'Significativo?']
tabela_pearson = tabela_pearson.round(4)

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.axis('tight')
ax.axis('off')

table_data = [tabela_pearson.columns.tolist()] + tabela_pearson.values.tolist()
table = ax.table(cellText=table_data, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Colorir apenas cabeçalho em cinza
for i in range(len(tabela_pearson.columns)):
    table[(0, i)].set_facecolor('#D3D3D3')
    table[(0, i)].set_text_props(weight='bold')

plt.title('TABELA 3: CORRELAÇÃO ENTRE VOTOS EM LULA E VOTOS EM MULHERES', 
          fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('tabela_pearson.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ tabela_pearson.png gerada com sucesso")

# TABELA 4: Resumo da Amostra
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

plt.title('TABELA 4: TAMANHO DA AMOSTRA POR GRUPO', 
          fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('tabela_amostra.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ tabela_amostra.png gerada com sucesso")

print("\n" + "="*60)
print("TODAS AS TABELAS FORAM GERADAS COM SUCESSO!")
print("="*60)
print("\nArquivos criados:")
print("  • tabela_ttest.png")
print("  • tabela_descritivas.png")
print("  • tabela_pearson.png")
print("  • tabela_amostra.png")
