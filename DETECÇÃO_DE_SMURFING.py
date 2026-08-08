# ==============================================================================
# DETECÇÃO DE SMURFING EM LAVAGEM DE DINHEIRO VIA IA E TEORIA DOS GRAFOS
# ==============================================================================

# 1. INSTALAÇÃO E IMPORTAÇÃO DE BIBLIOTECAS
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Configuração de semente aleatória para reprodutibilidade
np.random.seed(42)

# ==============================================================================
# 2. SIMULAÇÃO DE DADOS BANCÁRIOS
# ==============================================================================
L = 10000.0  # Teto regulatório de notificação (R$ 10.000,00)
num_contas = 50
master_account = 10 # Conta concentradora (Smurf Master)
smurfs = list(range(0, 10)) # Contas de laranjas/smurfs (0 a 9)

transactions = []

# (A) Transações Orgânicas / Normais entre contas comuns (11 a 49)
for i in range(11, num_contas):
    destinos = np.random.choice([k for k in range(11, num_contas) if k != i], size=np.random.randint(1, 4))
    for t in destinos:
        valor = np.random.uniform(50, 15000)
        transactions.append({'origem': i, 'destino': t, 'valor': valor, 'is_smurf': 0})

# (B) Padrão de Smurfing: Laranjas enviando múltiplos depósitos fracionados < L para o Master
for smurf in smurfs:
    for _ in range(5): # 5 microtransações por laranja
        valor = np.random.uniform(0.91 * L, 0.99 * L) # Ex: entre R$ 9.100 e R$ 9.900
        transactions.append({'origem': smurf, 'destino': master_account, 'valor': valor, 'is_smurf': 1})

df_tx = pd.DataFrame(transactions)

# ==============================================================================
# 3. EXTRAÇÃO DE ATRIBUTOS MATEMÁTICOS (GRAPH & STATS)
# ==============================================================================
# Construção do Grafo Dirigido G = (V, E)
G = nx.DiGraph()
for _, row in df_tx.iterrows():
    G.add_edge(int(row['origem']), int(row['destino']), weight=row['valor'])

# Cálculo da Centralidade de Intermediação (Betweenness Centrality)
betweenness = nx.betweenness_centrality(G)

features = []
for node in range(num_contas):
    tx_entradas = df_tx[df_tx['destino'] == node]
    
    # 1. Grau de Entrada Ponderado D_in
    d_in = tx_entradas['valor'].sum()
    
    # 2. Contagem de transações na faixa crítica [0.9L, L)
    tx_limiar = tx_entradas[(tx_entradas['valor'] >= 0.9 * L) & (tx_entradas['valor'] < L)]
    count_limiar = len(tx_limiar)
    
    # 3. Entropia de Shannon H(X) da distribuição de valores recebidos
    if len(tx_entradas) > 1:
        counts, _ = np.histogram(tx_entradas['valor'], bins=5)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        h_entropy = -np.sum(probs * np.log2(probs))
    else:
        h_entropy = 0.0
        
    cb = betweenness.get(node, 0.0)
    is_suspicious = 1 if (node == master_account or node in smurfs) else 0
    
    features.append({
        'conta': node,
        'd_in': d_in,
        'count_limiar': count_limiar,
        'entropia': h_entropy,
        'betweenness': cb,
        'is_suspicious': is_suspicious
    })

df_feat = pd.DataFrame(features)

# Cálculo do Z-Score para disparos atípicos proximo ao limiar
mean_cnt = df_feat['count_limiar'].mean()
std_cnt = df_feat['count_limiar'].std() + 1e-6
df_feat['z_score_limiar'] = (df_feat['count_limiar'] - mean_cnt) / std_cnt

# ==============================================================================
# 4. TREINAMENTO DO MODELO PREDITIVO (RISK SCORE)
# ==============================================================================
X_cols = ['d_in', 'z_score_limiar', 'entropia', 'betweenness']
X = df_feat[X_cols]
y = df_feat['is_suspicious']

# Normalização das variáveis para convergência logística
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Regressão Logística
model = LogisticRegression()
model.fit(X_scaled, y)

# Cálculo da Probabilidade de Risco P(Y=1 | X)
df_feat['score_risco'] = model.predict_proba(X_scaled)[:, 1]

# ==============================================================================
# 5. DEMONSTRAÇÃO DOS RESULTADOS E VISUALIZAÇÃO
# ==============================================================================
print("=== TOP 10 CONTAS COM MAIOR SCORE DE RISCO ===")
top_suspeitas = df_feat.sort_values(by='score_risco', ascending=False).head(10)
print(top_suspeitas[['conta', 'd_in', 'count_limiar', 'score_risco', 'is_suspicious']])

# Plot do Grafo Financeiro
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42)

# Cores por nível de risco predito
node_colors = [df_feat.loc[df_feat['conta'] == n, 'score_risco'].values[0] for n in G.nodes()]

nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.Reds, node_size=500)
nx.draw_networkx_edges(G, pos, alpha=0.3, arrowstyle='->', arrowsize=10)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

plt.colorbar(nodes, label='Score de Risco (IA)')
plt.title('Rede de Transações Bancárias - Detecção de Smurfing', fontsize=14)
plt.axis('off')
plt.show()
