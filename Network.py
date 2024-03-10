import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Se carga la infomacion del archivo excel 
df = pd.read_excel('Table 1.xlsx',sheet_name='matrix_passes_between_players',index_col=0)

# Crea un grafo dirigido desde el DataFrame
G = nx.from_pandas_adjacency(df)

# Realizar operaciones en el grafo, por ejemplo, contar los nodos y los bordes
num_nodos = G.number_of_nodes()
num_bordes = G.number_of_edges()

print("Número de nodos:", num_nodos)
print("Número de bordes:", num_bordes)

# Se separa los nodos por equipo
e1 = list(G.nodes())[1:14]
e2 = list(G.nodes())[15:G.number_of_nodes()]

se1 = G.subgraph(e1)
se2 = G.subgraph(e2)

# Imprimir información básica sobre el grafo

plt.figure()
nx.draw(se1, with_labels=True, node_color='skyblue', node_size=500, font_size=10)
plt.figure()
nx.draw(se2, with_labels=True, node_color='skyblue', node_size=500, font_size=10)

# Closeness
cse1 = nx.closeness_centrality(se1)
bcse1 = nx.betweenness_centrality(se1)
for i, closeness_centrality in cse1.items():
    print(f"Nodo {i}: Closeness Centrality {closeness_centrality}")
cse2 = nx.closeness_centrality(se2)
bcss2 = nx.betweenness_centrality(se1)
for i, closeness_centrality in cse2.items():
    print(f"Nodo {i}: Closeness Centrality {closeness_centrality}")


plt.show()