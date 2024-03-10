import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Se carga la infomacion del archivo excel
# dr = pd.read_excel('Table 1.xlsx',sheet_name='player_code_name',index_col=0) 
df = pd.read_excel('Table 1.xlsx',sheet_name='matrix_passes_between_players',index_col=0)

# Crea un grafo dirigido desde el DataFrame
G = nx.from_pandas_adjacency(df)
# R = nx.from_pandas_edgelist(dr, source = 'NODE NUMBER', target = 'PLAYER NAME', )

# Se separa los nodos por equipo
e1 = list(G.nodes())[1:14]
e2 = list(G.nodes())[15:G.number_of_nodes()]

se1 = G.subgraph(e1)
se2 = G.subgraph(e2)

# Imprimir información básica sobre el grafo
# plt.figure()
# nx.draw(G, with_labels=True, node_color='skyblue', node_size=500, font_size=10)
# plt.figure()
# nx.draw(se2, with_labels=True, node_color='skyblue', node_size=500, font_size=10)

# preliminary results
# Equipo 1
print ("+ + + + + + + + + + + + + + +")
print("Equipo #1")
print ("- - - - - - - - - - - - - - -")
num_nodos = se1.number_of_nodes()
num_bordes = se1.number_of_edges()
avedegree = nx.average_degree_connectivity(se1)
avedegreeG = sum(avedegree.values())/len(avedegree)
avepathlength = nx.average_shortest_path_length(se1)
aveclustering = nx.average_clustering(se1)
density = nx.density(se1)
diameter = nx.diameter(se1)
radius = nx.radius(se1)

print ("Número de nodos:",num_nodos)
print ("- - - - - - - - - - - - - - -")
print ("Número de bordes:",num_bordes)
print ("- - - - - - - - - - - - - - -")
for i, promedio in avedegree.items():
     print (f"Average Degree {i}",{promedio})
print ("- - - - - - - - - - - - - - -")
print ("Average Degree Global",avedegreeG)
print ("- - - - - - - - - - - - - - -")
print ("Average Path Length",avepathlength)
print ("- - - - - - - - - - - - - - -")
print ("Average Clustering Coefficiente",aveclustering)
print ("- - - - - - - - - - - - - - -")
print ("Density",density)
print ("- - - - - - - - - - - - - - -")
print ("Diameter",diameter)
print ("- - - - - - - - - - - - - - -")
print ("Radius",radius)
print ("+ + + + + + + + + + + + + + +")

print()

# Equipo 2
print ("+ + + + + + + + + + + + + + +")
print("Equipo #2")
print ("- - - - - - - - - - - - - - -")
num_nodos = se2.number_of_nodes()
num_bordes = se2.number_of_edges()
avedegree = nx.average_degree_connectivity(se2)
averagedegreeG = sum(avedegree.values())/len(avedegree)
avepathlength = nx.average_shortest_path_length(se2)
aveclustering = nx.average_clustering(se2)
density = nx.density(se2)
diameter = nx.diameter(se2)
radius = nx.radius(se2)

print ("Número de nodos:",num_nodos)
print ("- - - - - - - - - - - - - - -")
print ("Número de bordes:",num_bordes)
print ("- - - - - - - - - - - - - - -")
for i, promedio in avedegree.items():
     print (f"Average Degree {i}",{promedio})
print ("- - - - - - - - - - - - - - -")
print ("Average Degree Global",avedegreeG)
print ("- - - - - - - - - - - - - - -")
print ("Average Path Length",avepathlength)
print ("- - - - - - - - - - - - - - -")
print ("Average Clustering Coefficiente",aveclustering)
print ("- - - - - - - - - - - - - - -")
print ("Density",density)
print ("- - - - - - - - - - - - - - -")
print ("Diameter",diameter)
print ("- - - - - - - - - - - - - - -")
print ("Radius",radius)
print ("+ + + + + + + + + + + + + + +")


# Node Calulos
#Equipos #1
print("Equipo #1")
print ("- - - - - - - - - - - - - - -")
print("Node    | Closeness ")
print ("- - - - - - - - - - - - - - -")
Closeness = nx.closeness_centrality(se1)
for i, valor in Closeness.items(): 
    print(f"Node {i} | {valor}")
    print ("- - - - - - - - - - - - - - -")
print ("- - - - - - - - - - - - - - -")
print("Node    | Pagerank ")
print ("- - - - - - - - - - - - - - -")
Pagerank = nx.pagerank(se1)
for i, valor in Pagerank.items(): 
    print(f"Nod{i}   | {valor}")
    print ("- - - - - - - - - - - - - - -")

print ("- - - - - - - - - - - - - - -")
print("Node    | Betweenness ")
print ("- - - - - - - - - - - - - - -")
betweenness = nx.betweenness_centrality(se1)
for i, valor in betweenness.items():
    print(f"Node {i} | {valor}")
    print ("- - - - - - - - - - - - - - -")



plt.figure(figsize=(8, 4))
plt.subplot(1, 3, 1)
nx.draw(se1, with_labels=True)
plt.subplot(1, 3, 2)
nx.draw(se1, with_labels=True, node_color=range(13), node_size= [10000*betweenness[node] for node in se1.nodes()])
plt.subplot(1, 3, 3)
nx.draw(se1, with_labels=True, node_color=range(13), node_size= [1000*Closeness[node] for node in se1.nodes()])

plt.show()