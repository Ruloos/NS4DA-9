import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tabulate import tabulate

# Se carga la infomacion del archivo excel
# dr = pd.read_excel('Table 1.xlsx',sheet_name='player_code_name',index_col=0) 
df = pd.read_excel('Table 1.xlsx',sheet_name='matrix_passes_between_players',index_col=0)

# Crea un grafo dirigido desde el DataFrame
G = nx.from_pandas_adjacency(df)
# R = nx.from_pandas_edgelist(dr, source = 'NODE NUMBER', target = 'PLAYER NAME', )

# Se separa los nodos por equipo
e1 = list(G.nodes())[0:14]
e2 = list(G.nodes())[14:G.number_of_nodes()]

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

print ("- - - - - - - - - - - - - - -")
for i, promedio in avedegree.items():
     print (f"Average Degree {i}",{promedio})

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

for i, promedio in avedegree.items():
     print (f"Average Degree {i}",{promedio})

# Node Calulos
#Equipos #1
print("Equipo #1")
Closeness = nx.closeness_centrality(se1)
Pagerank = nx.pagerank(se1)
betweenness = nx.betweenness_centrality(se1)
data = []
for node in se1.nodes():
     data.append([node,Closeness[node],Pagerank[node],betweenness[node]])
data_sorted = sorted(data,key=lambda x: x[0])
headers = ["Nodo", "Closeness Centrality", "PageRank","Betweenness"]
print(tabulate(data_sorted, headers=headers, tablefmt="grid"))

pos = nx.spring_layout(se1)
plt.figure()
plt.title("Network Equipo#1")
nx.draw(se1,pos, with_labels=True, node_color=range(14),alpha=0.8)
plt.figure()
plt.title("Betweeness Equipo#1")
nx.draw(se1,pos, labels={node: f"{betweenness[node]:.2f}" for node in se1.nodes()}, node_color=range(14), node_size= [100000*betweenness[node] for node in se1.nodes()],alpha=0.8)
plt.figure()
plt.title("Closeness Equipo#1")
nx.draw(se1,pos, labels={node: f"{Closeness[node]:.2f}" for node in se1.nodes()}, node_color=range(14), node_size= [1000*Closeness[node] for node in se1.nodes()],alpha=0.8)

#Equipos #2
print("Equipo #2")
Closeness = nx.closeness_centrality(se2)
Pagerank = nx.pagerank(se2)
betweenness = nx.betweenness_centrality(se2)
data = []
for node in se2.nodes():
     data.append([node,Closeness[node],Pagerank[node],betweenness[node]])
data_sorted = sorted(data,key=lambda x: x[0])
headers = ["Nodo", "Closeness Centrality", "PageRank","Betweenness"]
print(tabulate(data_sorted, headers=headers, tablefmt="grid"))


# for i, valor in Closeness.items(): 
#     print(f"Node {i} | {valor}")
#     print ("- - - - - - - - - - - - - - -")
# print ("- - - - - - - - - - - - - - -")
# print("Node    | Pagerank ")
# print ("- - - - - - - - - - - - - - -")

# for i, valor in Pagerank.items(): 
#     print(f"Node {i}  | {valor}")
#     print ("- - - - - - - - - - - - - - -")

# print ("- - - - - - - - - - - - - - -")
# print("Node    | Betweenness ")
# print ("- - - - - - - - - - - - - - -")

# for i, valor in betweenness.items():
#     print(f"Node {i} | {valor}")
#     print ("- - - - - - - - - - - - - - -")


pos = nx.spring_layout(se2)
plt.figure()
plt.title("Network Equipo#2")
nx.draw(se2,pos, with_labels=True, node_color=range(14),alpha=0.8)
plt.figure()
plt.title("Betweeness Equipo#2")
nx.draw(se2,pos, labels={node: f"{betweenness[node]:.2f}" for node in se2.nodes()}, node_color=range(14), node_size= [100000*betweenness[node] for node in se2.nodes()],alpha=0.8)
plt.figure()
plt.title("Closeness Equipo#2")
nx.draw(se2,pos, labels={node: f"{Closeness[node]:.2f}" for node in se2.nodes()}, node_color=range(14), node_size= [1000*Closeness[node] for node in se2.nodes()],alpha=0.8)


plt.show()