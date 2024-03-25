import random
import csv
import networkx as nx
import matplotlib.pyplot as plt
import folium
from geopy.geocoders import Nominatim
import geopandas as gpd
import numpy as np
import osmnx as ox
import warnings

from shapely.ops import unary_union
from shapely.geometry import Point, LineString, MultiPolygon, Polygon

warnings.simplefilter(action='ignore', category=FutureWarning)

def confini():
    
    italy = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    italy = italy[italy.name == 'Italy']

    #italy = italy[~italy.name.isin(['Sicily', 'Sardinia'])]

    italy_polygon = italy.explode()
    
    coordinate_lista = []

    # Itera attraverso i poligoni all'interno di italy_polygon
    for poligono in italy_polygon.geometry:
        # Ottieni le coordinate del poligono e aggiungile alla lista
        coordinate_poligono = list(poligono.exterior.coords)
        coordinate_lista.append(coordinate_poligono)
    
    # Separate the x and y coordinates
    x_coords = [coord[0] for coord in coordinate_lista[0]]
    y_coords = [coord[1] for coord in coordinate_lista[0]]
    

    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot of Coordinates')

    plt.grid(True)
    plt.show()

    
    #print(coordinate_lista[0])
    return coordinate_lista[0]

def is_segment_within_polygon(segment, polygon_coords):
    polygon = Polygon(polygon_coords)
    segment_line = LineString(segment)
    
    '''
    print (polygon_coords)
    print (segment_line)
    xs, ys = zip(*polygon_coords)
    plt.figure()
    plt.plot(xs,ys) 
    plt.show()
    '''
    
    #print (polygon, segment_line)
    #print (segment_line.intersects(polygon.boundary))
    return not segment_line.intersects(polygon.boundary)


def readGraph ():
    
    G = nx.Graph()
    diz1 = {}
    diz2 = {}
    
    with open('progetto finale/nomi.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            id_val = int(row[0])
            nome = row[1]
            diz1[id_val] = nome
            diz2[nome] = id_val
            G.add_node(id_val)
    
    lista = {}
    with open ('progetto finale/comuni.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            lista[row[0]] = (row[1], row[2])  # nome, lat, lon
    
    with open ('progetto finale/dataset.csv', 'r', encoding='utf-8') as f:
        ita = confini()
        reader = csv.reader(f)
        for row in reader:
            if row[0] != row[1]:
                    seg = [(float(lista[row[0]][1]), float(lista[row[0]][0])), (float(lista[row[1]][1]), float(lista[row[1]][0]))]
                    if is_segment_within_polygon(seg, ita):
                        G.add_edge(diz2[row[0]], diz2[row[1]], weight=float(row[2]))
                    else:
                        G.add_edge(diz2[row[0]], diz2[row[1]], weight=int(2**30))
    
    n = G.number_of_nodes()
    
    listaMeteo = np.zeros(n)
    
    with open('progetto finale/predizione.txt', 'r') as f:
        righe = f.readlines()
        for riga in righe:
            line = riga[18:]
            nodo, meteo = line.strip().split(': ')
            meteoNum = 0
            if meteo == 'pioggia_lieve':
                meteoNum = 1
            if meteo == 'pioggia_media':
                meteoNum = 2
            if meteo == 'pioggia_forte':
                meteoNum = 3
            listaMeteo[int(nodo)] = meteoNum
    
    with open ('progetto finale/predici.pl', 'r') as f:
        righe = f.readlines()
        for riga in righe:
            if riga.startswith('situazione'):
                nodo, meteo = riga.strip().split(", ")[0][11:], riga.strip().split(", ")[1][:-2]
                meteoNum = 0
                if meteo == 'pioggia_lieve':
                    meteoNum = 1
                if meteo == 'pioggia_media':
                    meteoNum = 2
                if meteo == 'pioggia_forte':
                    meteoNum = 3
                listaMeteo[int(nodo)] = meteoNum
    
    GPure = G.copy()
    
    dissestata = random.choices(population=[0, 1, 2, 3], weights=[0.8, 0.15, 0.04, 0.01], k=n**2)    # condizioni normali, quasi normali, dissestata, quasi inagibile
    lavori_in_corso = random.choices (population=[0, 1], weights=[0.95, 0.05], k=n**2)
    
    for i in G.nodes:
        for j in G.nodes:
            
            if i == j:
                continue
            
            if G[i][j]['weight'] == 2 ** 30:
                continue
            
            dissestamento = max(dissestata[i], dissestata[j])
            lavori = max (lavori_in_corso[i], lavori_in_corso[j])
            meteo = max (listaMeteo[i], listaMeteo[j])
            
            #print (G[i][j]['weight'], type(G[i][j]['weight']))
            
            if dissestamento == 1:
                G[i][j]['weight'] = float(G[i][j]['weight']) * 1.1
                G[j][i]['weight'] = float(G[j][i]['weight']) * 1.1
            
            if dissestamento == 2:
                G[i][j]['weight'] = float(G[i][j]['weight']) * 1.2
                G[j][i]['weight'] = float(G[j][i]['weight']) * 1.2
            
            if dissestamento == 3:
                G[i][j]['weight'] = float(G[i][j]['weight']) * 1.4
                G[j][i]['weight'] = float(G[j][i]['weight']) * 1.4
            
            if lavori == 1:
                G[i][j]['weight'] = float(G[i][j]['weight']) * 1.5
                G[j][i]['weight'] = float(G[j][i]['weight']) * 1.5
            
            if meteo == 1:
                G[i][j]['weight'] = float(G[i][j]['weight']) * 1.1
                G[j][i]['weight'] = float(G[j][i]['weight']) * 1.1
            
            if meteo == 2:
                G[i][j]['weight'] = float(G[i][j]['weight']) * 1.2
                G[j][i]['weight'] = float(G[j][i]['weight']) * 1.2
            
            if meteo == 3:
                G[i][j]['weight'] = float(G[i][j]['weight']) * 1.3
                G[j][i]['weight'] = float(G[j][i]['weight']) * 1.3
            
            G[i][j]['weight'] = int(G[i][j]['weight'])
            G[j][i]['weight'] = int(G[j][i]['weight'])
            
            GPure[i][j]['weight'] = int(GPure[i][j]['weight'])
            GPure[j][i]['weight'] = int(GPure[j][i]['weight'])
        
    return G, GPure, diz1, diz2


class three_opt_agent:
    
    def __init__ (self, graph, rep, two, three):
        self.migliore = 0
        self.debug = True
        self.two = two
        self.three = three
        ans = 2 ** 31
        
        self.graph = graph
        
        #nodi = list (self.graph.nodes)
        r = -1
        
        # christofides
        
        r += 1
        
        perm = nx.algorithms.approximation.christofides (self.graph)
        _ = self.calculate_cost (perm)
        
        if (self.debug):
            print ('\nTWO OPT:\n')
        
        path, cost = self.two_opt (perm)
        
        if (self.debug):
            print ('\nTHREE OPT:\n')
        
        path, ans = self.three_opt (path)
        
        ans = cost
        p = path
        print (f'\nrep: {r+1}, cost: {cost} --- nuovo minimo trovato')
    
        # nearest neighbour
        
        for _ in range (1, rep):
            
            if (self.debug):
                print ('\n------------------')
            
            r += 1
            perm, _ = self.nearest_neighbor()
            #print ('nearest neighbor cost:', initial_cost, ', perc:', self.env.perc_error (initial_cost, self.problemName), '%')
            
            #perm, initial_cost = self.cheapest_insertion (self.graph)
            #print ('cheapest insertion cost:', initial_cost, ', perc:', self.env.perc_error (initial_cost, self.problemName), '%')
            
            if (self.debug):
                print ('\nTWO OPT:\n')
            
            path, cost = self.two_opt (perm)
            
            if (self.debug):
                print ('\nTHREE OPT:\n')
            
            path, cost = self.three_opt (path)
            
            if cost < ans:
                ans = cost
                p = path
                print (f'\nrep: {r+1}, cost: {cost} --- nuovo minimo trovato')
            else:
                print (f'\nrep: {r+1}, cost: {cost}')
        
        self.p = p
        
        print (f'\nmiglior risultato: {ans}')
    
    def returnPath(self):
        return self.p
    
    def calculate_cost (self, perm):
        sum = 0
        for i in range (len(perm) - 1):
            sum += self.graph[perm[i]][perm[i+1]]['weight']
        return sum

    def cheapest_insertion(self, graph):
        solution_graph = nx.Graph() # grafo vuoto
        initial_node = random.sample(range(min(graph.nodes()), max(graph.nodes())+1), 1)[0]   # torna una lista con un elemento e lo prendo
        solution_graph.add_node (initial_node)

        while len(solution_graph.nodes) < len(graph.nodes):
            min_cost = float('inf')
            min_edge = None
            
            # Cerca l'arco più conveniente da inserire
            for node in solution_graph.nodes:
                for neighbor in graph.nodes:
                    if neighbor not in solution_graph.nodes:
                        cost = graph[node][neighbor]['weight']
                        if cost < min_cost:
                            min_cost = cost
                            min_edge = (node, neighbor)

            solution_graph.add_node(min_edge[1])
            solution_graph.add_edge(*min_edge)
        
        path = self.get_tsp_path (solution_graph, initial_node)
        return path, self.calculate_cost (path)
    
    def get_tsp_path (self, graph, start):
        tsp_path = list (nx.dfs_preorder_nodes(graph, source=start))
        tsp_path.append (tsp_path[0])
        return tsp_path
        
    def nearest_neighbor(self):
        unvisited = list (self.graph.nodes)
        current = random.choice (unvisited)
        path = [current]
        unvisited.remove (current)

        while unvisited:
            closest = min(unvisited, key=lambda x: self.graph[current][x]['weight'])
            path.append (closest)
            unvisited.remove (closest)
            current = closest

        path.append (path[0])
        return path, self.calculate_cost (path)
    
    
    def two_opt (self, path):
        
        curr = path
        cost = self.calculate_cost(path)
        
        num_nodes = self.graph.number_of_nodes()
        
        for it in range (self.two):
            
            idxs = sorted(random.sample(range(1, num_nodes), 2))
            i, j = idxs[0], idxs[1]
            
            a, b, c, d = curr[i], curr[j], curr[i+1], curr[j+1]
            delta = self.graph[c][d]['weight'] + self.graph[a][b]['weight'] - self.graph[a][c]['weight'] - self.graph[b][d]['weight']
            
            if delta < 0:
                curr[i+1:j+1] = list(reversed(curr[i+1:j+1]))
                cost += delta
                if (self.debug):
                    print (it, cost)
        
        return curr, cost
    
    
    def swap_three (self, i, j, k, case):
        
        if case == 0:
            newtour = self.curr[:i+1]
            newtour.extend(self.curr[j+1:k+1])
            newtour.extend(reversed(self.curr[i+1:j+1]))
            newtour.extend(self.curr[k+1:])
        
        elif case == 1:
            newtour = self.curr[:i+1]
            newtour.extend(reversed(self.curr[j+1:k+1]))
            newtour.extend(self.curr[i+1:j+1])
            newtour.extend(self.curr[k+1:])

        elif case == 2:
            newtour = self.curr[:i+1]
            newtour.extend(reversed(self.curr[i+1:j+1]))
            newtour.extend(reversed(self.curr[j+1:k+1]))
            newtour.extend(self.curr[k+1:])

        elif case == 3:
            newtour = self.curr[:i+1]
            newtour.extend(self.curr[j+1:k+1])
            newtour.extend(self.curr[i+1:j+1])
            newtour.extend(self.curr[k+1:])
        
        return newtour
    
    def three_opt (self, path):
        
        self.curr = path
        cost = self.calculate_cost(path)
        
        num_nodes = self.graph.number_of_nodes()
        
        for it in range (self.three):
            
            idxs = sorted(random.sample(range(1, num_nodes), 3))
            i, j, k = idxs[0], idxs[1], idxs[2]
            
            for case in range (4):
                
                new_solution = self.swap_three (i, j, k, case)
                new_cost = self.calculate_cost(new_solution)
                
                if new_cost < cost:
                    self.curr = new_solution[:]
                    cost = new_cost
                    if (self.debug):
                        print (it, cost)

        return self.curr, cost

G, GPure, diz1, diz2 = readGraph()

agente_three_opt = three_opt_agent (GPure, rep=2, two=200000, three=50000)

path = agente_three_opt.returnPath()
agente_three_opt = three_opt_agent (G, rep=2, two=200000, three=50000)

path2 = agente_three_opt.returnPath()
#for p in path:
#    print (diz1[p])

#print ('-------------------------------')
#for i in range (len(path) - 1):
#    if G[path[i]][path[i+1]]['weight'] > 2**20:
#        print (diz1[path[i]], diz1[path[i+1]])
#print ('-------------------------------')
citta2 = []
citta1 = []
print(path)
print (path2)
for p in path:
    citta1.append(diz1[p])
for p in path2:
    citta2.append(diz1[p])

#print (path)
for citta in [citta1,citta2]:
    geolocator = Nominatim(user_agent="city_route")
    mappa = folium.Map(location=None, zoom_start=6)

    coordinate = []
    lista = []

    with open ('progetto finale/comuni.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            lista.append((row[0], row[1], row[2]))
    if citta == citta2:
        for p in path2:
            for r in lista:
                if r[0] == diz1[p]:
                    coordinate.append((float(r[1]), float(r[2])))
    else:
        for p in path:
            for r in lista:
                if r[0] == diz1[p]:
                    coordinate.append((float(r[1]), float(r[2])))
                    #print (r[0])
    if citta == citta2:
        folium.PolyLine(locations=coordinate, color='blue', weight=5).add_to(mappa)
    else:
        folium.PolyLine(locations=coordinate, color='red', weight=5).add_to(mappa)
    for i in range(len(coordinate)):
        folium.Marker(coordinate[i], popup=citta[i]).add_to(mappa)

    folium.Marker(coordinate[0], popup=citta[0]).add_to(mappa)

    mappa.fit_bounds([[36.6, 6.2], [47.0, 19.0]])
    if citta == citta1:
        mappa.save('progetto finale/city_route_map_pure.html')
    else:
        mappa.save('progetto finale/city_route_map_with_weight.html')