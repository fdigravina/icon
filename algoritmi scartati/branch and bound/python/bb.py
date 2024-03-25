from math import factorial
import networkx as nx
import tsplib95

problem = tsplib95.load ('tsplib95/burma14.tsp')

G = problem.get_graph ()
n = G.number_of_nodes()

lower = min(G[u][v]['weight'] for u, v in G.edges)

sol = []
bound = 3350
cont = 0

def bb (path, cost, remaining):
    
    global sol
    global bound
    global cont
    
    m = len(path)
    cont += 1
    
    if cont % 10000 == 0:
        print (cont)
    
    if m == n + 1:
        if cost < bound and path[0] == path[-1]:
            sol, bound = path, cost
        return
    
    if cost + lower * (n+1 - m) > bound:
        return
    
    for x in remaining:
        
        if x == path[-1]:
            continue
        
        r = remaining.copy()
        r.remove(x)
        
        p = path.copy()
        p.append(x)
        
        peso = cost
        
        if len(p) > 1:
            peso += G[path[-1]][x]['weight']
        
        bb (p, peso, r)
        


s = set(G.nodes)

for i in G.nodes:
    bb ([i], 0, s)

print ('path:', sol, 'costo:', bound)
print ('esplorati:', cont, 'su:', factorial(n), ', %:', round(cont*100/factorial(n+1), 2))