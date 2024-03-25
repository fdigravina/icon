import os
import tsplib95
import random

import networkx as nx

class three_opt_agent:
    
    def __init__ (self, filename):
        problem = tsplib95.load (filename)
        self.graph = problem.get_graph ()
        self.env = environment ()
        self.migliore = 0
        self.debug = False
        
        self.problemName = problem.name
        if self.problemName.endswith('.tsp'):
            self.problemName = self.problemName[:-4]
        
        self.theory_solution()
        
        y = self.env.theory_best(self.problemName)
        ans = 2 ** 31
        
        nodi = list (self.graph.nodes)
        r = 0
        
        # christofides
        
        if filename != 'bays29.tsp' and filename != 'brg180.tsp' and filename != 'swiss42.tsp':
            
            perm = nx.algorithms.approximation.christofides (self.graph)
            initial_cost = self.calculate_cost (perm)
            
            #print ('christofides cost:', initial_cost, ', perc:', self.env.perc_error (initial_cost, self.problemName), '%')
            
            if (self.debug):
                print ('\nTWO OPT:\n')
            
            path, cost = self.two_opt (perm)
            
            if (self.debug):
                print ('\nTHREE OPT:\n')
            
            _, ans = self.three_opt (path)
            
            ans = min (ans, cost)
            self.migliore = self.env.perc_error (ans, self.problemName)
            print (f'rep: {r+1}, cost: {cost}, perc: {self.env.perc_error(cost, self.problemName)}% --- nuovo minimo trovato')
        
        
        # nearest neighbour
        
        for start in nodi:
            
            if len(nodi) > 60:
                break
            
            if ans == y:
                break
            
            if (self.debug):
                print ('\n------------------')
            
            r += 1
            perm, initial_cost = self.nearest_neighbor (start)
            #print ('nearest neighbor cost:', initial_cost, ', perc:', self.env.perc_error (initial_cost, self.problemName), '%')
            
            #perm, initial_cost = self.cheapest_insertion (self.graph)
            #print ('cheapest insertion cost:', initial_cost, ', perc:', self.env.perc_error (initial_cost, self.problemName), '%')
            
            if (self.debug):
                print ('\nTWO OPT:\n')
            
            path, cost = self.two_opt (perm)
            
            if (self.debug):
                print ('\nTHREE OPT:\n')
            
            if ans == y:
                break
            
            _, cost = self.three_opt (path)
            
            if cost < ans:
                ans = cost
                self.migliore = self.env.perc_error (ans, self.problemName)
                print (f'rep: {r+1}, cost: {cost}, perc: {self.env.perc_error(cost, self.problemName)}% --- nuovo minimo trovato')
            else:
                print (f'rep: {r+1}, cost: {cost}, perc: {self.env.perc_error(cost, self.problemName)}%')
        
        
        while ans != y:
            
            r += 1
            perm = nodi.copy()
            random.shuffle(perm)
            perm.append(perm[0])
            
            if self.debug:
                print ('\n------------------')
            
            if self.debug:
                print ('\nTWO OPT:\n')
            
            path, cost = self.two_opt (perm)
            
            if self.debug:
                print ('\nTHREE OPT:\n')
            
            if ans == y:
                break
            
            _, cost = self.three_opt (path)
            
            if cost < ans:
                ans = cost
                self.migliore = self.env.perc_error (ans, self.problemName)
                print (f'rep: {r+1}, cost: {cost}, perc: {self.env.perc_error(cost, self.problemName)}% --- nuovo minimo trovato')
            else:
                print (f'rep: {r+1}, cost: {cost}, perc: {self.env.perc_error(cost, self.problemName)}%')
        
        print ('\n\n')
        self.print_solution (ans)
        print ('\n\n')
    
    def return_migliore (self):
        return self.migliore
    
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
        
    def nearest_neighbor(self, start):
        unvisited = list (self.graph.nodes)
        current = start
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
        improvement = True
        iterations = 0
        
        while improvement:
            
            improvement = False
            best = int (2**30)
            
            # cerco di scambiare gli archi (i, i+1) e (j, j+1)
            for i in range (0, num_nodes-1):
                for j in range (i+2, num_nodes):
                    
                    iterations += 1
                    a, b, c, d = curr[i], curr[j], curr[i+1], curr[j+1]
                    delta = self.graph[c][d]['weight'] + self.graph[a][b]['weight'] - self.graph[a][c]['weight'] - self.graph[b][d]['weight']
                    
                    if (delta < best):
                        best, x, y = delta, i+1, j+1
            
            if best < 0:
                curr[x:y] = list(reversed(curr[x:y]))
                cost += best
                improvement = True
                if (self.debug):
                    print (iterations, cost)
        
        return curr, cost
    
    
    def three_opt (self, path):
        
        curr = path
        cost = self.calculate_cost(path)
        
        num_nodes = self.graph.number_of_nodes()
        improvement = True
        iterations = 0
        
        while improvement:
            
            improvement = False
            best = int (2**30)
            
            for i in range (1, num_nodes-2):
                for j in range (i+2, num_nodes-1):
                    for k in range (j+2, num_nodes):
                    
                        iterations += 1
                        
                        segments = [curr[0:i], curr[i:j], curr[j:k], curr[k:]]
                        new_solution = segments[0] + segments[2] + segments[1] + segments[3]
                        new_cost = self.calculate_cost(new_solution)
                        
                        if new_cost < best:
                            best, x, y, z = new_cost, i, j, k
                        
            if best < cost:
                segments = [curr[0:x], curr[x:y], curr[y:z], curr[z:]]
                curr = segments[0] + segments[2] + segments[1] + segments[3]
                cost = best
                improvement = True 
                if (self.debug):
                    print (iterations, cost)

        return curr, cost
    
    def theory_solution (self):
        print (self.graph)
        print ('soluzione teorica ottimale:', self.env.theory_best(self.problemName))
    
    def print_solution (self, ans):
        print ('miglior risposta trovata:', ans)
        print ('errore percentuale:', self.env.perc_error(ans, self.problemName), '%')

class environment:
    
    def theory_best (self, problem_name):
        with open('tsplib95/tsp_best_solutions.txt', 'r') as file:
            for line in file:
                name, distance = line.strip().split(' : ')
                if name == problem_name:
                    return int(distance)
    
    def perc_error (self, ans, problem_name):
        known_best = self.theory_best (problem_name)
        difference = ((ans - known_best) / known_best) * 100
        return round (difference, 2)

filename = 'tsplib95/ulysses22.tsp'
agente_three_opt = three_opt_agent (filename)