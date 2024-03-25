import tsplib95
import random
import itertools
import torch

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

class instance_classifier_agent:
    
    def __init__ (self, n_graphs):
        
        self.n_graphs = n_graphs
        self.train_size = int (self.n_graphs * 0.8)

        self.dataset = []

        for _ in range (self.n_graphs):
            if random.random () < 0.5:
                self.graph, self.label = self.create_dense_graph ()
            else:
                self.graph, self.label = self.create_sparse_graph ()
            if _ % 10 == 0:
                print (_)
            self.dataset.append ((self.graph, self.label))
        
        random.shuffle (self.dataset)
        
        self.train_data = self.dataset[:self.train_size]
        self.test_data = self.dataset[self.train_size:]

        self.train_features = [self.extract_features(graph) for graph, _ in self.train_data]
        self.test_features = [self.extract_features(graph) for graph, _ in self.test_data]

        self.train_labels = [label for _, label in self.train_data]
        self.test_labels = [label for _, label in self.test_data]
        
        self.act ()
    
    def act (self):
        self.decision_tree = DecisionTreeClassifier ()
        self.decision_tree.fit (self.train_features, self.train_labels)
        self.predictions = self.decision_tree.predict (self.test_features)
        self.show_results ()
    
    def show_results (self):
        print ('accuracy:', round(accuracy_score(self.test_labels, self.predictions), 2))
        #self.show_decision_tree()
    
    def show_decision_tree (self):
        plt.figure(figsize=(12, 8), dpi=500)
        plot_tree(self.decision_tree, feature_names=self.get_feature_names(), class_names=['sparso', 'denso'], filled=True)
        plt.savefig('decision_tree.png', format='png')
    
    # dense = 1
    def create_dense_graph (self):
        nodes = random.randint (5, 100)
        p = random.uniform (0.5, 1) # edge probability
        
        g = nx.Graph ()
        g.add_nodes_from (range(nodes))
        
        for u in g.nodes:
            for v in g.nodes:
                if (random.random() < p):
                    g.add_edge (u, v)
        
        return g, 1

    # sparse = 0
    def create_sparse_graph (self):
        nodes = random.randint (5, 100)
        p = random.uniform (0.1, 0.5) # edge probability
        
        g = nx.Graph ()
        g.add_nodes_from (range(nodes))
        
        for u in g.nodes:
            for v in g.nodes:
                if (random.random() < p):
                    g.add_edge (u, v)
        
        return g, 0

    def contiene_cicli (self, graph):
        return not nx.is_forest (graph)

    def get_feature_names(self):
        return ['numero_nodi',
                'numero_archi',
                'densità',
                'ha nodi isolati',
                'contiene_cicli'
                ]

    def extract_features (self, graph):
        features = [
            graph.number_of_nodes (),
            graph.number_of_edges (),
            graph.number_of_nodes () / graph.number_of_edges (),
            len (list(nx.isolates(graph))) == 0,
            self.contiene_cicli (graph)
        ]
        return features

class nn_nearest_neighbour:
    
    def __init__ (self, num_graphs, num_nodes):
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.X_train, self.y_train, self.X_test, self.y_test = self.generate_dataset()
        self.model = self.train_model(self.X_train, self.y_train, self.X_test, self.y_test) # addestra il modello
    
    # Funzione per generare grafi casuali come dati
    def generate_random_graph(self):
        G = nx.complete_graph(range(self.num_nodes))
        for node1 in G.nodes:
            for node2 in G.nodes:
                if node1 != node2:
                    G[node1][node2]['weight'] = random.randint (1, 50)
        return G

    # Funzione per generare il dataset di addestramento e test
    def generate_dataset(self):
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for _ in range(self.num_graphs):
            G = self.generate_random_graph()
            nodes = list(G.nodes())
            random.shuffle(nodes)
            
            # Calcola le distanze tra i nodi
            distances = {}
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i != j:
                        distances[(i, j)] = G[i][j]['weight']
                        distances[(j, i)] = G[j][i]['weight']
                    else:
                        distances[(i, i)] = 500
            
            # Genera il training set
            for i in range(self.num_nodes):
                distances_from_current_node = [distances[(i, j)] for j in range(self.num_nodes)]
                X_train.append(distances_from_current_node)
                y_train.append(np.argmin(distances_from_current_node))

            # Genera il test set
            start_node = random.choice(list(G.nodes()))
            distances_from_start_node = [distances[(start_node, j)] for j in range(self.num_nodes)]
            X_test.append(distances_from_start_node)
            y_test.append(np.argmin(distances_from_start_node))

        return (
            torch.tensor(X_train).float(),
            torch.tensor(y_train).long(),  # Cambio il tipo di dato a 'long'
            torch.tensor(X_test).float(),
            torch.tensor(y_test).long()  # Cambio il tipo di dato a 'long'
        )
    
    # Addestramento del modello
    def train_model(self, X_train, y_train, X_test, y_test):
        
        # Definizione dei possibili valori degli iperparametri
        param_grid = {
            'hidden_size': [64],
            'lr': [0.01],
            'num_epochs': [1000]
        }

        best_accuracy = 0
        best_params = None

        # Iterazione attraverso tutte le combinazioni degli iperparametri
        for params in ParameterGrid(param_grid):
            model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=params['hidden_size'], output_size=self.num_nodes)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=params['lr'])
            self.num_epochs = params['num_epochs']
            
            for epoch in range(self.num_epochs):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
                
                if epoch % 100 == 0:
                    print(f'Epoch [{epoch}/{self.num_epochs}], Loss: {loss.item()}')

            # Valutazione sul set di test
            with torch.no_grad():
                correct = 0
                total = 0
                predicted_nodes = model(X_test)
                for i in range(len(X_test)):
                    predicted_node_index = predicted_nodes[i].argmax().item()
                    if predicted_node_index == y_test[i]:
                        correct += 1
                    total += 1

                accuracy = correct / total
                print(f"Parameters: {params}, Accuracy: {accuracy*100}%")

                # Aggiornamento dei migliori iperparametri
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = params

        print(f"Best parameters: {best_params}, Best accuracy: {best_accuracy}")

        return model

# Definizione del modello di regressione
class NeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

class dp_agent:
    
    def __init__ (self, filename):
        self.inf = 2 ** 32   # valore che simula infinito
        self.filename = filename
        self.env = environment ()
        self.read ()
        self.percentage_error()
    
    def distance (self, a, b):
        return self.graph[a][b]['weight']

    def read (self):
        
        f = open (self.filename)
        
        self.problem = tsplib95.load (self.filename)
        self.graph = self.problem.get_graph ()
        self.n = self.graph.number_of_nodes ()
        
        self.problemName = self.problem.name
        if self.problemName.endswith('.tsp'):
            self.problemName = self.problemName[:-4]
        
        self.theory_solution()
        
        nodes = list(self.graph.nodes())
        self.adj = np.zeros(shape=(self.n, self.n))
        
        offset = min(nodes)
        
        for i in nodes:
            for j in nodes:
                self.adj[i-offset][j-offset] = self.distance (i, j)
        
        self.dp = [[-1 for _ in range(2 ** self.n)] for _ in range (self.n)]   # inizializzo la matrice dp
        self.parent = [[-1 for _ in range(2 ** self.n)] for _ in range (self.n)]   # inizializzo la matrice parent


    def tsp (self, pos, bitmask):
        
        if bitmask == 2 ** self.n - 1:
            return self.adj[pos][0]
        
        if self.dp[pos][bitmask] != -1:
            return self.dp[pos][bitmask]
        
        ans = self.inf
        
        for nxt in range(self.n):
            if nxt != pos and (bitmask & (2 ** nxt)) == 0:
                curr = self.adj[pos][nxt] + self.tsp(nxt, bitmask | (2 ** nxt))
                if curr < ans:
                    ans = curr
                    self.parent[pos][bitmask] = nxt
        
        self.dp[pos][bitmask] = ans
        return int(self.dp[pos][bitmask])

    def reconstruct_path (self):
        
        path = []
        curr = 0
        mask = 1

        while curr != -1:
            path.append (curr)
            curr = self.parent[curr][mask]
            mask = mask | int(2 ** curr)

        path.append (path[0])
        return path
    
    def theory_solution (self):
        print (self.graph)
        print ('soluzione teorica ottimale:', self.env.theory_best(self.problemName))
    
    def percentage_error (self):
        #print ('distanza trovata: ', agente_dp.tsp(0, 1))
        #print ('percorso trovato: ', agente_dp.reconstruct_path())
        print ('errore percentuale: ', self.env.perc_error(self.tsp(0, 1), self.problemName))

class three_opt_agent:
    
    def __init__ (self, filename, rep, iterations_two, iterations_three):
        problem = tsplib95.load (filename)
        self.graph = problem.get_graph ()
        self.iterations_two = iterations_two
        self.iterations_three = iterations_three
        self.env = environment ()
        self.migliore = 0
        
        self.problemName = problem.name
        if self.problemName.endswith('.tsp'):
            self.problemName = self.problemName[:-4]
        
        self.theory_solution()
        
        y = self.env.theory_best(self.problemName)
        
        ans = 2 ** 31
        
        # christofides è lo stesso per ogni ripetizione
        if rep == 0:
            
            perm = nx.algorithms.approximation.christofides (self.graph)
            initial_cost = self.calculate_cost (self.graph, perm)
            print ('christofides cost:', initial_cost, ', perc:', self.env.perc_error (initial_cost, self.problemName), '%')
            
            print ('two opt:')
            path, cost = self.two_opt (self.graph, perm)
            
            print ('three opt:')
            _, ans = self.three_opt (self.graph, path)
            
            ans = min (ans, cost)
            self.migliore = self.env.perc_error (ans, self.problemName)
        
        else:
            for r in range (rep):
                
                perm, initial_cost = self.nearest_neighbor (self.graph)
                #print ('nearest neighbor cost:', initial_cost, ', perc:', self.env.perc_error (initial_cost, self.problemName), '%')
                
                #perm, initial_cost = self.cheapest_insertion (self.graph)
                #print ('cheapest insertion cost:', initial_cost, ', perc:', self.env.perc_error (initial_cost, self.problemName), '%')
                
                #print ('two opt:')
                path, cost = self.two_opt (self.graph, perm)
                
                #print ('three opt:')
                if (self.env.perc_error(cost, self.problemName) <= 2.0):
                    _, cost = self.three_opt (self.graph, path)
                
                ans = min (ans, cost)
                print (f'rep: {r+1}, cost: {cost}, perc: {self.env.perc_error(cost, self.problemName)}%')
                #self.migliore = self.env.perc_error (ans, self.problemName)
                
                if ans == y:
                    break
            
        self.print_solution (ans)
    
    def return_migliore (self):
        return self.migliore
    
    def calculate_cost (self, graph, perm):
        sum = 0
        for i in range (len(perm) - 1):
            sum += graph[perm[i]][perm[i+1]]['weight']
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
        return path, self.calculate_cost (graph, path)
    
    def get_tsp_path (self, graph, start):
        tsp_path = list (nx.dfs_preorder_nodes(graph, source=start))
        tsp_path.append (tsp_path[0])
        return tsp_path
        
    def nearest_neighbor(self, graph):
        unvisited = list (graph.nodes)
        current = random.choice (unvisited)
        path = [current]
        unvisited.remove (current)

        while unvisited:
            closest = min(unvisited, key=lambda x: graph[current][x]['weight'])
            path.append (closest)
            unvisited.remove (closest)
            current = closest

        path.append (path[0])
        return path, self.calculate_cost (graph, path)
    
    
    def two_opt(self, graph, best_solution):
        current_solution = best_solution[:]
        best_cost = self.calculate_cost(graph, best_solution)
        num_nodes = len (current_solution)
        
        for _ in range(self.iterations_two):
            selected_indices = sorted(random.sample(range(1, num_nodes), 2))
            new_solution = current_solution[:]

            new_solution[selected_indices[0]:selected_indices[1]] = list(reversed(new_solution[selected_indices[0]:selected_indices[1]]))
            new_cost = self.calculate_cost(graph, new_solution)

            if new_cost < best_cost:
                current_solution = new_solution[:]
                best_cost = new_cost
                #print (_, best_cost)
        
        return current_solution, best_cost
    
    
    def three_opt (self, graph, best_solution):
        current_solution = best_solution[:]
        best_cost = self.calculate_cost(graph, best_solution)
        num_nodes = len (current_solution)
        
        for it in range(self.iterations_three):
            
            selected_indices = sorted(random.sample(range(1, num_nodes), 3))
            
            for (i, j, k) in itertools.permutations(selected_indices):
                
                segments = [current_solution[0:i], current_solution[i:j], current_solution[j:k], current_solution[k:]]
                new_solution = segments[0] + segments[2] + segments[1] + segments[3]
                new_cost = self.calculate_cost(graph, new_solution)
                
                if new_cost < best_cost:
                    current_solution = new_solution[:]
                    best_cost = new_cost
                    #print (it, best_cost)

        return current_solution, best_cost
    
    def theory_solution (self):
        print (self.graph)
        print ('soluzione teorica ottimale:', self.env.theory_best(self.problemName))
    
    def print_solution (self, ans):
        print ('miglior risposta trovata:', ans)
        print ('errore percentuale:', self.env.perc_error(ans, self.problemName), '%')

class cluster_three_opt_agent (three_opt_agent):
    
    def __init__ (self, filename, k, rep, iterations_two, iterations_three):
        
        problem = tsplib95.load (filename)
        self.graph = problem.get_graph ()
        self.iterations_two = iterations_two
        self.iterations_three = iterations_three
        self.env = environment ()
        
        self.problemName = problem.name
        if self.problemName.endswith('.tsp'):
            self.problemName = self.problemName[:-4]
        
        self.theory_solution()
        self.k = k   # numero di cluster
        self.kmeans ()
        ans = 2 ** 31
        
        for r in range (rep):
            _, cost = self.hybrid_three_opt (self.graph)
            ans = min (ans, cost)
            print (self.calculate_cost(self.graph, _))
            print ('rep:', r+1, ', cost:', cost, ', perc:', self.env.perc_error(cost, self.problemName), '%')
            
        self.print_solution (ans)

    def distance(self, a, b):
        return self.graph[a][b]['weight']

    # Definizione della funzione per assegnare i punti ai cluster
    def assign_clusters (self):
        clusters = [[] for _ in range(self.k)]
        for node in self.graph.nodes:
            distances = [self.distance(node, centroid) for centroid in self.centroids]
            cluster = np.argmin(distances)
            clusters[cluster].append(node)
        return clusters
    
    def kmeans (self):
        self.centroids = random.sample(list(self.graph.nodes), self.k) # indici dei centroidi
        self.clusters = self.assign_clusters()

        '''for node in self.graph.nodes:
            distances = [self.distance(node, centroid) for centroid in self.centroids]
            cluster_label = np.argmin(distances)
            print(f"Nodo {node} nel cluster {cluster_label}")'''
        
        for i in range(len(self.clusters)):
            print ('cluster', i, 'ha lunghezza', len(self.clusters[i]))
        
    def hybrid_three_opt (self, graph):
        
        optimized_paths = []
        
        for cluster in self.clusters:
            if len(cluster) > 3:
                subgraph = graph.subgraph(cluster)
                print ('cluster', self.clusters.index(cluster))
                optimized_path, _ = self.two_opt(subgraph, list(subgraph.nodes()))
                optimized_paths.append (optimized_path)
            else:
                optimized_paths.append (cluster)
        
        final_path = list (itertools.chain.from_iterable(optimized_paths))
        final_path.append (final_path[0])
        
        path, _ = self.two_opt (self.graph, final_path)
        return self.three_opt (graph, path)

class genetic_agent:
    
    def __init__ (self, filename, max_iterations):
        self.num_iterazioni = 10
        self.errori = []
        self.controllo = 1
        self.mutation_rate = 0.001
        self.best_solution = 1e15 
        self.cambia = False
        self.file_path = filename
        self.data = self.read_data()
        self.file_path = filename[9:]
        self.num_citta = self.data.number_of_nodes()
        self.max_iterations = max_iterations
        self.genetic_algorithm()
        
    
    def read_data(self):
        self.problem = tsplib95.load(self.file_path)
        self.Graph = self.problem.get_graph()
        return self.Graph

    def calcola_distanze_percorso (self, percorso):
        d = 0
        for i in range(0, len(self.data.nodes)-1):
            d += self.data[int(percorso[i])][int(percorso[i+1])]['weight']
        d += self.data[int(percorso[-1])][int(percorso[0])]['weight']
        return d

    def nearest_neighbor(self, node):
        unvisited = list(self.data.nodes)
        current = node
        path = [current]
        unvisited.remove(current)

        while unvisited:
            closest = min(unvisited, key=lambda x: self.data[current][x]['weight'])
            path.append(closest)
            unvisited.remove(closest)
            current = closest

        return path
    
    def genetic_algorithm (self):
        
        self.miglior_distanza = 1e15
        while (self.controllo <= self.max_iterations):
            print (self.cambia)
            self.dimensione = int(self.num_citta*self.controllo)
            self.fitness = np.zeros(self.dimensione)
            self.max_fitness = 0.0
            self.ordine = [i for i in range(min(self.data.nodes), max(self.data.nodes)+1)]
            
            if not self.cambia:
                self.popolazione = np.zeros((self.dimensione, self.num_citta))
            else:
                new_popolazione = np.zeros ((self.dimensione, self.num_citta))
                for i in range (len(self.popolazione)):
                    for j in range (len(self.popolazione[0])):
                        new_popolazione[i][j] = self.popolazione [i][j]
                self.popolazione = new_popolazione
            self.miglior_percorso = []
            
            if not self.cambia:
                for i in range(self.num_citta):
                    self.popolazione[i] = self.nearest_neighbor(i+min(self.data.nodes))
            
                for i in range(self.num_citta, self.dimensione):
                    self.popolazione[i] = random.sample(self.ordine, len(self.ordine))
            else :
                for i in range(self.num_citta*(self.controllo-1),len(self.popolazione)):
                    self.popolazione[i] = random.sample(self.ordine, len(self.ordine))
                    self.cambia = False
            
            print('teoria:', self.theory_best(self.file_path))
            
            for i in range(self.num_iterazioni):
                self.fitness, self.miglior_distanza, self.miglior_percorso = self.calcola_fitness()
                self.fitness = self.normalizzazione_vettoriale(self.fitness)
                self.popolazione = self.nuova_generazione()

            print('perc_error:', self.perc_error(self.miglior_distanza))
            
            if (self.controllo <= self.max_iterations):
                self.controllo += 1
                self.num_iterazioni += self.controllo % 2 
                self.mutation_rate += 0.001
                print (f"nuova esecuzione con: \nmutation rate = {round(self.mutation_rate, 2)}"
                       f"\ndimensione della popolazione = {self.num_citta*self.controllo}\nnumero generazioni = {self.num_iterazioni}")
                
                if (self.best_solution > self.miglior_distanza):
                    self.best_solution = min(self.miglior_distanza, self.best_solution) 
                    self.cambia = True
                self.errori.append([self.perc_error(self.miglior_distanza),
                            self.mutation_rate, self.num_iterazioni])
            else:
                self.errori.append([self.perc_error(self.miglior_distanza),
                            self.mutation_rate, self.num_iterazioni])
                for risultato in self.errori:
                    print(
                        f"File: {self.file_path}, Perc Error: {risultato[0]}, mutation_rate: {round(risultato[1], 2)}, generazioni: {risultato[2]}")
                    i += 1
    
    
    def theory_best (self, problem_name):
        self.problem_name = problem_name
        if self.problem_name.endswith('.tsp'):
            self.problem_name = problem_name[:-4]
        
        with open('tsplib95/tsp_best_solutions.txt', 'r') as file:
            for line in file:
                name, distance = line.strip().split(' : ')
                if name == self.problem_name:
                    return int(distance)

    def perc_error(self, ans):
        known_best = self.theory_best(self.problem.name)
        difference = ((ans - known_best) / known_best) * 100
        return round(difference, 2)

    def calcola_fitness(self):
        for i in range(len(self.popolazione)):
            d = self.calcola_distanze_percorso(np.array(self.popolazione[i]))
            if d < self.miglior_distanza:
                self.miglior_distanza = d
                self.miglior_percorso = self.popolazione[i]
            self.fitness[i] = 1 / (pow(d, 8) + 1)
        return self.fitness, self.miglior_distanza, self.miglior_percorso

    def normalizzazione_vettoriale(self, vettore):
        somma = np.sum(vettore)
        for i in range(len(vettore)):
            vettore[i] = vettore[i]/somma
        return vettore

    def nuova_generazione(self):
        new_population = np.zeros_like(self.popolazione)
        for i in range(len(self.popolazione)):
            genitore_1, genitore_2 = np.random.choice(
                len(self.popolazione), 2, p=self.fitness.ravel(), replace=False)
            figlio = self.crossover(
                self.popolazione[genitore_1], self.popolazione[genitore_2])
            figlio = self.mutate(figlio)
            new_population[i] = figlio
        return new_population

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            idx1, idx2 = np.random.choice(
                len(individual), size=2, replace=False)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(len(parent1))
        child1 = np.zeros_like(parent1)
        # Crossover operation
        child1[:crossover_point] = parent1[:crossover_point]
        child1[crossover_point:] = [
            gene for gene in parent2 if gene not in parent1[:crossover_point]]
        return child1

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

filename = 'tsplib95/ulysses16.tsp'

#rete_neurale = nn_nearest_neighbour(num_graphs=100, num_nodes=6)
#agente_cluster = cluster_three_opt_agent (filename, k=5, rep=2, iterations_two=50000, iterations_three=3000)
#agente_three_opt = three_opt_agent (filename, rep=200, iterations_two=8500, iterations_three=0)
#agente_genetico = genetic_agent(filename, max_iterations=10)
agente_dp = dp_agent(filename)
