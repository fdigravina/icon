import time
import random
import math
import networkx as nx
import tsplib95
import os


class TSPGeneticAlgorithm:
    def __init__(self, graph):
        self.graph = graph
        self.node_list = list(self.graph.nodes)
        self.num_nodes = len(self.node_list)

    def total_distance(self, tour):
        distance = 0
        for i in range(len(tour)):
            distance += self.graph[tour[i]
                                   ][tour[(i + 1) % len(tour)]]['weight']
        return distance

    def nearest_neighbor(self):
        unvisited_cities = set(self.node_list)
        current_city = random.choice(self.node_list)
        unvisited_cities.remove(current_city)
        tour = [current_city]
        while unvisited_cities:
            nearest_city = min(
                unvisited_cities, key=lambda city: self.graph[current_city][city]['weight'])
            unvisited_cities.remove(nearest_city)
            tour.append(nearest_city)
            current_city = nearest_city
        return tour

    def genetic_algorithm(self, population_size, elite_size, mutation_rate, generations):
        elite_size = int(self.num_nodes * elite_size / 100)
        population = [self.nearest_neighbor() for _ in range(population_size)]
        for generation in range(generations):
            if generation % 1000 == 0:    
                print ("--- La soluzione trovata all'iterazione", generation, " e': ",
                   self.total_distance(min(population, key=lambda tour: self.total_distance(tour))))
            population = sorted(
                population, key=lambda tour: self.total_distance(tour))
            next_generation = population[:elite_size]
            while len(next_generation) < population_size:
                if random.random() < mutation_rate:
                    offspring = self.nearest_neighbor()
                else:
                    parent1, parent2 = random.sample(
                        population[:elite_size], 2)
                    offspring = self.mutate(self.crossover(parent1, parent2))
                next_generation.append(offspring)
            population = next_generation
        best_tour = min(population, key=lambda tour: self.total_distance(tour))
        best_distance = self.total_distance(best_tour)
        return best_tour, best_distance

    def crossover(self, parent1, parent2):
        start = random.randint(0, self.num_nodes - 1)
        end = random.randint(start + 1, self.num_nodes)
        offspring = parent1[start:end]
        for city in parent2:
            if city not in offspring:
                offspring.append(city)
        return offspring

    def mutate(self, tour):
        i, j = sorted(random.sample(range(self.num_nodes), 2))
        tour[i:j + 1] = reversed(tour[i:j + 1])
        return tour


def theory_best(file):
    with open('tsplib95/tsp_best_solutions.txt', 'r') as fileT:
        for line in fileT:
            name, distance = line.strip().split(' : ')
            if name == file:
                return int(distance)


def perc(file, ans):
    problem_name = file[:-4]
    known_best = theory_best(problem_name)
    difference = ((ans - known_best) / known_best) * 100
    return round(difference, 2)


f = "tsplib95/ulysses22.tsp"

problem = tsplib95.load(f)

G = problem.get_graph()
n = G.number_of_nodes()

import csv

def scrivi_su_csv(numero_nodi, mutation, iteration, generation, risultato, tempo, nome_file):
    with open(nome_file, 'a', newline='') as file_csv:  # 'a' per appendere al file
        writer = csv.writer(file_csv)
        print (risultato)
        writer.writerow(
            [numero_nodi, mutation, iteration, generation, risultato, tempo])

tsp_solver = TSPGeneticAlgorithm(G)
risultati = []

inizio = time.time()
tsp_solution, tsp_distance = tsp_solver.genetic_algorithm(
    population_size=50, elite_size=20, mutation_rate=0.001, generations=8000)
print("soluzione trovata: ", perc(f[9:], tsp_distance), "% in ", round(time.time() - inizio, 2), " secondi.")