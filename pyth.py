import heapq
import copy
import random
import time
import re
from sys import maxsize
from itertools import permutations, combinations


class PositionError(Exception):
    pass


class Map:
    def __init__(self, file_name):
        self.file_name = None
        self.nodes = None
        self.entities = None
        self.__width = None
        self.__height = None

        self.__loadMap(file_name)

    def __loadMap(self, file_name):
        entities = {'papers': [], 'base': None, 'start': None}
        nodes = {}
        with open(file_name + '.txt') as f:
            for i, line in enumerate(f):
                for j, col in enumerate(line.split()):
                    if col[0] == '(':
                        entities['papers'].append((i, j))
                        col = col[1:-1]
                    elif col[0] == '[':
                        entities['base'] = (i, j)
                        col = col[1:-1]
                    elif col[0] == '{':
                        entities['start'] = (i, j)
                        col = col[1:-1]
                    nodes[i, j] = Node((i, j), int(col))

        if all(entities.values()) and len(entities['papers']) > 1:
            self.file_name = file_name
            self.nodes = nodes
            self.entities = entities
            self.__height = i+1
            self.__width = j+1

    def __getitem__(self, pos):
        assert len(pos) == 2, "Coordinate must have two values."
        if not (0 <= pos[0] < self.height) or \
           not (0 <= pos[1] < self.width):
            raise PositionError(str(pos))
        return self.nodes[pos]  # self.nodes.get(pos)

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height


class Node:
    def __init__(self, pos, terrain):
        self.pos = pos
        self.terrain = terrain
        self.parent = -1        # node position from which we got to this node
        self.dist = maxsize     # distance from starting node to current node
        self.g = maxsize
        self.h = maxsize

    def __lt__(self, other):
        if self.dist != other.dist:
            return self.dist < other.dist
        return self.h < other.h


def dijkstra(data, start, move_type):
    heap = []
    data[start].dist = 0
    heapq.heappush(heap, data[start])

    while heap:
        node = heapq.heappop(heap)
        for adjacent in move_type:
            neighbor = (node.pos[0]+adjacent[0], node.pos[1]+adjacent[1])

            # avoid out of bounds or walls
            if not 0 <= neighbor[0] < data.height or \
               not 0 <= neighbor[1] < data.width or \
               data[neighbor].terrain == -1:
                continue

            if data[neighbor].dist > node.dist + data[neighbor].terrain:
                data[neighbor].dist = node.dist + data[neighbor].terrain
                data[neighbor].parent = node.pos
                heapq.heappush(heap, data[neighbor])

    return data


def aStar(data, start, end, move_type):
    open_list = []
    close_list = []
    data[start].g = 0
    heapq.heappush(open_list, data[start])

    while open_list:
        node = heapq.heappop(open_list)
        close_list.append(node.pos)

        if node.pos == end:
            break

        for adjacent in move_type:
            neighbor = (node.pos[0]+adjacent[0], node.pos[1]+adjacent[1])

            # avoid out of bounds or walls
            if not 0 <= neighbor[0] < data.height or \
               not 0 <= neighbor[1] < data.width or \
               data[neighbor].terrain == -1:
                continue
            if neighbor not in close_list:
                x = abs(data[neighbor].pos[0] - end[0])
                y = abs(data[neighbor].pos[1] - end[1])
                h = x + y
                g = node.g + data[neighbor].terrain
                f = g + h
                if f < data[neighbor].dist:
                    data[neighbor].g = g
                    data[neighbor].h = h
                    data[neighbor].dist = f
                    data[neighbor].parent = node.pos
                if data[neighbor] not in open_list:
                    heapq.heappush(open_list, data[neighbor])

    return data


def findMinDist(npc_data, map_data):
    # ! danger of dependency on the order of entities
    papers, base, start = map_data.entities.values()

    mini = maxsize

    for permutation in permutations(papers):
        distance = npc_data[start][base].dist
        for begin, finish in zip((base,)+permutation, permutation):
            distance += npc_data[begin][finish].dist
        if distance < mini:
            mini, order = distance, permutation

    return (start, base) + order, mini


def karp(npc_data, map_data):
    papers, base, start = map_data.entities.values()
    papers = frozenset(papers)
    nodes = {}

    for row in range(len(papers)):
        for comb in combinations(papers, row):
            comb = frozenset(comb)
            for dest in papers - comb:
                routes = []
                if comb == frozenset():         # case for base starting
                    cost = npc_data[base][dest].dist + \
                           npc_data[start][base].dist
                    nodes[dest, frozenset()] = cost, base
                else:
                    for begin in comb:          # single val from set
                        sub_comb = comb - frozenset({begin})
                        prev_cost = nodes[begin, sub_comb][0]
                        cost = npc_data[begin][dest].dist + prev_cost
                        routes.append((cost, begin))
                    nodes[dest, comb] = min(routes)

    com = []
    for i, node in enumerate(reversed(dict(nodes))):
        if i < len(papers):
            com.append((nodes.pop(node), node[0]))
        elif i == len(papers):
            val, step = min(com)
            papers -= {step}
            path = [step]
            cost, next_step = val
            break

    for _ in papers:
        path.append(next_step)
        papers -= {next_step}
        next_step = nodes[next_step, papers][1]
    path.extend([base, start])

    return path[::-1], cost


def getPaths(npc_data, order):
    # ToDo
    """Gets paths between entities by ...

    Arguments:
        npc_data {dict} -- describes shortest paths between all entities
                with keys being tuple coordinates. Access via dict[start][dest]
        order {tuple} --

    Returns:
        list -- each value is a list of tuples with ordered coordinates
    """
    paths = []

    for begin, finish in zip(order, order[1:]):
        path = []
        while finish != begin:
            path.append(finish)
            finish = npc_data[begin][finish].parent
        paths.append(path[::-1])

    return paths


def printSolution(paths):
    """Prints the order of paths between entities. Each line starts with
    order number followed by order of tuple coordinates that represent
    the movement progression from start to destination entity.

    Arguments:
        path {list} -- each value is a list of tuples with ordered coordinates
    """
    for i, path in enumerate(paths, 1):
        print(f"\n{i}: ", end=' ')
        for step in path:
            print(step, end=' ')
    print()


def getMoves(query):
    """Gets moving possibilities. Default moving type is Manhattan,
    if queries first character is 'D', it will be extended by Diagonal moves

    Arguments:
        query {string} -- string used for selection via first character

    Returns:
        list -- list of tuples that represent moving options
    """
    moveType = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if query[0] == 'D':
        moveType.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    return moveType


def findPaths(map_data, move_type):
    """Find all the shortest possible paths between all entities
    papers, base, start in the map using Dijkstra and A* algorithms.

    Arguments:
        map_data {Map} -- describes the whole map and its entities
        move_type {list} -- list of tuples that represent moving options

    Returns:
        dict -- dictionary that describes shortest paths between all entities
                with keys being tuple coordinates. Access via dict[start][dest]
    """
    papers, base, start = map_data.entities.values()

    ent_data = {p: dijkstra(
        copy.deepcopy(map_data), p, move_type) for p in papers}
    ent_data.update({start: aStar(
        copy.deepcopy(map_data), start, base, move_type)})
    ent_data.update({base: dijkstra(
        copy.deepcopy(map_data), base, move_type)})

    return ent_data


def main():
    # DECLARATION: Load or create walls
    # query = "10x12 (1,5) (2,1) (3,4) (4,2) (6,8) (6,9)"
    # query = "10x10 (12,2) (0,0)"
    query = "mapa10.txt"

    map_walls = loadWalls(query)
    if not map_walls:
        print("Invalid query!")
        return

    # INITIALIZATION: Add terrain
    map_terrained = None
    attempts = 4

    map_terrained = evolutionize(map_walls, attempts, True)
    if not map_terrained:
        print("Evolution algorithm could not find a solution, try again.")
        return

    # ? create checks between each step
    # INITIALIZATION: Add random npcs
    entities = generateEntities(map_terrained, 2)
    saveMap(map_terrained, 'yosh', entities)

    map_data = Map('yosh')
    if not map_data:
        print("Invalid map!")
        return

    query2 = 'M'            # query3 = 'S'
    move_type = getMoves(query2)

    npc_data = findPaths(map_data, move_type)

    print("Starting permutations")
    order, dist = findMinDist(npc_data, map_data)
    path = getPaths(npc_data, order)
    printSolution(path)
    print("Cost: " + str(dist))

    print("Starting Kerp")
    order2, dist2 = karp(npc_data, map_data)
    path2 = getPaths(npc_data, order2)
    printSolution(path2)
    print("Cost: " + str(dist2))


def entityGenerator(map_terrained):
    """Generator of positions that are used for entities.

    Arguments:
        map_terrained {list} -- 2D map list of terrained map

    Yields:
        tuple -- (x, y) coordinate of generated position
    """
    reserved = set()
    for i, row in enumerate(map_terrained):
        for j, x in enumerate(row):
            if x == -1:
                reserved.add((i, j))

    while True:
        x = random.randint(0, len(map_terrained) - 1)
        y = random.randint(0, len(map_terrained[0]) - 1)
        if (x, y) not in reserved:
            reserved.add((x, y))
            yield (x, y)


def generateEntities(map_terrained, amount):
    """Generates position for papers, base and starting location.

    Arguments:
        map_terrained {list} -- 2D map list of terrained map
        amount {int} -- amount of papers we wish to generate

    Returns:
        dict -- has 3 entity keys as names with coordinate values
    """
    gen = entityGenerator(map_terrained)

    papers = [next(gen) for _ in range(amount)]
    base = next(gen)
    start = next(gen)

    return {'papers': papers, 'base': base, 'start': start}


def saveMap(map_terrained, file_name, entities):
    """Saves terrained map into text file named file_name
    with entities being surrounded with certain brackets.

    Arguments:
        map_terrained {list} -- 2D map list of terrained map
        file_name {str} -- name of the text file we're saving the map into
        entities {dict} -- has 3 entity keys as names with coordinate values
    """
    with open(file_name + '.txt', 'w') as f:
        for i, row in enumerate(map_terrained):
            for j in range(len(row)):
                string = str(map_terrained[i][j])
                if (i, j) in entities['papers']:
                    string = '(' + string + ')'
                elif (i, j) == entities['base']:
                    string = '[' + string + ']'
                elif (i, j) == entities['start']:
                    string = '{' + string + '}'
                f.write("{:^5}".format(string))
            f.write('\n')


def loadWalls(query):
    """ Loads or creates a map of walls that is represented by 2D list.
    Query can consist of either a file name, or command that creates a new map.
    These two examples will have the same outcome:

    query -- "file_name.txt" < "001\\n000\\n010"\n
    query -- "3x3 (0,2) (2,1)"

    Arguments:
        query {string} -- file name to load, or command that creates a new map

    Returns:
        list -- 2D map list with walls that are represented by number 1
    """
    map_walls = []

    # Load from string
    if re.search(r'[0-9]+x[0-9]+(\ \([0-9]+,[0-9]+\))+$', query):
        query = query.split()
        ROWS, COLS = map(int, query[0].split('x'))
        walls = {eval(coordinate) for coordinate in query[1:]}
        map_walls = [[0] * COLS for _ in range(ROWS)]

        for wall in walls:
            try:
                map_walls[wall[0]][wall[1]] = 1
            except IndexError:
                map_walls = None

    # Load from file
    elif re.search(r'\.txt', query):
        with open(query) as f:
            line = f.readline().rstrip()
            map_walls.append([int(column) for column in line])
            prev_length = len(line)

            for line in f:
                line = line.rstrip()
                if prev_length != len(line):
                    map_walls = None
                    break
                prev_length = len(line)
                map_walls.append([int(column) for column in line])

    return map_walls


def listToTuple(map_list):
    """Converts a 2D map list to dictionary of tuples as keys with x, y coors
    """
    map_tuple = {}
    for i, row in enumerate(map_list):
        for j in range(len(row)):
            map_tuple[i, j] = map_list[i][j]

    return map_tuple


def evolutionize(map_list, attempts, print_stats):
    """Runs evolutionary algorithm for map with walls to fill it with terrain.

    Arguments:
        map_list {list} -- 2D map list of walls
        attempts {int} -- number of times this algorithm will run
        print_stats {boolean} -- turns on debug mode that prints the solution

    Returns:
        list -- 2D map list that is filled by the most amount of terrain
    """
    found_solution = False
    attempt_number = 1
    while not found_solution and attempt_number < attempts:
        map_tuple = listToTuple(map_list)
        map_tuple = {key: -map_tuple[key] for key in map_tuple.keys()}

        SHAPE = len(map_list), len(map_list[0])
        ROCKS = sum(val != 0 for val in map_tuple.values())
        TO_RAKE = SHAPE[0] * SHAPE[1] - ROCKS
        GENES = (SHAPE[0] + SHAPE[1]) * 2    # gene - an intruction (PERIMETER)
        CHROMOSOMES = 30               # chromosome - solution defined by genes
        GENERATIONS = 100              # generation - set of all chromosomes

        MIN_MUT_RATE = 0.05
        MAX_MUT_RATE = 0.80
        CROSS_RATE = 0.90

        start_time = time.time()
        gen_times = []
        prev_max = 0

        # generating chromosomes for one population/generation
        population = []
        genes = random.sample(range(1, GENES), GENES-1)
        for _ in range(CHROMOSOMES):
            random.shuffle(genes)
            chromosome = [num * random.choice([-1, 1]) for num in genes]
            population.append(chromosome)

        # loop of generations
        mut_rate = MIN_MUT_RATE
        for i in range(GENERATIONS):
            generation_time = time.time()

            # evaluate all chromosomes and save the best one
            fit, fit_max, j_max = [], 0, 0
            for j in range(CHROMOSOMES):
                unraked, fills = rakeMap(
                    population[j], copy.copy(map_tuple), SHAPE)
                raked = TO_RAKE - unraked
                fit.append(raked)
                if raked > fit_max:
                    j_max, fit_max, map_filled = j, raked, fills

            if prev_max < fit_max:
                print(f"Generation: {i+1},", end="\t")
                print(f"Max raked: {fit_max} (out of {TO_RAKE})", end="\t")
                print(f"Mutation rate: {round(mut_rate, 2)}")
            if fit_max == TO_RAKE:
                found_solution = True
                gen_times.append(time.time() - generation_time)
                break

            # increase mutation rate each generation to prevent local maximums
            mut_rate = mut_rate if mut_rate >= MAX_MUT_RATE else mut_rate + .01

            # next generation creating, 1 iteration for 2 populations
            children = []
            for i in range(0, CHROMOSOMES, 2):

                # pick 2 better chromosomes out of 4
                pick = random.sample(range(CHROMOSOMES), 4)
                better1 = pick[0] if fit[pick[0]] > fit[pick[1]] else pick[1]
                better2 = pick[2] if fit[pick[2]] > fit[pick[3]] else pick[3]

                # copying better genes to 2 child chromosomes
                children.extend([[], []])
                for j in range(GENES-1):
                    children[i].append(population[better1][j])
                    children[i+1].append(population[better2][j])

                # mutating 2 chromosomes with uniform crossover
                # (both inherit the same amount of genetic info)
                if random.random() < CROSS_RATE:
                    for c in range(2):
                        for g in range(GENES-1):
                            if random.random() < mut_rate:

                                # search for gene with mut_num number
                                mut_num = random.randint(1, GENES)
                                mut_num *= random.choice([-1, 1])
                                f = 0
                                for k, gene in enumerate(children[i+c]):
                                    if gene == mut_num:
                                        f = k

                                # swap it with g gene, else replace g with it
                                if f:
                                    tmp = children[i+c][g]
                                    children[i+c][g] = children[i+c][f]
                                    children[i+c][f] = tmp
                                else:
                                    children[i+c][g] = mut_num

            # keep the best chromosome for next generation
            for i in range(GENES-1):
                children[0][i] = population[j_max][i]

            population = children

            prev_max = fit_max
            gen_times.append(time.time() - generation_time)

        # printing stats, solution and map
        if print_stats:
            total = round(time.time() - start_time, 2)
            avg = round(sum(gen_times) / len(gen_times), 2)
            chromo = " ".join(map(str, population[j_max]))
            print("{} a solution!".format(
                "Found" if found_solution else "Couldn't find"))
            print(f"Total time elapsed is {total}s, \
                each generation took {avg}s in average.")
            print(f"Chromosome: {chromo}")

            for row in map_filled:
                for col in row:
                    print("{0:2}".format(col), end=' ')
                print()
            attempt_number += 1
            if not found_solution and attempt_number != attempts:
                print(f"\nAttempt number {attempt_number}.")

    return map_filled if found_solution else []


def rakeMap(chromosome, map_tuple, SHAPE):
    """Attempts to fill the map terrain with chromosome that is defined
    by the order of instructions known as genes.

    Arguments:
        chromosome {list} -- collection of genes (instructions)
        map_tuple {dict} -- map defined by tuples of x, y coordinates as keys
        SHAPE {list} -- consist of height and width length values

    Returns:
        int -- amount of unraked (unfilled) positions
        list -- 2D map list that is filled with terrain
    """
    ROWS, COLS = SHAPE[0], SHAPE[1]
    HALF_PERIMETER = SHAPE[0] + SHAPE[1]
    UNRAKED = 0

    order = 1
    for gene in chromosome:

        # get starting position and movement direction
        pos_num = abs(gene)
        if pos_num <= COLS:                         # go DOWN
            pos, move = (0, pos_num-1), (1, 0)
        elif pos_num <= HALF_PERIMETER:             # go RIGHT
            pos, move = (pos_num-COLS-1, 0), (0, 1)
        elif pos_num <= HALF_PERIMETER + ROWS:      # go LEFT
            pos, move = (pos_num-HALF_PERIMETER-1, COLS-1), (0, -1)
        else:                                       # go UP
            pos, move = (ROWS-1, pos_num-HALF_PERIMETER-ROWS-1), (-1, 0)

        # checking whether we can enter the garden with current pos
        if map_tuple[pos] == UNRAKED:
            parents = {}
            parent = 0

            # move until we reach end of the map
            while 0 <= pos[0] < ROWS and 0 <= pos[1] < COLS:

                # collision to raked sand/rock
                if map_tuple[pos] != UNRAKED:
                    pos = parent            # get previous pos
                    parent = parents[pos]   # get previous parent

                    # change moving direction
                    if move[0] != 0:    # Y -> X
                        R = pos[0], pos[1]+1
                        L = pos[0], pos[1]-1
                        R_inbound = R[1] < COLS
                        L_inbound = L[1] >= 0
                        R = R_inbound and map_tuple[R] == UNRAKED
                        L = L_inbound and map_tuple[L] == UNRAKED

                        if R and L:
                            move = (0, 1) if gene > 0 else (0, -1)
                        elif R:
                            move = 0, 1
                        elif L:
                            move = 0, -1
                        elif R_inbound and L_inbound:
                            move = False
                        else:
                            break   # reached end of the map so we can leave

                    else:               # X -> Y
                        D = pos[0]+1, pos[1]
                        U = pos[0]-1, pos[1]
                        D_inbound = D[0] < ROWS
                        U_inbound = U[0] >= 0
                        D = D_inbound and map_tuple[D] == UNRAKED
                        U = U_inbound and map_tuple[U] == UNRAKED

                        if D and U:
                            move = (1, 0) if gene > 0 else (-1, 0)
                        elif D:
                            move = 1, 0
                        elif U:
                            move = -1, 0
                        elif D_inbound and U_inbound:
                            move = False
                        else:
                            break

                    # if we cant change direction, remove the path
                    if not move:
                        order -= 1
                        while parents[pos] != 0:
                            map_tuple[pos] = 0
                            pos = parents[pos]
                        map_tuple[pos] = 0
                        break

                map_tuple[pos] = order
                parents[pos] = parent
                parent = pos
                pos = pos[0]+move[0], pos[1]+move[1]

            order += 1

    fills = []
    unraked = 0
    j = -1
    for i, fill in enumerate(map_tuple.values()):
        if fill == UNRAKED:
            unraked += 1
        if i % COLS == 0:
            j += 1
            fills.append([])
        fills[j].append(fill)

    return unraked, fills


main()

# split into more files (map creating, finding solution)

# ToDo: Add function annotations
# ToDo: Finish docstrings, current ones need corrections
# ToDo: Held Karr (Add Shortest subset combo)
# ToDo: Pathfinding (C: Climbing, S: Swamp)
# ToDo: Validation checks (__loadMap, coordinate checkings for example)
# ToDo: Create tests
# ToDo: Add Rule based system in the end (each paper is one fact)
