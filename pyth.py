import heapq, copy, collections, random, time
from sys import maxsize
from itertools import permutations, combinations

class Node:
    def __init__(self, pos, ter):
        self.pos = pos          # position
        self.terrain = ter          # terrain
        self.parent = -1          # node position from which we got to this node (parent rename!)
        self.dist = maxsize     # distance from start to current node
        self.g = maxsize        # distance for A*
        self.h = maxsize

    def __lt__(self, other):
        if self.dist != other.dist:
            return self.dist < other.dist
        return self.h < other.h

def dijkstra(data, start, adjacency):
    h = []
    data[start].dist = 0
    heapq.heappush(h, data[start])

    for i, _ in enumerate(data):
        node = heapq.heappop(h)
        for adjacent in adjacency:
            neighbor = (node.pos[0]+adjacent[0], node.pos[1]+adjacent[1])
            if neighbor in data and data[neighbor].dist > node.dist + data[neighbor].terrain:
                data[neighbor].dist = node.dist + data[neighbor].terrain
                data[neighbor].parent = node.pos
                heapq.heappush(h, data[neighbor])

    return data

def aStar(data, start, end, adjacency):
    openL = []
    closedL = []
    data[start].g = 0
    heapq.heappush(openL, data[start])

    for _ in data:      # until all nodes have been discovered
        node = heapq.heappop(openL)
        closedL.append(node.pos)

        if node.pos == end:
            break

        for adjacent in adjacency:
            neighbor = (node.pos[0]+adjacent[0], node.pos[1]+adjacent[1])
            if neighbor in data and neighbor not in closedL:        # also TRAVERSABLE (can do later)
                h = abs(data[neighbor].pos[0] - end[0]) + abs(data[neighbor].pos[1] - end[1])
                g = node.g + data[neighbor].terrain
                f = g + h
                if f < data[neighbor].dist:
                    data[neighbor].g = g
                    data[neighbor].h = h
                    data[neighbor].dist = f
                    data[neighbor].parent = node.pos
                if data[neighbor] not in openL:
                    heapq.heappush(openL, data[neighbor])

    return data
    

def load(file):
    with open(file) as f:
        ROWS, COLS, moveType = f.readline().split()[:3]
        map2D = [f.readline().rstrip('\n') for line in range(int(ROWS))]
        adjacency = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if moveType == 'D':
            adjacency.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

        # if maxRow*maxCol != len(mapStr):    # az na konci!
            # S - start (required)
            # N - princesses (required)
            # D - one dragon (required)
            # E - end (optional)
            # P - portal (optional)
        #    print("Incorrect number of map characters")
        #    return

    mapTerr = {'N': 100, 'H': 2}
    mapTerr = collections.defaultdict(lambda: 1, mapTerr)
    princesses = []
    mapData = {}
    start = 0, 0

    for i in range(int(ROWS)):
        for j in range(int(COLS)):
            if map2D[i][j] == 'D':
                dragon = i, j
            elif map2D[i][j] == 'P':
                princesses.append((i, j))
            elif map2D[i][j]  == 'S':
                start = i, j
            mapData[i, j] = Node((i, j), mapTerr[map2D[i][j]])
    
    return mapData, princesses, dragon, start, adjacency

def findMinDist(npcData, princesses, dragon, start):
    mini = maxsize

    for permutation in permutations(princesses):
        distance = npcData[start][dragon].dist     # distance to get to dragon
        for begin, finish in zip((dragon,)+permutation, permutation):
            distance += npcData[begin][finish].dist
        if distance < mini:
            mini, order = distance, permutation

    return (start, dragon) + order, mini

def karp(npcData, princesses, dragon, start):
    princesses = frozenset(princesses)
    nodes = {}

    for row in range(len(princesses)):              # set length
        for comb in combinations(princesses, row):  # set value     (right side)
            comb = frozenset(comb)
            for finish in princesses - comb:        # destination   (left side)
                routes = []
                if comb == frozenset():             # case for dragon starting
                    cost = npcData[dragon][finish].dist + npcData[start][dragon].dist
                    nodes[finish, frozenset()] = cost, dragon
                else:
                    for begin in comb:              # it always is a single value from the set
                        # when we have begin, we need to find combo of set length - 1 in which the begin isnt there
                        subcomb = comb - frozenset({begin})     # this is how we get previous level combo we needed
                        prevCost = nodes[begin, subcomb][0]
                        cost = npcData[begin][finish].dist + prevCost
                        routes.append((cost, begin))
                    nodes[finish, comb] = min(routes)

    com = []
    for i, node in enumerate(reversed(dict(nodes))):
        if i < len(princesses):
            com.append((nodes.pop(node), node[0]))
        elif i == len(princesses):
            val, step = min(com)
            princesses -= {step}
            path = [step]
            cost, nextStep = val
            break
    
    for _ in range(len(princesses)):
        path.append(nextStep)
        princesses -= {nextStep}
        nextStep = nodes[nextStep, princesses][1]
    path.extend([dragon, start])

    return path[::-1], cost

def getPath(npcData, order):
    path = []

    for begin, finish in zip(order, order[1:]):
        path2 = []
        while finish != begin:
            path2.append(finish)
            finish = npcData[begin][finish].parent
        path.append(path2[::-1])

    return path

def printSolution(path):
    for i, road in enumerate(path, 1):
        print(f"\n{i}: ", end=' ')
        for step in road:
            print(step, end=' ')
    print()


def main():

    mapData, princesses, dragon, start, adjacency = load("mapa8.txt")
    '''
    npcData = {p: dijkstra(copy.deepcopy(mapData), p, adjacency) for p in princesses}
    npcData.update({start: aStar(copy.deepcopy(mapData), start, dragon, adjacency)}) # can use A* for different purposes tho
    npcData.update({dragon: dijkstra(copy.deepcopy(mapData), dragon, adjacency)})

    print("Starting permutations")
    order, dist = findMinDist(npcData, princesses, dragon, start)
    path = getPath(npcData, order)
    printSolution(path)
    print("Cost: " + str(dist))

    print("Starting Kerp")
    order2, dist2 = karp(npcData, princesses, dragon, start)
    path2 = getPath(npcData, order2)
    printSolution(path2)
    print("Cost: " + str(dist2))

    ### ZEN GARDEN ###
    a=2
    '''
    with open("mapa8.txt") as f:

        ROWS, COLS = map(int, f.readline().split()[:2])
        TO_RAKE = COLS * ROWS - sum([string.count('N') for string in f])
        HALF_PERIMETER = ROWS + COLS

        GENES = HALF_PERIMETER * 2  # gene - an intruction 
        CHROMOSOMES = 10            # chromosome - solution that is defined by order of genes
        GENERATIONS = 100           # generation - set of all chromozomes

        MIN_MUT_RATE = 0.05
        MAX_MUT_RATE = 0.80
        CROSS_RATE = 0.90

        startTime = time.time()
        generationTimes = []

        # generating chromozomes for one population/generation
        population = []
        genes = random.sample(range(1, GENES), GENES-1)
        for _ in range(CHROMOSOMES):
            random.shuffle(genes)
            chromosome = [num * random.choice([-1, 1]) for num in genes]
            population.append(chromosome)

        # loop of generations
        mutRate = MIN_MUT_RATE
        for generation in range(GENERATIONS):
            genTime = time.time()
            
            # evaluate all chromosomes and find the best one
            fitness, fMax, iMax = [], 0, 0
            for chromosome in range(CHROMOSOMES):
                raked = rakeGarden(population[chromosome], copy.deepcopy(mapData), False, ROWS, COLS, HALF_PERIMETER, TO_RAKE) # should be without map arg..
                fitness.append(raked)
                if raked > fMax:
                    population[iMax]
                    iMax, fMax = chromosome, raked

            # ToDo: do this in the end
            print(f"Generation: {generation}, Max raked: {fMax} (out of {TO_RAKE}), Mutation rate: {mutRate}")
            if fMax == TO_RAKE:
                total = round(time.time() - startTime, 2)
                avg = round(sum(generationTimes) / len(generationTimes), 2)
                chromo = " ".join(map(str, population[iMax]))
                print(f"Found a solution in {total}s time, each generation took {avg}s in average.")
                print(f"Chromosome: {chromo}")
                rakeGarden(population[iMax], copy.deepcopy(mapData), True, ROWS, COLS, HALF_PERIMETER, TO_RAKE)
                break
            
            # increasing mutation each generation change to prevent local maximums
            mutRate = mutRate if mutRate >= MAX_MUT_RATE else mutRate + 0.01

            # loop for creating next generation, 1 iteration for 2 populations that we mutate
            children = []
            for i in range(0, CHROMOSOMES, 2):

                # pick 2 better chromosomes out of 4
                pick = random.sample(range(CHROMOSOMES), 4)
                better1 = pick[0] if fitness[pick[0]] > fitness[pick[1]] else pick[1]
                better2 = pick[2] if fitness[pick[2]] > fitness[pick[3]] else pick[3]

                # copying better genes to 2 child chromosomes
                children.extend([[],[]])
                for j in range(GENES-1):
                    children[i].append(population[better1][j])
                    children[i+1].append(population[better2][j])

                # mutating 2 chromosomes with uniform crossover (both inherit the same amount of genetic info)
                if random.random() < CROSS_RATE:
                    for c in range(2):
                        for g in range(GENES-1):
                            if random.random() < mutRate:

                                # search for gene with mutNum number
                                mutNum = random.randint(1, GENES) * random.choice([-1, 1])
                                f = 0
                                for k, gene in enumerate(children[i+c]):
                                    if gene == mutNum:
                                        f = k

                                # if found, swap it with g gene, if not, replace g with it
                                if f:
                                    tmp = children[i+c][g]
                                    children[i+c][g] = children[i+c][f]
                                    children[i+c][f] = tmp
                                else:
                                    children[i+c][g] = mutNum

            # keep the best chromosome for next generation
            for i in range(GENES-1):
                children[0][i] = population[iMax][i]

            population = children

            generationTimes.append(time.time() - genTime)

def printMap(mapData, COLS):
    for i, aa in enumerate(mapData):
        if i % COLS == 0:
            print()
        if mapData[aa].dist == maxsize:
            saj = 99
            if mapData[aa].terrain == 100:
                saj = 0
        else:
            saj = mapData[aa].dist

        print("{0:2}".format(saj), end=' ')
    print("\n")


def rakeGarden(chromozome, mapData, printConsole, ROWS, COLS, HALF_PERIMETER, TO_RAKE):
    order = 1         # raking order and terrain
    for gene in chromozome:

        # get starting position and movement direction
        posNum = abs(gene)
        if posNum <= COLS:                         # go DOWN    <0, 20>        # 20r, 10c
            pos, move = (0, posNum-1), (1, 0)
        elif posNum <= HALF_PERIMETER:             # go RIGHT   (20, 30>
            pos, move = (posNum-COLS-1, 0), (0, 1)
        elif posNum <= HALF_PERIMETER + ROWS:      # go LEFT    (30, 40>
            pos, move = (posNum-HALF_PERIMETER-1, COLS-1), (0, -1)
        else:                                      # go UP      <40, 60)
            pos, move = (ROWS-1, posNum-HALF_PERIMETER-ROWS-1), (-1, 0)
        
        # checking whether we can enter the garden with current pos
        if mapData[pos].terrain != 100 and mapData[pos].dist == maxsize:
            parent = -1

            # move until we reach end of the map 
            while pos in mapData:
                mapData[pos].parent = parent

                # if collision to raked sand
                if mapData[pos].terrain == 100 or mapData[pos].dist != maxsize:
                    prevPos = mapData[pos].parent
                    
                    # changing direction
                    if move[0] != 0:    # Y -> X
                        right = prevPos[0], prevPos[1] + 1
                        left = prevPos[0], prevPos[1] - 1
                        if right in mapData and (mapData[right].dist == maxsize and mapData[right].terrain != 100) and \
                            left in mapData and (mapData[left].dist == maxsize and mapData[left].terrain != 100):
                            move = (0, 1) if gene > 0 else (0, -1)
                        elif right in mapData and (mapData[right].dist == maxsize and mapData[right].terrain != 100):
                            move = 0, 1
                        elif left in mapData and (mapData[left].dist == maxsize and mapData[left].terrain != 100):
                            move = 0, -1
                        else:
                            if right not in mapData or left not in mapData:
                                break
                            else:
                                move = False
                    else:               # X -> Y
                        down = prevPos[0] + 1, prevPos[1]
                        up = prevPos[0] - 1, prevPos[1]
                        if up in mapData and (mapData[up].dist == maxsize and mapData[up].terrain != 100) and \
                            down in mapData and (mapData[down].dist == maxsize and mapData[down].terrain != 100):
                            move = (1, 0) if gene > 0 else (-1, 0)
                        elif down in mapData and (mapData[down].dist == maxsize and mapData[down].terrain != 100):
                            move = 1, 0
                        elif up in mapData and (mapData[up].dist == maxsize and mapData[up].terrain != 100):
                            move = -1, 0      # -1, 0
                        else:
                            if up not in mapData or down not in mapData:
                                break
                            else:
                                move = False
                    
                    # if we cant change direction, remove the path
                    if not move:
                        cancelTransitNum = 1
                        while mapData[prevPos].parent != -1:
                            mapData[prevPos].dist = maxsize
                            pare = mapData[prevPos].parent
                            mapData[prevPos].parent = -1
                            prevPos = pare
                        mapData[prevPos].dist = maxsize
                        mapData[prevPos].parent = -1
                        break
                    pos = prevPos[0] + move[0], prevPos[1] + move[1]
                    mapData[pos].parent = prevPos

                cancelTransitNum = 0
                mapData[pos].dist = order
                parent = pos
                pos = pos[0]+move[0], pos[1]+move[1]

            if cancelTransitNum == 0:
                order += 1

    if printConsole:
        printMap(mapData, COLS)

    unraked = 0
    for i in range(ROWS):
        for j in range(COLS):
            if mapData[i, j].dist == maxsize and mapData[i, j].terrain != 100:
                unraked += 1

    return TO_RAKE - unraked

main()

#objs:
# terrain -1 (rock), 0 is unraked, >0 is raked
# create evolution as a map creator: create file with the map / or just show up hows it done
# integrate it with path finding algs (two ways: mountain, basic)
# create class for a map and make functions standalone

# apply rules somehow and make simulation (would be better without resources, just moves)
# race of 3 playres in random positions