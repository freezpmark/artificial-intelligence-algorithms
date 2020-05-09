import heapq, copy, collections
from sys import maxsize
from itertools import permutations, combinations
from operator import add

class Node:
    def __init__(self, pos, ter):
        self.pos = pos          # position
        self.ter = ter          # terrain
        self.path = -1          # node position from which we got to this node (parent rename!)
        self.dist = maxsize     # distance from start to current node
        self.f = maxsize

    def __lt__(self, other):
        return self.dist < other.dist

def dijkstra(data, start, adjacency):
    h = []
    data[start].dist = 0
    heapq.heappush(h, data[start])

    for i, _ in enumerate(data):
        node = heapq.heappop(h)
        for adjacent in adjacency:
            neighbor = (node.pos[0]+adjacent[0], node.pos[1]+adjacent[1])
            if neighbor in data and data[neighbor].dist > node.dist + data[neighbor].ter:
                data[neighbor].dist = node.dist + data[neighbor].ter
                data[neighbor].path = node.pos
                heapq.heappush(h, data[neighbor])

    return data

def aStar(data, start, end, adjacency):
    openL = []
    closedL = []
    data[start].dist = 0
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
                g = node.dist + data[neighbor].ter
                f = g + h
                if f < data[neighbor].f:
                    data[neighbor].f = f
                    data[neighbor].dist = g
                    data[neighbor].path = node.pos
                if neighbor not in openL:
                    heapq.heappush(openL, data[neighbor])

    return data
    

def load(file):
    with open(file) as f:
        # maxRow, maxCol = map(int, f.readline().split()[:2])
        maxRow, maxCol, moveType = f.readline().split()[:3]
        map2D = [f.readline().rstrip('\n') for line in range(int(maxRow))]
        adjacency = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if moveType == 'D':
            adjacency.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

        #mapStr = f.readline()
        # if maxRow*maxCol != len(mapStr):    # az na konci!
            # S - start (required)
            # N - princesses (required)
            # D - one dragon (required)
            # E - end (optional)
            # D - dragon (optional)
            # P - portal (optional)
        #    print("Incorrect number of map characters")
        #    return

    mapTerr = {'N': 100, 'H': 2}
    mapTerr = collections.defaultdict(lambda: 1, mapTerr)
    princesses = []
    mapData = {}
    start = 0, 0

    for i in range(int(maxRow)):
        for j in range(int(maxCol)):
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
            finish = npcData[begin][finish].path
        path.append(path2[::-1])

    return path

def printSolution(path):
    for i, road in enumerate(path, 1):
        print(f"\n{i}: ", end=' ')
        for step in road:
            print(step, end=' ')
    print()


def main():
    mapData, princesses, dragon, start, adjacency = load("mapa4.txt")

    npcData = {p: dijkstra(copy.deepcopy(mapData), p, adjacency) for p in princesses}
    npcData.update({start: aStar(copy.deepcopy(mapData), start, dragon, adjacency)})
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
    
main()
