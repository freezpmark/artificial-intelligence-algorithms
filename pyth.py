import heapq, copy, collections
from sys import maxsize
from itertools import permutations, combinations

class Node:
    def __init__(self, pos, ter):
        self.pos = pos          # position
        self.ter = ter          # terrain
        self.path = -1          # node position from which we got to this node
        self.dist = maxsize     # distance from start to current node
        self.neighbors = []         # neighbors node positions

        self.f = 0
        self.g = 0

    def __lt__(self, other):
        return self.dist < other.dist

def dijkstra(data, start):
    h = []
    data[start].dist = 0
    heapq.heappush(h, data[start])

    for _ in data:
        node = heapq.heappop(h)
        for neighbor in node.neighbors:
            if data[neighbor].dist > node.dist + data[neighbor].ter and data[neighbor].ter:
                data[neighbor].dist = node.dist + data[neighbor].ter
                data[neighbor].path = node.pos
                heapq.heappush(h, data[neighbor])

    return data

'''
def aStar(data, start, end):
    neighbors = []
    checked = []
'''
    



def load(file, diagonal):
    with open(file) as f:
        maxRow, maxCol = map(int, f.readline().split()[:2])
        mapStr = ''
        for line in range(maxRow):
            mapStr += f.readline().rstrip('\n')

        #mapStr = f.readline()
        if maxRow*maxCol != len(mapStr):    # az na konci!
            # S - start (required)
            # N - princesses (required)
            # D - one dragon (required)
            # E - end (optional)
            # D - dragon (optional)
            # P - portal (optional)
            print("Incorrect number of map characters")
            return

    mapTerr = {'N': 200, 'H': 2, 'M': 0}
    mapTerr = collections.defaultdict(lambda: 1, mapTerr)
    princesses = []
    mapData = []
    start = 0
    for i, ter in enumerate(mapStr):
        if ter == 'D':
            dragon = i
        if ter == 'P':
            princesses.append(i)
        if ter == 'S':
            start = i
        
        length = mapTerr[ter]
        mapData.append(Node(i, length))
        if i >= maxCol:     # although we could also make automatic new map nodes with max size!
            mapData[i].neighbors.append(i-maxCol)   # UPPER
        if i < maxCol*(maxRow-1):
            mapData[i].neighbors.append(i+maxCol)   # LOWER
        if i % maxCol != 0:
            mapData[i].neighbors.append(i-1)        # LEFT
        if (i+1) % maxCol != 0:
            mapData[i].neighbors.append(i+1)        # RIGHT
        if diagonal:
            if (i+1) % maxCol != 0 and i >= maxCol:             # RIGHT UPPER
                mapData[i].neighbors.append(i+1-maxCol)
            elif (i+1) % maxCol != 0 and i < maxCol*(maxRow-1): # RIGHT LOWER
                mapData[i].neighbors.append(i+1+maxCol)
            elif i % maxCol != 0 and i >= maxCol:               # LEFT UPPER
                mapData[i].neighbors.append(i-1-maxCol)
            elif i % maxCol != 0 and i < maxCol*(maxRow-1):     # LEFT LOWER
                mapData[i].neighbors.append(i-1+maxCol)

    return mapData, princesses, dragon, start, maxRow, maxCol

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

def printSolution(path, maxRow, maxCol):
    for i, road in enumerate(path, 1):
        print(f"\n{i}: ", end=' ')
        for step in road:
            a = step // maxRow
            b = step % maxCol
            print(f"{a}{b}", end=' ')
    print()


def main():
    mapData, princesses, dragon, start, maxRow, maxCol = load("mapa4.txt", True)

    npcData = {p: dijkstra(copy.deepcopy(mapData), p) for p in princesses}
    npcData.update({start: dijkstra(copy.deepcopy(mapData), start)}) # tu bude A*, k drakovi
    #npcData.update({start: aStar(copy.deepcopy(mapData), start, dragon)})
    npcData.update({dragon: dijkstra(copy.deepcopy(mapData), dragon)})

    #dStartCost = aStar(copy.deepcopy(mapData), start, dragon)


    print("Starting permutations")
    order, dist = findMinDist(npcData, princesses, dragon, start)
    path = getPath(npcData, order)
    printSolution(path, maxRow, maxCol)
    print("Cost: " + str(dist))

    print("Starting Kerp")
    order2, dist2 = karp(npcData, princesses, dragon, start)
    path2 = getPath(npcData, order2)
    printSolution(path2, maxRow, maxCol)
    print("Cost: " + str(dist2))
    
main()
