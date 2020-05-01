import heapq, copy, collections
from sys import maxsize
from itertools import permutations, combinations

class Node:
    def __init__(self, pos, ter):
        self.pos = pos          # position
        self.ter = ter          # terrain
        self.path = -1          # node position from which we got to this node
        self.dist = 1000        # distance from start to current node
        self.edges = []         # neighbor node positions

    def __lt__(self, other):
        return self.dist < other.dist

def dijkstra(data, start):
    h = []
    data[start].dist = 0
    heapq.heappush(h, data[start])

    for _ in data:
        node = heapq.heappop(h)
        for edge in node.edges:
            if data[edge].dist > node.dist + data[edge].ter:
                data[edge].dist = node.dist + data[edge].ter
                data[edge].path = node.pos
                heapq.heappush(h, data[edge])

    return data

def load(file):
    with open(file) as f:
        maxRow, maxCol = map(int, f.readline().split()[:2])
        mapStr = f.readline()
        if maxRow*maxCol != len(mapStr):    # az na konci!
            # S - start is required
            # N - princesses are required
            # E - end (optional)
            # D - dragon (optional)
            # P - portal (optional)
            print("Incorrect number of map characters")
            return

    mapTerr = {'N': 200, 'H': 2}
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
        if i >= maxCol:
            mapData[i].edges.append(i-maxCol)   # UPPER
        if i < maxCol*(maxRow-1):
            mapData[i].edges.append(i+maxCol)   # LOWER
        if i % maxCol != 0:
            mapData[i].edges.append(i-1)        # LEFT
        if (i+1) % maxCol != 0:
            mapData[i].edges.append(i+1)        # RIGHT

    return mapData, princesses, dragon, start

def findMinDist(npcData, princesses, dragon, start):
    mini = maxsize

    for permutation in permutations(princesses):
        distance = npcData[start][dragon].dist     # distance to get to dragon
        for begin, finish in zip((dragon,)+permutation, permutation):
            distance += npcData[begin][finish].dist
        if distance < mini:
            mini, order = distance, permutation

    return order, mini

def karp(npcData, princesses, dragon, start):
    DIST, POS = 0, 1
    sets = [{frozenset(): (npcData[start][dragon].dist, dragon)}]

    for row in range(1, len(princesses)+1):
        sets.append(collections.defaultdict(lambda:[maxsize]))
        for comb in combinations(princesses, row):
            for prevComb in sets[row-1]:
                comb = frozenset(comb)
                finish = list(comb - prevComb)[0]   # only one and first element can be there
                begin = sets[row-1][prevComb][POS]
                cost = sets[row-1][prevComb][DIST] + npcData[begin][finish].dist
                if sets[row][comb][DIST] > cost:
                    sets[row][comb] = cost, finish, begin
    
    paath = []
    prev = comb
    for i in range(row, 0, -1):
        paath.append(sets[i][prev][POS])
        prev = prev - {sets[i][prev][POS]}

    return paath[::-1], sets[row][comb][DIST]


def getPath(npcData, princesses, dragon, start, order):
    path = []
    for star, dest in zip(order[::-1][1:]+(dragon,), order[::-1]):
        path2 = []
        pos = npcData[dest][star].pos
        while pos != dest:
            step = npcData[dest][pos].path
            path2.append(step)
            pos = npcData[dest][step].pos

        path.append(path2)

    path2 = []
    pos = npcData[start][dragon].pos
    path2.append(pos)

    while pos != start:
        step = npcData[start][pos].path
        path2.append(step)
        pos = npcData[start][step].pos

    path.append(path2[::-1])

    return path

def printPath(path):
    for i, road in enumerate(path[::-1], 1):
        print(f"\n{i}: ", end=' ')
        for step in road:
            a = step // 5
            b = step % 5
            print(f"{a}{b}", end=' ')


def main():
    mapData, princesses, dragon, start = load("mapa.txt") # load start too

    npcData = {p: dijkstra(copy.deepcopy(mapData), p) for p in princesses}
    npcData.update({start: dijkstra(copy.deepcopy(mapData), start)})
    npcData.update({dragon: dijkstra(copy.deepcopy(mapData), dragon)})

    order, dist = findMinDist(npcData, princesses, dragon, start)
    order2, dist2 = karp(npcData, princesses, dragon, start)
    
    path = getPath      (npcData, princesses, start, dragon, order)

    printPath(path)
    
main()



"""
# from heapq import heappush, heappop

end = ["1","2","3","4","5","6","7","8","0"]
start = ["0","1","2","3","4","5","6","7","8"]
start_text = ("Starting state:\n" + " ".join(start) +
            "\nEnter 9 numbers if you want to have custom starting state\n")
end_text = ("Ending state:\n" + " ".join(end) + 
            "\nEnter another 9 numbers if you want to have custom ending state")

change = input(start_text)
end = end if change else input(end_text).split()
start = start if change else change.split()

print(change)
print(end)
print(start)
"""


# we need the initial i node in order to get to j node (npcData[i][j]) by backtracking
# backtracking causes reversal path and distances. 
# in order to have consistent distances with all other npcs,
# we have to hackingly compute this node again with the help absolute value 

#for i in npcData[dragon]:
#    npcData[dragon][i.pos].dist = abs(i.dist - npcData[dragon][dragon].dist)