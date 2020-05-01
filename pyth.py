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
    mapTerr = collections.defaultdict(lambda:1, mapTerr)
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
        for start, dest in zip(permutation, (dragon,)+permutation):
            distance += npcData[start][dest].dist
        if distance < mini:
            mini, order = distance, permutation

    return order

def karp(npcData, princesses, dragon, start):
    DIST, PATH = 0, 1
    sets = [{frozenset(): (npcData[dragon][start].dist, start)}]

    for row in range(1, len(princesses)+1):
        sets.append({})
        for subset in combinations(princesses, row):
            for combo in sets[row-1]:
                ssub = frozenset(subset)        # current combination
                dest = list(ssub - combo)[0]        # 0 -> 1
                start = sets[row-1][combo][PATH]

                if row == 1:                    # get normal matrix !
                    dest, start = dragon, dest

                cost = sets[row-1][combo][DIST] + npcData[start][dest].dist

                if sets[row].get(ssub, [maxsize])[DIST] > cost:
                    sets[row][ssub] = cost, start
        
    return subset

def getPath(npcData, start, dest, backward=False):
    path = []
    pos = npcData[dest][start].pos
    if backward:
        path.append(pos)

    while pos != dest:
        step = npcData[dest][pos].path
        path.append(step)
        pos = npcData[dest][step].pos

    return path

def main():
    mapData, princesses, dragon, start = load("mapa.txt") # load start too

    npcData =       {p:         dijkstra(copy.deepcopy(mapData), p) for p in princesses}
    npcData.update( {start:     dijkstra(copy.deepcopy(mapData), start)})
    npcData.update( {dragon:    dijkstra(copy.deepcopy(mapData), dragon)})
    
    # we need the initial i node in order to get to j node (npcData[i][j]) by backtracking
    # backtracking causes reversal path and distances. 
    # in order to have consistent distances with all other npcs,
    # we have to hackingly compute this node again with the help absolute value 

    #for i in npcData[dragon]:
    #    npcData[dragon][i.pos].dist = abs(i.dist - npcData[dragon][dragon].dist)

    order = findMinDist(npcData, princesses, dragon, start)
    order2 = karp(npcData, princesses, dragon, start)
    
    path = []
    for star, dest in zip(order[::-1][1:]+(dragon,), order[::-1]):
        path.append(getPath(npcData, star, dest))
    path.append(getPath(npcData, dragon, start, True)[::-1])

    for i, road in enumerate(path[::-1], 1):
        print(f"\n{i}: ", end=' ')
        for step in road:
            a = step // 5
            b = step % 5
            print(f"{a}{b}", end=' ')
    
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
