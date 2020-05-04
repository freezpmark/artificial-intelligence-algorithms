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
    '''
    for row in range(1, len(princesses)+1):
        sets.append(collections.defaultdict(lambda:[maxsize]))
        for comb in combinations(princesses, row):
            for prevComb in combinations(comb, len(comb)-1):
                prevComb = frozenset(prevComb)
            
    '''
    '''
    for row in range(1, len(princesses)+1):
        sets.append(collections.defaultdict(lambda:[maxsize]))
        for comb in combinations(princesses, row):
            for prevComb in sets[row-1]:
                comb = frozenset(comb)
                finish = list(comb - prevComb)[0]   # only one and first element can be there
                begin = sets[row-1][prevComb][POS]
                cost = sets[row-1][prevComb][DIST] + npcData[begin][finish].dist
                if sets[row][comb][DIST] > cost:
                    sets[row][comb] = cost, finish
    '''
    #sets = [{frozenset(): (npcData[start][dragon].dist, dragon)}]
    #sets = [{frozenset(): (npcData[start][dragon].dist, dragon, start)}]
    #nmap = [{frozenset(): {dragon: (npcData[start][dragon].dist, start)}}]
    # [rowLIST][setDICT][finish]

    # nakreslit si algoritmus (grafy), zapisovat si vypocitavania (matika), 
    # spojit dokopy a hladat vztahy

    # rowLIST = cislo riadku
    # setDICT = dictionary, pri ktorom key je frozenset kombinacie (groupa v ramci riadku)
    # finish =  dictionary, pri ktorom key je finish, a value je jeho cost a predosli node (node)
    
    #########################################
    # Understand the algorithm works
    # visualize it, math it on paint 
    # search for patterns to get the CYCLE structure done first
    # then think of data types
    DIST, POS, PAT = 0, 1, 2
    #nodes = {(dragon, frozenset()): (npcData[start][dragon].dist, start)}
    nodes = {}

    princesses = frozenset(princesses)
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
                        prevCost = nodes[begin, subcomb][DIST]
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


    
    

    '''                         IF DRAGON INCLUDED IN CYCLE
    if comb == frozenset():
        nodes[dragon, frozenset()] = npcData[start][dragon].dist, start
        break
    '''
                # first cycle must be empty combo, but we must iterate 3 finishes





                    #prevCost = min(i for i in range(10))


                    # get begin (for the one step part of cost)
                    # + the cost in previous set value in which the dest is begin AND from
                    # the remaining values in previous set comb

                    # we need a key which consists of DEST and COMB
                    # data types: string, tuple, class (class is too complex for it)
                    # string would be great, the last letter could serve as the dest and the rest 
                        # represent combo, but ive seen people doin it like that
                        # and since im doing it via these frozensets, I shall continue with it
                    
                    #prevComb = frozenset({prevComb})
                    #prevCost, begin, path = sets[row][comb2 - prevComb]
                    #begin = sets[row][prevComb][POS]
                    #cost = npcData[begin][finish].dist + prevCost

                    #nmap[row][comb2].update({finish: (cost, begin, finish)})
                    #sets[row+1].update({comb2: (cost, finish, begin)})



                    #subcomb2 = frozenset(subcomb)
                    #begin = sets[row-1][subcomb2][2]
                    #prevCost = sets[row-1][subcomb2]
                # comb je zaciatok
                # dostanem ale 24,20,4
                # potrebujem tie ktore su nadtym, tj 24,20; 24,4; 20,4
    '''
    for prevComb in prevCombs:
        prevComb2 = frozenset(prevComb)
    for prevComb in nmap[row-1]:
        if prevComb[princesses - comb2 - {finish}] in nmap[row-1]
    '''
                # ideme do nodu FINISH, ako zistit begin? v combo mame begin - to je ale mnozina
                # takze musime ju zminusovat s niecim takze list(comb2 - ...)
                # zminusovat ju musime z predosleho rowu.. ako? kde a jak
                # v prvom pripade to je comb2 - {}
                # v druhom pripade to bude tiez comb2 - {} bo v minulom sme este nemali nic v mnozine
                # ist do frozensetov ktore nemaju finish

                # spravit kombinacie s poctom o -1
    '''
    if row:
        begin = list(comb2)[0]
        # comb3 = comb2.union()
        prevCost = nmap[row-1][prevComb][begin][DIST]       # PRESUNUT HORE
    else:
        prevCost, begin = npcData[start][dragon].dist, dragon
    cost = npcData[begin][finish].dist + prevCost

    nmap[row][comb2].update({finish: (cost, begin, finish)})
    '''

    '''
    for prevComb in sets[row-1]:                    # 
        prevComb = frozenset(prevComb)
        for finish in comb2 - prevComb:
            begin = sets[row-1][prevComb][POS]
            cost = sets[row-1][prevComb][DIST] + npcData[begin][finish].dist
            if sets[row][comb2][DIST] > cost:
                sets[row][comb2] = cost, finish
            a=1
            b=2
        a=1
        b=2
    '''
    '''
    for compart in combinations(comb, len(comb)-1): 
        sets[row-1][compart] - compart

    for prevComb in sets[row-1]:

        comb = frozenset(comb)
        finish = list(comb - prevComb)[0]   # only one and first element can be there
        begin = sets[row-1][prevComb][POS]
        cost = sets[row-1][prevComb][DIST] + npcData[begin][finish].dist
        if sets[row][comb][DIST] > cost:
            sets[row][comb] = cost, finish
    '''

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


def main():
    mapData, princesses, dragon, start, maxRow, maxCol = load("mapa2.txt")

    npcData = {p: dijkstra(copy.deepcopy(mapData), p) for p in princesses}
    npcData.update({start: dijkstra(copy.deepcopy(mapData), start)})
    npcData.update({dragon: dijkstra(copy.deepcopy(mapData), dragon)})

    order, dist = findMinDist(npcData, princesses, dragon, start)
    order2, dist2 = karp(npcData, princesses, dragon, start)
    
    path = getPath(npcData, order)
    path2 = getPath(npcData, order2)

    print(dist, end='\n')
    printSolution(path, maxRow, maxCol)
    print(dist2, end='\n')
    printSolution(path2, maxRow, maxCol)
    a=2
    
main()
