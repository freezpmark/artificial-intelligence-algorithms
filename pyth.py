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
    DIST, POS = 0, 1
    sets = [{frozenset(): (npcData[start][dragon].dist, dragon)}]
    princesses = frozenset(princesses)
    '''
    for row in range(1, len(princesses)+1):
        sets.append(collections.defaultdict(lambda:[maxsize]))
        for comb in combinations(princesses, row):
            for prevComb in combinations(comb, len(comb)-1):
                prevComb = frozenset(prevComb)
                #prevComb = sets[row-1][frozenset(subcomb)]
            

            #for prevComb in sets[row-1]:
                comb = frozenset(comb)
                finish = list(comb - prevComb)[0]   # only one and first element can be there
                begin = sets[row-1][prevComb][POS]
                cost = sets[row-1][prevComb][DIST] + npcData[begin][finish].dist
                if sets[row][comb][DIST] > cost:
                    sets[row][comb] = cost, finish
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
    sets = [{frozenset(): (npcData[start][dragon].dist, dragon)}]
    #nmap = [{frozenset(): {dragon: (npcData[start][dragon].dist, start)}}]
    nmap = []
    # [rowLIST][setDICT][finish]

    # nakreslit si algoritmus (grafy), zapisovat si vypocitavania (matika), 
    # spojit dokopy a hladat vztahy

    # rowLIST = cislo riadku
    # setDICT = dictionary, pri ktorom key je frozenset kombinacie (groupa v ramci riadku)
    # finish =  dictionary, pri ktorom key je finish, a value je jeho cost a predosli node (node)
    princesses = frozenset(princesses)

    for row in range(len(princesses)):
        nmap.append({})
        for comb in combinations(princesses, row):          # prava strana combo
            comb2 = frozenset(comb)
            nmap[row].update({comb2: {}})

            for finish in princesses - comb2:                # lava strana
                for subcomb in combinations(comb, row-1):
                    subcomb2 = frozenset(subcomb)
                    begin = nmap[row-1][subcomb2][2]
                    prevCost = nmap[row-1][subcomb2]
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

                if row:
                    begin = list(comb2)[0]
                    # comb3 = comb2.union()
                    prevCost = nmap[row-1][prevComb][begin][DIST]       # PRESUNUT HORE
                else:
                    prevCost, begin = npcData[start][dragon].dist, dragon
                cost = npcData[begin][finish].dist + prevCost

                nmap[row][comb2].update({finish: (cost, begin, finish)})
        #prevCombs = combs




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

    paath = []
    prev = comb
    for i in range(row, 0, -1):
        paath.append(sets[i][prev][POS])
        prev = prev - {sets[i][prev][POS]}

    return [start, dragon] + paath[::-1], sets[row][comb][DIST]

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
