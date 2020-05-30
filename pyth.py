import heapq
import copy
import evolution
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
        with open(file_name) as f:
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


def dijkstra(data, start, moves):
    heap = []
    data[start].dist = 0
    heapq.heappush(heap, data[start])

    while heap:
        node = heapq.heappop(heap)
        for adjacent in moves:
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


def aStar(data, start, end, moves):
    open_list = []
    close_list = []
    data[start].g = 0
    heapq.heappush(open_list, data[start])

    while open_list:
        node = heapq.heappop(open_list)
        close_list.append(node.pos)

        if node.pos == end:
            break

        for adjacent in moves:
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


def naivePermutations(npc_data, map_data):
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


def heldKarp(npc_data, map_data):
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


def findPaths(map_data, moves):
    """Find all the shortest possible paths between all entities
    papers, base, start in the map using Dijkstra and A* algorithms.

    Arguments:
        map_data {Map} -- describes the whole map and its entities
        moves {list} -- list of tuples that represent moving options

    Returns:
        dict -- dictionary that describes shortest paths between all entities
                with keys being tuple coordinates. Access via dict[start][dest]
    """
    papers, base, start = map_data.entities.values()

    ent_data = {p: dijkstra(
        copy.deepcopy(map_data), p, moves) for p in papers}
    ent_data.update({start: aStar(
        copy.deepcopy(map_data), start, base, moves)})
    ent_data.update({base: dijkstra(
        copy.deepcopy(map_data), base, moves)})

    return ent_data


def main():
    query = "amap.txt"  # SAME AS: "10x12 (1,5) (2,1) (3,4) (4,2) (6,8) (6,9)"
    attempts = 3
    papers = 3
    moving_direction = "M"
    # ToDo: terrain_type = "C"

    file_name = evolution.create(query, attempts, papers)
    if not file_name:
        return

    # ? no need to check every step, remake inc
    map_data = Map('evo_' + file_name)
    if not map_data:
        print("Invalid map!")
        return

    moves = getMoves(moving_direction)
    npc_data = findPaths(map_data, moves)

    print("Starting Naive solution")
    order, dist = naivePermutations(npc_data, map_data)
    path = getPaths(npc_data, order)
    printSolution(path)
    print("Cost: " + str(dist))

    # alot faster solution
    print("Starting Held Karp")
    order2, dist2 = heldKarp(npc_data, map_data)
    path2 = getPaths(npc_data, order2)
    printSolution(path2)
    print("Cost: " + str(dist2))


main()

# ToDo: Differentiate 2 cases in loadMap at evolution.py and make 2 functions
# ToDo: Add function annotations
# ToDo: Finish docstrings, current ones need corrections as theyre not clear
# ToDo: Held Karr (Add Shortest subset combo)
# ToDo: Pathfinding (C: Climbing, S: Swamp)
# ToDo: Validation checks (__loadMap, coordinate checkings for example)
# ToDo: Create tests
# ToDo: Add Rule based system in the end (each paper is one fact)
