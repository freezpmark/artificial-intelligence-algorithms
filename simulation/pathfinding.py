import heapq
from copy import deepcopy
from itertools import combinations, permutations
from sys import maxsize
from typing import Any, Dict, FrozenSet, List, Tuple


class PositionError(Exception):
    pass


class Node:
    def __init__(self, pos, terrain):
        self.pos = pos  # type: Tuple[int, int]
        self.terrain = terrain  # type: int
        self.parent = -1  # type: Tuple[int, int]
        self.dist = maxsize  # type: int
        self.g = maxsize  # type: int
        self.h = maxsize  # type: int

    def __lt__(self, other):
        if self.dist != other.dist:
            return self.dist < other.dist
        return self.h < other.h


class Map:
    def __init__(self, file_name) -> None:
        self.file_name = ""  # type: str
        self.__width = 0  # type: int
        self.__height = 0  # type: int
        self.nodes = {}  # type: Dict[Tuple[int, int], Node]
        self.properties = {}  # type: Dict[str, Any]
        self.__loadMap(file_name)

    def __loadMap(self, file_name) -> None:
        properties = {
            "points": [],
            "base": 0,
            "start": 0,
        }  # type: Dict[str, Any]
        nodes = {}  # type: Dict[Tuple[int, int], Node]
        try:
            with open("simulation/maps/" + file_name + "_pro.txt") as f:
                for i, line in enumerate(f):
                    for j, col in enumerate(line.split()):
                        if col[0] in "([{":
                            if col[0] == "(":
                                properties["points"].append((i, j))
                            elif col[0] == "[":
                                properties["base"] = (i, j)
                            elif col[0] == "{":
                                properties["start"] = (i, j)
                            col = col[1:-1]
                        nodes[i, j] = Node((i, j), int(col))
        except FileNotFoundError:
            pass

        if all(properties.values()) and len(properties["points"]) > 1:
            self.file_name = file_name
            self.nodes = nodes
            self.properties = properties
            self.__height = i + 1
            self.__width = j + 1

    def __getitem__(self, pos):
        assert len(pos) == 2, "Coordinate must have two values."
        if not (0 <= pos[0] < self.height) or not (0 <= pos[1] < self.width):
            raise PositionError(str(pos))
        return self.nodes[pos]

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height


def getMoves(query: str) -> List[Tuple[int, int]]:
    """Gets moving directions. Manhattan type is set by default, to extend
    it with diagonal moves use "D" as first character in the query.

    Args:
        query (str): first character determines the type of movement directions

    Returns:
        List[Tuple[int, int]]: tuples of x, y coordinate movement options
    """

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if query[0] == "D":
        moves.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    return moves


def unpassable(neighbor: Tuple[int, int], data: Map):
    """Checks whether neighbor position is walled or out of map.

    Args:
        neighbor (Tuple[int, int]): position on which movement was applied
        data (Map): contains information about the map

    Returns:
        bool: passability
    """

    return (
        not 0 <= neighbor[0] < data.height
        or not 0 <= neighbor[1] < data.width
        or data[neighbor].terrain == -1
    )


def dijkstra(
    data: Map, start: Tuple[int, int], moves: List[Tuple[int, int]]
) -> Map:
    """Finds and saves the shortest path to all destinations from the start.

    Args:
        data (Map): contains information about the map
        start (Tuple[int, int]): starting position
        moves (List[Tuple[int, int]]): tuples of movement options

    Returns:
        Map: contains distances between starting position and any destination
            Access via Map[starting][destination].dist
    """

    heap = []  # type: List[Node]
    data[start].dist = 0
    heapq.heappush(heap, data[start])

    while heap:
        node = heapq.heappop(heap)
        for move in moves:
            neighbor = node.pos[0] + move[0], node.pos[1] + move[1]
            if not unpassable(neighbor, data):
                if data[neighbor].dist > node.dist + data[neighbor].terrain:
                    data[neighbor].dist = node.dist + data[neighbor].terrain
                    data[neighbor].parent = node.pos
                    heapq.heappush(heap, data[neighbor])

    return data


def aStar(
    data: Map,
    start: Tuple[int, int],
    dest: Tuple[int, int],
    moves: List[Tuple[int, int]],
) -> Map:
    """Finds and saves the shortest path to destination from the start.

    Args:
        data (Map): contains information about the map
        start (Tuple[int, int]): starting position
        dest (Tuple[int, int]): ending position
        moves (List[Tuple[int, int]]): tuples of movement options

    Returns:
        Map: contains distances between starting position and destination
            Access via Map[starting][destination].dist
    """

    open_list = []  # type: List[Node]
    close_list = []
    data[start].g = 0
    heapq.heappush(open_list, data[start])

    while open_list:
        node = heapq.heappop(open_list)
        if node.pos == dest:
            break

        close_list.append(node.pos)
        for move in moves:
            neighbor = node.pos[0] + move[0], node.pos[1] + move[1]
            if not unpassable(neighbor, data) and neighbor not in close_list:
                x = abs(data[neighbor].pos[0] - dest[0])
                y = abs(data[neighbor].pos[1] - dest[1])
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


def naivePermutations(
    pro_data: Dict[Tuple[int, int], Map], map_data: Map
) -> Tuple[List[Any], int]:
    """Computes the distance between all possible combinations of properties in
    order to find the shortest paths.

    Args:
        pro_data (Dict[Tuple[int, int], Map]): contains distances between
            all properties. Access via Dict[starting][destination].dist
        map_data (Map): contains information about the map

    Returns:
        Tuple[List[Any], int]: (order of properties' coordinates, distance)
    """

    points, base, start = map_data.properties.values()
    mini = maxsize

    for permutation in permutations(points):
        distance = pro_data[start][base].dist
        for begin, finish in zip((base,) + permutation, permutation):
            distance += pro_data[begin][finish].dist
        if distance < mini:
            mini, order = distance, permutation

    return list((start, base) + order), mini


def heldKarp(
    pro_data: Dict[Tuple[int, int], Map], map_data: Map
) -> Tuple[List[Any], int]:
    """Finds the shortest combination of paths between properties
    using Held Karp's algorithm.

    Args:
        pro_data (Dict[Tuple[int, int], Map]): contains distances between
            all properties. Access via Dict[starting][destination].dist
        map_data (Map): contains information about the map

    Returns:
        Tuple[List[Any], int]: (order of properties' coordinates, distance)
    """

    points, base, start = map_data.properties.values()
    points = frozenset(points)

    key = Tuple[Tuple[int, int], FrozenSet[int]]
    value = Tuple[int, Tuple[int, int]]
    nodes = {}  # type: Dict[key, value]

    for row in range(len(points)):
        for comb in combinations(points, row):
            combSet = frozenset(comb)
            for dest in points - combSet:
                routes = []
                if combSet == frozenset():  # case for base starting
                    cost = (
                        pro_data[base][dest].dist + pro_data[start][base].dist
                    )
                    nodes[dest, frozenset()] = cost, base
                else:
                    for begin in combSet:  # single val from set
                        sub_comb = combSet - frozenset({begin})
                        prev_cost = nodes[begin, sub_comb][0]
                        cost = pro_data[begin][dest].dist + prev_cost
                        routes.append((cost, begin))
                    nodes[dest, combSet] = min(routes)

    com = []
    for i, node in enumerate(reversed(dict(nodes))):
        if i < len(points):
            com.append((nodes.pop(node), node[0]))
        elif i == len(points):
            val, step = min(com)
            points -= {step}
            path = [step]
            cost, next_step = val
            break

    for _ in points:
        path.append(next_step)
        points -= {next_step}
        next_step = nodes[next_step, points][1]
    path.extend([base, start])

    return path[::-1], cost


def findShortestDistances(
    map_data: Map, moves: List[Tuple[int, int]]
) -> Dict[Tuple[int, int], Map]:
    """Finds the shortest distances between all properties:
    points, base, start in the map using Dijkstra and A* algorithms.

    Args:
        map_data (Map): contains information about the map
        moves (List[Tuple[int, int]]): tuples of x, y coordinate movement opts

    Returns:
        Dict[Tuple[int, int], Map]: contains distances between all properties
            Access via Dict[starting][destination].dist
    """

    points, base, start = map_data.properties.values()

    pro_data = {p: dijkstra(deepcopy(map_data), p, moves) for p in points}
    pro_data.update({start: aStar(deepcopy(map_data), start, base, moves)})
    pro_data.update({base: dijkstra(deepcopy(map_data), base, moves)})

    return pro_data


def getPaths(
    pro_data: Dict[Tuple[int, int], Map], order: List[Any]
) -> List[List[Tuple[int, int]]]:
    """Gets routes from ordered coordinates of properties via parent attribute.

    Args:
        pro_data (Dict[Tuple[int, int], Map]): contains distances between
            all properties. Access via Dict[starting][destination].dist
        order (List[Any]): order of properties' coordinates

    Returns:
        List[List[Tuple[int, int]]]: Lists of paths between ordered properties
    """

    paths = []
    for begin, finish in zip(order, order[1:]):
        path = []
        while finish != begin:
            path.append(finish)
            finish = pro_data[begin][finish].parent
        paths.append(path[::-1])

    return paths


def printSolution(paths: List[List[Tuple[int, int]]], distance: int) -> None:
    """Prints the order of paths between properties. Each line starts with
    order number followed by order of tuple coordinates that represent
    the movement progression from start to destination property.

    Args:
        paths (List[List[Tuple[int, int]]]): Lists of paths between
            ordered properties
        distance (int): total distance of solution
    """

    for i, path in enumerate(paths, 1):
        print(f"{i}: ", end=" ")
        for step in path:
            print(step, end=" ")
        print()

    print("Cost: " + str(distance) + "\n")


def main() -> None:
    file_name = "queried"
    movement = "M"

    moves = getMoves(movement)
    map_data = Map(file_name)
    pro_data = findShortestDistances(map_data, moves)

    print("Starting Naive solution")
    order, dist = naivePermutations(pro_data, map_data)
    path = getPaths(pro_data, order)
    printSolution(path, dist)

    print("Starting Held Karp")
    order2, dist2 = heldKarp(pro_data, map_data)
    path2 = getPaths(pro_data, order2)
    printSolution(path2, dist2)


main()

# create <query> -> <newfilename>           ( -> ... is optional)
# load <filename> -> <newfilename>          ( -> ... is optional)
# use better parser..?

# ToDo: Create tests
# ToDo: Held Karp (Add Shortest subset combo)
# ToDo: Pathfinding (C: Climbing, S: Swamp)
# ToDo: Add Rule based system in the end (each paper is one fact)
