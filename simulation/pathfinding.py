import heapq
from copy import deepcopy as dcopy
from itertools import combinations, permutations
from sys import maxsize
from typing import Any, Dict, FrozenSet, List, Tuple, Union

import evolution as evo
import forward_chain as chain


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
    def __init__(self, fname) -> None:
        self.fname = ""  # type: str
        self.__width = 0  # type: int
        self.__height = 0  # type: int
        self.nodes = {}  # type: Dict[Tuple[int, int], Node]
        self.properties = {}  # type: Dict[str, Any]
        self.__loadMap(fname)

    def __loadMap(self, fname) -> None:
        properties = {
            "points": [],
            "base": 0,
            "start": 0,
        }  # type: Dict[str, Any]
        nodes = {}  # type: Dict[Tuple[int, int], Node]
        try:
            with open("simulation/maps/" + fname + "_pro.txt") as f:
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
            self.fname = fname
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
    data: Map,
    start: Tuple[int, int],
    moves: List[Tuple[int, int]],
    climb: bool,
) -> Map:
    """Finds and saves the shortest path to all destinations from the start.

    Args:
        data (Map): contains information about the map
        start (Tuple[int, int]): starting position
        moves (List[Tuple[int, int]]): tuples of movement options
        climb (bool): climbing distance approach, distance to next position
            is measured as abs(current terrain number - next terrain number)

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
                if climb:
                    dist_next = abs(node.terrain - data[neighbor].terrain + 1)
                else:
                    dist_next = data[neighbor].terrain

                if data[neighbor].dist > node.dist + dist_next:
                    data[neighbor].dist = node.dist + dist_next
                    data[neighbor].parent = node.pos
                    heapq.heappush(heap, data[neighbor])

    return data


def aStar(
    data: Map,
    start: Tuple[int, int],
    dest: Tuple[int, int],
    moves: List[Tuple[int, int]],
    climb: bool,
) -> Map:
    """Finds and saves the shortest path to destination from the start.

    Args:
        data (Map): contains information about the map
        start (Tuple[int, int]): starting position
        dest (Tuple[int, int]): ending position
        moves (List[Tuple[int, int]]): tuples of movement options
        climb (bool): climbing distance approach, distance to next position
            is measured as abs(current terrain number - next terrain number)

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
                if climb:
                    g = node.g + abs(node.terrain - data[neighbor].terrain + 1)
                else:
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
    pro_data: Dict[Tuple[int, int], Map], subset_size: Union[int, None] = 0
) -> Tuple[List[Any], int]:
    """Computes the distance between all possible combinations of properties in
    order to find the shortest paths.

    Args:
        pro_data (Dict[Tuple[int, int], Map]): contains distances between
            all properties. Access via Dict[starting][destination].dist
        subset_size (int, optional): defines how many points we want to visit

    Returns:
        Tuple[List[Any], int]: (order of properties' coordinates, distance)
    """

    points, base, start = tuple(pro_data.values())[0].properties.values()
    if subset_size is None or not 1 < subset_size <= len(points):
        subset_size = len(points)
    mini = maxsize

    for permutation in permutations(points, subset_size):
        distance = pro_data[start][base].dist
        for begin, finish in zip((base,) + permutation, permutation):
            distance += pro_data[begin][finish].dist
        if distance < mini:
            mini, order = distance, permutation

    return list((start, base) + order), mini


def heldKarp(
    pro_data: Dict[Tuple[int, int], Map], subset_size: Union[int, None] = 0
) -> Tuple[List[Any], int]:
    """Finds the shortest combination of paths between properties
    using Held Karp's algorithm.

    Args:
        pro_data (Dict[Tuple[int, int], Map]): contains distances between
            all properties. Access via Dict[starting][destination].dist
        subset_size (int): defines how many points we want to visit

    Returns:
        Tuple[List[Any], int]: (order of properties' coordinates, distance)
    """

    points, base, start = tuple(pro_data.values())[0].properties.values()
    points = frozenset(points)
    if subset_size is None or not 1 < subset_size <= len(points):
        subset_size = len(points)

    key = Tuple[Tuple[int, int], FrozenSet[int]]
    value = Tuple[int, Tuple[int, int]]
    nodes = {}  # type: Dict[key, value]

    # get the shortest combinations of all sizes
    for row in range(subset_size):
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

    # get final destination and its parent to backtrack remaining properties
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

    # backtracking remaining properties
    for _ in range(subset_size - 1):
        path.append(next_step)
        points -= {next_step}
        next_step = nodes[next_step, points][1]
    path.extend([base, start])

    return path[::-1], cost


def noCombed(
    pro_data: Dict[Tuple[int, int], Map], subset_size: int = 0
) -> Tuple[List[Any], int]:
    """Gets the shortest path between properties with zero or one point.

    Args:
        pro_data (Dict[Tuple[int, int], Map]): contains distances between
            all properties. Access via Dict[starting][destination].dist
        subset_size (int, optional): defines how many points we want to visit
            (in this function either one or none)

    Returns:
        Tuple[List[Any], int]: (order of properties' coordinates, distance)
    """

    points, base, start = tuple(pro_data.values())[0].properties.values()
    order, dist = [start, base], pro_data[start][base].dist
    if subset_size:
        point, dist_to_p = min([(p, pro_data[base][p].dist) for p in points])
        order.append(point)
        dist += dist_to_p

    return order, dist


def findShortestDistances(
    map_data: Map, moves: List[Tuple[int, int]], climb: bool
) -> Dict[Tuple[int, int], Map]:
    """Finds the shortest distances between all properties:
    points, base, start in the map using Dijkstra and A* algorithms.

    Args:
        map_data (Map): contains information about the map
        moves (List[Tuple[int, int]]): tuples of x, y coordinate movement opts
        climb (bool): climbing distance approach, distance to next position
            is measured as abs(current terrain number - next terrain number)

    Returns:
        Dict[Tuple[int, int], Map]: contains distances between all properties
            Access via Dict[starting][destination].dist
    """

    points, base, start = map_data.properties.values()

    pro_data = {p: dijkstra(dcopy(map_data), p, moves, climb) for p in points}
    pro_data.update({start: aStar(dcopy(map_data), start, base, moves, climb)})
    pro_data.update({base: dijkstra(dcopy(map_data), base, moves, climb)})

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
        print(f"{i}: ", *path)

    print("Cost: " + str(distance) + "\n")


def runPathfinding(pars: Dict[str, Any]) -> List[List[Tuple[int, int]]]:
    """Finds a solution and prints it.

    Args:
        pars (Dict[str, Any]): parameters that contain these string values:
            fname (string): name of the file without _pro.txt
            movement (string): D - Diagonal, any - Manhattan
            climb (bool): climbing distance approach
            algorithm (string): NP - Naive Permutations, any - Held Karp
            subset_size (None/int): number of points we want to visit
                None means all

    Returns:
        paths (List[List[Tuple[int, int]]]): Lists of paths between
            ordered properties
    """

    moves = getMoves(pars["movement"])
    map_data = Map(pars["fname"])
    pro_data = findShortestDistances(map_data, moves, pars["climb"])

    if pars["subset_size"] is not None and pars["subset_size"] < 2:
        order, dist = noCombed(pro_data, pars["subset_size"])
    elif pars["algorithm"] == "NP":
        order, dist = naivePermutations(pro_data, pars["subset_size"])
    else:
        order, dist = heldKarp(pro_data, pars["subset_size"])

    path = getPaths(pro_data, order)

    printSolution(path, dist)
    return path


if __name__ == "__main__":

    points = 10
    evo_parameters = dict(
        max_runs=3,
        points_amount=points,
        export_name="queried",
        query="10x12 (1,5) (2,1) (3,4) (4,2) (6,8) (6,9)",
    )

    path_parameters = dict(
        fname="queried",
        movement="M",
        climb=True,
        algorithm="HK",
        subset_size=None,
    )

    chain_parameters = dict(
        save_fname_facts="facts",
        load_fname_facts="facts_init",
        load_fname_rules="rules",
        step_by_step=True,
        facts_amount=points+1,
        facts_random_order=True,
    )

    evo.runEvolution(evo_parameters)
    runPathfinding(path_parameters)
    chain.runProduction(chain_parameters)


# ToDo: Create tests
# ToDo: Optimize Python code with advanced techniques
# ToDo: Create visualizations
