import heapq
from copy import deepcopy as dcopy
from itertools import combinations, permutations
from sys import maxsize
from typing import Any, Dict, FrozenSet, List, Tuple

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
        self._width = 0  # type: int
        self._height = 0  # type: int
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
            self._height = i + 1
            self._width = j + 1
            self.nodes = nodes
            self.properties = properties

    def __getitem__(self, pos):
        assert len(pos) == 2, "Coordinate must have two values."
        if not (0 <= pos[0] < self.height) or not (0 <= pos[1] < self.width):
            raise PositionError(str(pos))
        return self.nodes[pos]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


def getMoves(query: str) -> List[Tuple[int, int]]:
    """Gets moving options.

    Args:
        query (str): determines the type of movement options
            ("M" - Manhattan, "D" - Diagonal + Manhattan)

    Returns:
        List[Tuple[int, int]]: tuples of x, y coordinate movement options
    """

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if query == "M":
        return moves
    elif query == "D":
        return moves + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    return []


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
        climb (bool): Climbing distance approach. If True, distance is measured
            with abs(current terrain number - next terrain number)

    Returns:
        Map: Contains distances from starting position to all destinations.
            Access via Map[start][destination].dist
    """

    heap = []  # type: List[Node]
    data[start].dist = 0
    heapq.heappush(heap, data[start])

    while heap:
        node = heapq.heappop(heap)
        for move in moves:
            neighbor = node.pos[0] + move[0], node.pos[1] + move[1]
            if not unpassable(neighbor, data):
                next_dist = (
                    data[neighbor].terrain
                    if not climb
                    else abs(node.terrain - data[neighbor].terrain + 1)
                ) + node.dist

                if data[neighbor].dist > next_dist:
                    data[neighbor].dist = next_dist
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
        climb (bool): Climbing distance approach. If True, distance is measured
            with abs(current terrain number - next terrain number)

    Returns:
        Map: Contains distance between starting position and destination.
            Access via Map[start][destination].dist
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
                h = abs(data[neighbor].pos[0] - dest[0]) + abs(
                    data[neighbor].pos[1] - dest[1]
                )
                g = (
                    data[neighbor].terrain
                    if not climb
                    else abs(node.terrain - data[neighbor].terrain + 1)
                ) + node.g
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
    pro_data: Dict[Tuple[int, int], Map], subset_size: int
) -> Tuple[List[Any], int]:
    """Computes the distance between all possible combinations of properties in
    order to find the shortest paths.

    Args:
        pro_data (Dict[Tuple[int, int], Map]): Contains distances between
            all properties. Access via Dict[starting][destination].dist
        subset_size (int): number of points to visit (more than 1)

    Returns:
        Tuple[List[Any], int]: (order of properties' coordinates, distance)
    """

    points, base, start = tuple(pro_data.values())[0].properties.values()
    cost = maxsize

    for permutation in permutations(points, subset_size):
        distance = pro_data[start][base].dist
        for begin, finish in zip((base,) + permutation, permutation):
            distance += pro_data[begin][finish].dist
        if distance < cost:
            cost, pro_order = distance, permutation

    return list((start, base) + pro_order), cost


def heldKarp(
    pro_data: Dict[Tuple[int, int], Map], subset_size: int
) -> Tuple[List[Any], int]:
    """Finds the shortest combination of paths between properties
    using Held Karp's algorithm.

    Args:
        pro_data (Dict[Tuple[int, int], Map]): Contains distances between
            all properties. Access via Dict[starting][destination].dist
        subset_size (int): number of points to visit (more than 1)

    Returns:
        Tuple[List[Any], int]: (order of properties' coordinates, distance)
    """

    points, base, start = tuple(pro_data.values())[0].properties.values()
    points = frozenset(points)

    key = Tuple[Tuple[int, int], FrozenSet[int]]
    value = Tuple[int, Tuple[int, int]]
    nodes = {}  # type: Dict[key, value]

    # get the shortest combinations of all sizes
    for row in range(subset_size):
        for comb in combinations(points, row):
            comb_set = frozenset(comb)
            for dest in points - comb_set:
                routes = []
                if comb_set == frozenset():  # case for base starting
                    cost = (
                        pro_data[base][dest].dist + pro_data[start][base].dist
                    )
                    nodes[dest, frozenset()] = cost, base
                else:
                    for begin in comb_set:  # single val from set
                        sub_comb = comb_set - frozenset({begin})
                        prev_cost = nodes[begin, sub_comb][0]
                        cost = pro_data[begin][dest].dist + prev_cost
                        routes.append((cost, begin))
                    nodes[dest, comb_set] = min(routes)

    # get final destination and its parent
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


def noComb(
    pro_data: Dict[Tuple[int, int], Map], subset_size: int
) -> Tuple[List[Any], int]:
    """Gets the shortest path between properties with 0 or 1 point.

    Args:
        pro_data (Dict[Tuple[int, int], Map]): Contains distances between
            all properties. Access via Dict[starting][destination].dist
        subset_size (int): number of points to visit (0 or 1 in this case)

    Returns:
        Tuple[List[Any], int]: (order of properties' coordinates, distance)
    """

    points, base, start = tuple(pro_data.values())[0].properties.values()
    pro_order, dist = [start, base], pro_data[start][base].dist
    if subset_size:
        point, dist_to_p = min([(p, pro_data[base][p].dist) for p in points])
        pro_order.append(point)
        dist += dist_to_p

    return pro_order, dist


def findShortestDistances(
    map_data: Map, moves: List[Tuple[int, int]], climb: bool
) -> Dict[Tuple[int, int], Map]:
    """Finds shortest distances between all properties (points, base, start)
    in the map using Dijkstra and A* algorithms.

    Args:
        map_data (Map): contains information about the map
        moves (List[Tuple[int, int]]): tuples of x, y coordinate movement opts
        climb (bool): Climbing distance approach. If True, distance is measured
            with abs(current terrain number - next terrain number)

    Returns:
        Dict[Tuple[int, int], Map]: Contains distances between all properties.
            Access via Dict[start][destination].dist
    """

    points, base, start = map_data.properties.values()

    pro_data = {p: dijkstra(dcopy(map_data), p, moves, climb) for p in points}
    pro_data.update({start: aStar(dcopy(map_data), start, base, moves, climb)})
    pro_data.update({base: dijkstra(dcopy(map_data), base, moves, climb)})

    return pro_data


def getPaths(
    pro_data: Dict[Tuple[int, int], Map], pro_order: List[Any]
) -> List[List[Tuple[int, int]]]:
    """Gets routes from ordered coordinates of properties via parent attribute.

    Args:
        pro_data (Dict[Tuple[int, int], Map]): Contains distances between
            all properties. Access via Dict[starting][destination].dist
        pro_order (List[Any]): order of properties' coordinates

    Returns:
        List[List[Tuple[int, int]]]: lists of paths between ordered properties
    """

    paths = []
    for begin, finish in zip(pro_order, pro_order[1:]):
        path = []
        while finish != begin:
            path.append(finish)
            finish = pro_data[begin][finish].parent
        paths.append(path[::-1])

    return paths


def printSolution(paths: List[List[Tuple[int, int]]], distance: int) -> None:
    """Prints the order of paths between properties. Each line starts with
    order number followed by order of tuple coordinates that represent
    the movement progression from start to destination.

    Args:
        paths (List[List[Tuple[int, int]]]): lists of paths between
            ordered properties
        distance (int): total distance of solution
    """

    for i, path in enumerate(paths, 1):
        print(f"{i}: ", *path)

    print("Cost: " + str(distance) + "\n")


def runPathfinding(pars: Dict[str, Any]) -> List[List[Tuple[int, int]]]:
    """Runs pathfinding algorithm on a map that is loaded from the text file.

    Args:
        pars (Dict[str, Any]): parameters:
            fname (string): name of the file to load (without _pro.txt)
            movement (string): "M" - Manhattan, "D" - Diagonal + Manhattan
            climb (bool): Climbing distance approach. If True, distance is
                measured with abs(current terrain number - next terrain number)
            algorithm (string): NP - Naive Permutations, HK - Held Karp
            subset_size (Union[int, None], optional): number of points to visit
                None means all

    Returns:
        List[List[Tuple[int, int]]]: lists of paths between ordered properties
    """

    fname, movement, climb, algorithm, subset_size = pars.values()

    map_data = Map(fname)
    if not map_data.properties:
        print("Invalid map!")
        return []

    moves = getMoves(movement)
    if not moves:
        print("Invalid movement type!")
        return []

    if subset_size is None:
        subset_size = len(map_data.properties["points"])
    elif subset_size < 0:
        print("Invalid subset size!")
        return []

    if subset_size < 2:
        algorithm = "NC"
    if algorithm not in ("NC", "NP", "HK"):
        print("Invalid algorithm input!")
        return []

    findShortestCombo = {"NC": noComb, "NP": naivePermutations, "HK": heldKarp}

    pro_data = findShortestDistances(map_data, moves, climb)
    pro_order, dist = findShortestCombo[algorithm](pro_data, subset_size)
    paths = getPaths(pro_data, pro_order)

    printSolution(paths, dist)

    return paths


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
        facts_amount=points + 1,
        facts_random_order=True,
    )

    evo.runEvolution(evo_parameters)
    runPathfinding(path_parameters)
    chain.runProduction(chain_parameters)

# ToDo: Create validations types for parameters (test)
# ToDo: Create pytest tests
# ToDo: Create performance tests
# ToDo: Create visualizations
