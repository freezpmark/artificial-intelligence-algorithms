"""This module serves to run 2. stage (out of 3) of creating simulation -
Shortest Pathfinding.

Finds the shortest path to visit all nodes. First node to visit is the home
node for which A* algorithm is used. Then, we run Dijkstra's algorithm for each
point node to find the shortest distances to all other point nodes.
To find the shortest path between all point nodes we can use either greedy
Naive permutation or Held-Karp algorithm which is alot faster.

Function hierarchy:
find_shortest_path                  - main function
    _validate_and_set_input_pars    - sets parameters
    _find_shortest_distances        - gets shortest distances between points
        _a_star                     - from start to end
        _dijkstra                   - from start to everywhere
            _passable               - checks whether place on map is passable
            get_next_dist           - gets next distance (climb/no climb)
    _held_karp                      - optimal way to find shortest combo path
    _naive_permutations             - greedy way to find shortest combo path
    _get_routes                     - gets coordinate path via parent nodes
    _print_solution                 - prints the routes - solution
    _save_solution                  - saves the solution into pickle file
"""

import heapq
import pickle
from copy import deepcopy
from itertools import combinations, permutations
from pathlib import Path
from sys import maxsize
from typing import Any, Dict, FrozenSet, List, Tuple


class InvalidParameter(Exception):
    """Parent exception for custom exceptions - parameter errors."""


class MovementError(InvalidParameter):
    """Exception for movement type: only "M" and "D" is allowed."""


class PointsAmountError(InvalidParameter):
    """Exception for points to visit amount: cannot be a negative number."""


class AlgorithmError(InvalidParameter):
    """Exception for algorithm choice: only "NP" and "HK" is allowed."""


class Node:
    """Represents discrete coordinate position on the 2D map.

    This class is being used in Map class. Map is composed of Nodes that
    represent the space of the map.

    Attributes:
        pos (Tuple[int, int]): node's position on the map
        terr (int): terrain (difficulty of traveling through this node)
        parent (Tuple[int, int]): parent position - neighbor position from
            which we moved to this node's position
        dist (int): distance to get to node's position
        g (int): heuristic variable for A* algorithm
            (distance between start and next_node)
        h (int): heuristic variable for A* algorithm
            (distance between next_node and destination)
    """

    __slots__ = ("pos", "terr", "parent", "dist", "g", "h")

    def __init__(self, position, terrain):
        self.pos = position  # type: Tuple[int, int]
        self.terr = terrain  # type: int

        # variables to fill with pathfinding algorithm
        self.parent = -1  # type: Tuple[int, int]
        self.dist = maxsize  # type: int

        # helping A* heuristic variables
        self.g = maxsize  # type: int
        self.h = maxsize  # type: int

    def __lt__(self, other):
        if self.dist != other.dist:
            return self.dist < other.dist
        return self.h < other.h


class Map:
    """Represents 2D map.

    This is the main data structure for navigating in the map. Map is composed
    of nodes attribute that represent the space of the map. They are formed by
    another class Node. Other attributes represent properties of the map.

    Attributes:
        fname (str): name of the file to load
        height (int): represents height size (amount of x coordinates)
        width (int): represents width size (amount of y coordinates)
        properties (Dict[str, Any]): represents special positions
            (points to visit, starting point, home)
        nodes (Dict[Tuple[int, int], Node]): Represent the space of the map
            with keys being coordinates and values being instance of Node class

    Methods:
        load_map(fname: str): loads propertied map from the text file and
            sets up all the attributes
    """

    __slots__ = ("fname", "width", "height", "properties", "nodes", "h")

    def __init__(self, fname: str) -> None:
        self.fname = ""  # type: str
        self.height = 0  # type: int
        self.width = 0  # type: int
        self.properties = {}  # type: Dict[str, Any]
        self.nodes = {}  # type: Dict[Tuple[int, int], Node]
        self.load_map(fname)

    def load_map(self, fname: str) -> None:
        """Loads a map and initializes instance's attributes with it.

        The map is being loaded from /data/maps directory.

        Args:
            fname (str): name of the file that is going to be loaded
        """

        properties = {
            "points": [],
            "home": 0,
            "start": 0,
        }  # type: Dict[str, Any]
        nodes = {}  # type: Dict[Tuple[int, int], Node]

        source_dir = Path(__file__).parents[0]
        fname_path = Path(f"{source_dir}/data/maps/{fname}_pro.txt")
        with open(fname_path, encoding="utf-8") as file:
            for i, row in enumerate(file):
                for j, col in enumerate(row.split()):

                    # if there are brackets, add propertied point
                    if not col.isnumeric():
                        if col.startswith("("):
                            properties["points"].append((i, j))
                        elif col.startswith("["):
                            properties["home"] = (i, j)
                        elif col.startswith("{"):
                            properties["start"] = (i, j)
                        if not col.startswith("-"):
                            col = col[1:-1]  # remove brackets if not neg. num

                    nodes[i, j] = Node((i, j), int(col))

        height, width = max(nodes)
        if all(properties.values()):
            self.fname = fname
            self.height = height + 1
            self.width = width + 1
            self.properties = properties
            self.nodes = nodes

    def __getitem__(self, pos: Tuple[int, int]):
        assert len(pos) == 2, "Coordinate must have two values."
        if not (0 <= pos[0] < self.height and 0 <= pos[1] < self.width):
            return None  # position out of bounds of the map
        return self.nodes[pos]  # Node


def find_shortest_path(
    fname: str,
    movement_type: str,
    climb: bool,
    algorithm: str,
    visit_points_amount: int,
) -> None:
    """Finds shortest visiting path order between all the properties on map.

    At first, it finds the shortest distances between all properties. Then it
    finds the shortest visiting path order between all the properties on the
    map. At the end, it prints the solution into console and saves it into
    pickle file.

    Args:
        fname (string): name of the file to load (with no suffix)
        movement_type (string): determines movement options throughout the map
            Options: "M", "D" (Manhattan or Diagonal + Manhattan)
        climb (bool): determines distance calculcation approach. If True,
            distance is measured as abs(current terrain number - next terrain
            number)
        algorithm (string): determines what algorithm to use to find the
            shortest path
            Options: "NP", "HK" (Naive Permutations or Held Karp)
        visit_points_amount (int): Amount of points to visit.
            0 means all. Must be at least 1.
    """

    alg_opts = {
        "NP": _naive_permutations,
        "HK": _held_karp,
    }

    try:
        map_ = Map(fname)
        moves, visit_points_amount, algorithm = _validate_and_set_input_pars(
            movement_type,
            algorithm,
            visit_points_amount,
            map_.properties["points"],
        )

        disted_map = _find_shortest_distances(map_, moves, climb)
        node_paths, dist = alg_opts[algorithm](disted_map, visit_points_amount)
        routed_paths = _get_routes(disted_map, node_paths)

        _print_solution(routed_paths, dist)
        _save_solution(routed_paths, fname)

    except (FileNotFoundError, InvalidParameter) as err:
        if isinstance(err, AlgorithmError):
            error = f"{err}: Custom AlgorithmError"
        elif isinstance(err, MovementError):
            error = f"{err}: Custom MovementError"
        elif isinstance(err, PointsAmountError):
            error = f"{err}: Custom PointsAmountError"

        elif isinstance(err, FileNotFoundError):
            error = f"{err}: Built-in FileNotFoundError"

        # by grouping the exceptions and using isintance, we don't have to
        # repeat the code below for all cases
        error += "exception was raised!"
        print(error)
        raise  # raises the catched exception


def _validate_and_set_input_pars(
    movement_type: str,
    algorithm: str,
    visit_points_amount: int,
    map_points: List[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], int, str]:
    """Validates and sets parameters from input for pathfinding.

    Gets movement possibilities from either Manhattan or Diagonal + Manhattan
    approach.
    Validates and sets amount of points to visit. If its None or higher than
    the amount of points on the map, it will be reduced down to the map's
    amount.
    Validates algorithm option.

    Args:
        movement_type (str): determines movement options throughout the map
            Options: "M", "D" (Manhattan or Diagonal + Manhattan)
        algorithm (str): determines what algorithm to use to find the
            shortest path
            Options: "NP", "HK" (Naive Permutations or Held Karp)
        visit_points_amount (int): Amount of points to visit.
            0 means all. Must be at least 1.
        map_points (List[Tuple[int, int]]): coordinates of points on the map

    Raises:
        MovementError: movement_type is not "M" or "D"
        PointsAmountError: visit_points_amount is negative number
        AlgorithmError: wrong algorithm abbreviation string (NP or HK only!)

    Returns:
        Tuple[List[Tuple[int, int]], int, str]: (
            movement coordinate options
            amount of points to visit
            algorithm that is going to be used
        )
    """

    # validate and get movement possibilities
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if movement_type == "D":
        moves += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    elif movement_type != "M":
        raise MovementError("Invalid movement type!")

    # validate and set visit_points_amount
    if not visit_points_amount or visit_points_amount > len(map_points):
        visit_points_amount = len(map_points)
    if visit_points_amount < 0:
        raise PointsAmountError("Invalid subset size!")

    # validate algorithm
    if algorithm not in ("NP", "HK"):
        raise AlgorithmError("Invalid algorithm input!")

    return moves, visit_points_amount, algorithm


def _find_shortest_distances(
    map_: Map, moves: List[Tuple[int, int]], climb: bool
) -> Map:
    """Finds shortest distances between all properties.

    Uses A* algorithm from start, from all the other points Dijkstra is used.
    It basically fills up the nodes attribute's (parent, distance) in the map.

    Args:
        map_ (Map): contains information about the map
        moves (List[Tuple[int, int]]): movement coordinate options
        climb (bool): determines distance calculcation approach. If True,
            distance is measured as abs(current terrain number - next terrain
            number), otherwise it is just (next terrain number)

    Returns:
        Map: Map that contains information about the shortest distances between
            all properties in nodes' attributes. (parent, distance)
            (Access via Dict[start][destination].dist)
    """

    points, home, start = map_.properties.values()

    from_start_to_home = _a_star(map_, moves, climb, start=start, dest=home)
    from_home_to_all = _dijkstra(map_, moves, climb, start=home)
    from_points_to_all = {
        point: _dijkstra(map_, moves, climb, start=point) for point in points
    }

    shortest_distance_nodes_between_properties = {
        start: from_start_to_home,
        home: from_home_to_all,
        **from_points_to_all,
    }

    map_.nodes = shortest_distance_nodes_between_properties

    return map_


def _a_star(
    map_: Map,
    moves: List[Tuple[int, int]],
    climb: bool,
    start: Tuple[int, int],
    dest: Tuple[int, int],
) -> Dict[Tuple[int, int], Node]:
    """Finds the shortest path from start to destination and saves it into map.

    Args:
        map_ (Map): contains information about the map
        moves (List[Tuple[int, int]]): movement coordinate options
        climb (bool): determines distance calculcation approach. If True,
            distance is measured as abs(current terrain number - next terrain
            number), otherwise it is just (next terrain number)
        start (Tuple[int, int]): starting position
        dest (Tuple[int, int]): ending position

    Returns:
        Dict[Tuple[int, int], Node]: contains path (via parent attribute) and
            distance from start to destination
    """

    nodes = deepcopy(map_.nodes)
    nodes[start].g = 0
    open_list = []  # type: List[Node]
    close_list = []
    heapq.heappush(open_list, nodes[start])

    while open_list:
        node = heapq.heappop(open_list)
        if node.pos == dest:
            break

        close_list.append(node.pos)
        for move in moves:
            next_pos = node.pos[0] + move[0], node.pos[1] + move[1]
            if _passable(next_pos, map_) and next_pos not in close_list:
                next_node = nodes[next_pos]

                # heuristic - distance between next_node and destination
                x_diff = abs(next_node.pos[0] - dest[0])
                y_diff = abs(next_node.pos[1] - dest[1])
                h = x_diff + y_diff

                # distance between start and next_node
                step_dist = get_next_dist(node.terr, next_node.terr, climb)
                g = node.g + step_dist

                f = g + h  # estimated distance between start and destination
                if f < next_node.dist:
                    next_node.g = g
                    next_node.h = h
                    next_node.dist = f
                    next_node.parent = node.pos
                if next_node not in open_list:
                    heapq.heappush(open_list, next_node)

    return nodes


def _dijkstra(
    map_: Map,
    moves: List[Tuple[int, int]],
    climb: bool,
    start: Tuple[int, int],
) -> Dict[Tuple[int, int], Node]:
    """Finds all the shortest paths from start and saves them into map.

    Args:
        map_ (Map): contains information about the map
        moves (List[Tuple[int, int]]): movement coordinate options
        climb (bool): determines distance calculcation approach. If True,
            distance is measured as abs(current terrain number - next terrain
            number), otherwise it is just (next terrain number)
        start (Tuple[int, int]): starting position

    Returns:
        Dict[Tuple[int, int], Node]: contains path (via parent attribute) and
            distance from start to all points
    """

    nodes = deepcopy(map_.nodes)
    nodes[start].dist = 0
    heap = []  # type: List[Node]
    heapq.heappush(heap, nodes[start])

    while heap:
        node = heapq.heappop(heap)
        for move in moves:
            next_pos = node.pos[0] + move[0], node.pos[1] + move[1]
            if _passable(next_pos, map_):
                next_node = nodes[next_pos]
                step_dist = get_next_dist(node.terr, next_node.terr, climb)
                next_node_dist = node.dist + step_dist

                if next_node.dist > next_node_dist:
                    next_node.dist = next_node_dist
                    next_node.parent = node.pos
                    heapq.heappush(heap, next_node)

    return nodes


def _passable(next_pos: Tuple[int, int], map_: Map):
    """Checks whether next_pos position is passable (not walled or out of map).

    Args:
        next_pos (Tuple[int, int]): position on which movement was applied
        map_ (Map): contains information about the map

    Returns:
        bool: passability
    """

    valid_pos = map_[next_pos]
    if valid_pos:
        valid_pos = not valid_pos.terr < 0

    return valid_pos


def get_next_dist(prev_terr: int, next_terr: int, climb: bool) -> int:
    """Gets next distance based on whether its climbing approach or not.

    Args:
        prev_terr (int): terrain of position from which we move
        next_terr (int): terrain of position to which we move
        climb (bool): determines distance calculcation approach. If True,
            distance is measured as abs(current terrain number - next terrain
            number), otherwise it is just (next terrain number)

    Returns:
        int: distance to the next position
    """

    if climb:
        return abs(prev_terr - next_terr) + 1
    else:
        return next_terr


def _held_karp(
    map_: Map, visit_points_amount: int
) -> Tuple[List[Tuple[int, int]], int]:
    """Finds the shortest visiting path order between all properties on map.

    For finding the solution, Held Karp algorithm is used.

    Args:
        map_ (Map): Map that contains information about the shortest distances
            between all properties in nodes' attributes.
            (Access via Dict[start][destination].dist)
        visit_points_amount (int): amount of points to visit

    Returns:
        Tuple[List[Tuple[int, int]], int]: (
            shortest visiting path order of properties,
            distance of the path
        )
    """

    points, home, start = map_.properties.values()
    points_set = frozenset(points)

    coor_and_comb = Tuple[Tuple[int, int], FrozenSet[int]]
    cost_and_parent_coor = Tuple[int, Tuple[int, int]]
    nodes: Dict[coor_and_comb, cost_and_parent_coor] = {}

    for comb_size in range(visit_points_amount):
        for comb in combinations(points_set, comb_size):
            comb_set = frozenset(comb)
            points_to_visit = points_set - comb_set
            for dest in points_to_visit:
                routes = []
                if comb_set:
                    for begin in comb_set:
                        sub_comb = comb_set - frozenset({begin})
                        prev_cost = nodes[begin, sub_comb][0]
                        cost = map_[begin][dest].dist + prev_cost
                        routes.append((cost, begin))
                    nodes[dest, comb_set] = min(routes)

                else:  # first visit (start -> home)
                    cost = map_[home][dest].dist + map_[start][home].dist
                    nodes[dest, frozenset()] = cost, home

    # get total cost, ending node and its parent
    last_nodes_raw = list(nodes.items())[-len(points_set) :]
    last_nodes = [(*node[1], node[0][0]) for node in last_nodes_raw]
    last_optimal_node = min(last_nodes)
    cost, parent, end = last_optimal_node
    points_set -= {end}
    path = [end]

    # backtrack remaining nodes via parents
    for _ in range(visit_points_amount - 1):
        path.append(parent)
        points_set -= {parent}
        parent = nodes[parent, points_set][1]
    path.extend([home, start])

    return path[::-1], cost


def _naive_permutations(
    map_: Map, visit_points_amount: int
) -> Tuple[List[Tuple[int, int]], int]:
    """Finds the shortest visiting path order between all properties on map.

    Computes all possible permutations to find the shortest path order.

    Args:
        map_ (Map): Map that contains information about the shortest distances
            between all properties in nodes' attributes.
            (Access via Dict[start][destination].dist)
        visit_points_amount (int): amount of points to visit

    Returns:
        Tuple[List[Tuple[int, int]], int]: (
            shortest visiting path order of properties,
            distance of the path
        )
    """

    points, home, start = map_.properties.values()
    total_cost = maxsize

    for permutation in permutations(points, visit_points_amount):
        distance = map_[start][home].dist
        for begin, finish in zip((home,) + permutation, permutation):
            distance += map_[begin][finish].dist
        if distance < total_cost:
            total_cost = distance
            permutation_path = permutation

    return list((start, home) + permutation_path), total_cost


def _get_routes(
    map_: Map, node_paths: List[Tuple[int, int]]
) -> List[List[Tuple[int, int]]]:
    """Gets step by step coordinate routes from node paths via parent.

    Args:
        map_ (Map): Map that contains information about the shortest distances
            between all properties in nodes attribute.
            (Access via Dict[start][destination].dist)
        node_paths (List[Tuple[int, int]]): shortest visiting path order of
            properties

    Returns:
        List[List[Tuple[int, int]]]: coordinate routed paths
    """

    paths = []

    for begin, route in zip(node_paths, node_paths[1:]):
        path = []
        while route != begin:
            path.append(route)
            route = map_[begin][route].parent
        paths.append(path[::-1])

    return paths


def _print_solution(
    routed_paths: List[List[Tuple[int, int]]], dist: int
) -> None:
    """Prints the routed paths into the console.

    Each line starts with order number followed by order of tuple coordinates
    that represent the movement progression from start to destination.

    Args:
        routed_paths (List[List[Tuple[int, int]]]): routed paths (solution)
        dist (int): total distance
    """

    for i, routed_path in enumerate(routed_paths, 1):
        print(f"{i}: ", *routed_path)

    print("Cost: " + str(dist) + "\n")


def _save_solution(
    routed_paths: List[List[Tuple[int, int]]], fname: str
) -> None:
    """Saves the routed paths into pickle file for gif visualization.

    Saves the solution into /data/solutions directory. If the directory does
    not exist, it will create one.

    Args:
        routed_paths (List[List[Tuple[int, int]]]): routed paths (solution)
        fname (str): name of pickle file into which solution will be saved
    """

    source_dir = Path(__file__).parents[0]
    solutions_dir = Path(f"{source_dir}/data/solutions")
    solutions_dir.mkdir(parents=True, exist_ok=True)

    fname_path = Path(f"{solutions_dir}/{fname}_path")
    with open(fname_path, "wb") as file:
        pickle.dump(routed_paths, file)


if __name__ == "__main__":

    FNAME = "queried"
    MOVEMENT_TYPE = "M"
    CLIMB = False
    ALGORITHM = "HK"
    VISIT_POINTS_AMOUNT = 10

    path_parameters = dict(
        fname=FNAME,
        movement_type=MOVEMENT_TYPE,
        climb=CLIMB,
        algorithm=ALGORITHM,
        visit_points_amount=VISIT_POINTS_AMOUNT,
    )  # type: Dict[str, Any]

    find_shortest_path(**path_parameters)
