"""This module serves to run 1. stage (out of 3) of creating simulation -
Map creating with evolutionary algorithm.

The task is to rake the sand on the entire Zen garden. Character always starts
at the edge of the garden and leaves straight shaping path until it meets
another edge or obstacle. On the edge, character can walk as he pleases.
If it comes to an obstacle - wall or raked sand - he has to turn around,
if he is not able to, raking is over.

Function hierarchy:
create_maps                         - main function
    save_map                        - saving decorator for create_* functions
    create_walls                    - creates walls from query into file
    create_terrain                  - creates terrain from walls into file
        _evolutionize               - runs evolutionary algorithm
            _rake_map               - raking the map with one solution each
                _get_start_pos      - gets starting position from gene
                _in_bounds          - checks whether we are out of map
                _get_row_movement   - get new row movement (upon collision)
                _get_col_movement   - get new column movement (upon collision)
            _calculate_fitness      - evaluate solution
            _create_next_generation - creating next generation
            _fill_map               - fills walled map with terrain (solution)
        _save_solution              - save solution for animation
    create_properties               - creating properties
        _save_map                   - saving map (walls/terrain/properties)
        load_map                    - loading map (terrain/properties)
        _generate_properties        - generates properties
            _free_position_finder   - finds free position for safe generation
"""

import functools
import pickle
import random
import re
from copy import copy
from pathlib import Path
from time import time
from typing import Any, Dict, Generator, List, Tuple

CHROMOSOMES = 30
GENERATIONS = 100
MIN_MUT_RATE = 0.05
MAX_MUT_RATE = 0.80
CROSS_RATE = 0.90


class QueryError(Exception):
    """Exception for walled map inquiry: wrong size or rocks positioning."""


def create_maps(
    fname: str,
    begin_from: str,
    query: str,
    max_runs: int,
    points_amount: int,
) -> None:
    """Creates and saves walled, terrained and propertied maps into text files.

    There are three dependant stages of creating a complete map. Each stage
    uses map that was created by the previous stage. We can therefore start
    from any stage as long as the previous stage one was run. (except for the
    first one). All remaining stages will run after that.
    It also saves solutions for gif visualization.

    Args:
        fname (str): name of file(s) that is/are going to be created
        begin_from (str): Defines which stage to start from.
            Options: "walls", "terrain", "properties"
        query (str): Query through which walled map is going to be created.
            Contains size of map and coordinates of walls.
            Option: "10x12 (1,5) (2,1) (3,4) (4,2) (6,8) (6,9)"
        max_runs (int): max number of attempts to find solution with evo. alg.
        points_amount (int): amount of destination points to visit (another
            module's task to process)
    """

    next_ = False
    try:
        if begin_from == "walls":
            create_walls(fname, query)
            next_ = True
        if begin_from == "terrain" or next_:
            create_terrain(fname, max_runs, display=True)
            next_ = True
        if begin_from == "properties" or next_:
            create_properties(fname, points_amount)
    except (QueryError, FileNotFoundError) as err:
        print(err)


def save_map(suffix_after: str):
    """Decorator that ensures saving of the map into the file.

    Args:
        suffix_after (str): Suffix of fname.
            Options: "_wal", "_ter", "_pro";
    """

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            # cant make load_map decorator because it will call the f. 2 times
            # using results from the function is not ideal,
            # but just made this to show how decorators work!
            save_args = function(*args, **kwargs)
            _save_map(save_args[0] + suffix_after, *save_args[1:])

        return wrapper

    return decorator


@save_map("_wal")
def create_walls(
    fname: str, query: str, display: bool = False
) -> Tuple[str, List[List[str]], bool]:
    """Creates walled map from query and saves it into a file.

    Map is filled with "1" being walls and "0" being walkable space.
    Return values are used for the decorator.

    Args:
        fname (str): name of the file that is going to be created
        query (str): Query through which walled map is going to be created.
            Contains size of map and coordinates of walls.
            Option: "10x12 (1,5) (2,1) (3,4) (4,2) (6,8) (6,9)"
        display (bool, optional): Option to print created walls into console.
            Defaults to False.

    Raises:
        QueryError: query does not match regular expression pattern

    Returns:
        Tuple[str, List[List[str]], bool]: (
            name of the file that is going to be created,
            2D map that will be saved into file,
            option to print created walls into console
        )
    """

    walled_map = []
    if re.search(r"[0-9]+x[0-9]+(\ \([0-9]+,[0-9]+\))+$", query):
        query_split = query.split()
        row_amount, col_amount = map(int, query_split[0].split("x"))

        walled_map = [["0"] * col_amount for _ in range(row_amount)]
        wall_query = query_split[1:]
        for wall_raw_coordinate in wall_query:
            wall_coordinate = wall_raw_coordinate[1:-1].split(",")
            i, j = map(int, wall_coordinate)
            try:
                walled_map[i][j] = "1"
            except IndexError:
                walled_map = []

    if not walled_map:
        raise QueryError("Invalid query!")

    return fname, walled_map, display


@save_map("_ter")
def create_terrain(
    fname: str, max_runs: int, display: bool = False
) -> Tuple[str, List[List[str]], bool]:
    """Creates terrained map from walled map and saves it into a file.

    Map is filled with "-1" being walls and walkable places are filled with
    various numbers generated by the evolutionary algorithm. In the case of not
    finding the perfect solution, unterrained places will be indicated by "-2"
    numbers that are considered unwalkable like walls.
    It also saves raking paths for gif visualization.
    Return values are used for the decorator.

    Args:
        fname (str): name of the file that is going to be created
        max_runs (int): max number of attempts to find solution with evo. alg.
        display (bool, optional): Option to print created terrain into console.
            Defaults to False.

    Raises:
        FileNotFoundError: file does not exist

    Returns:
        Tuple[str, List[List[str]], bool]: (
            name of the file that is going to be created,
            2D map that will be saved into file,
            option to print created walls into console
        )
    """

    walled_map = load_map(fname, "_wal")
    terrained_map, rake_paths = _evolutionize(walled_map, max_runs, display)
    _save_solution(rake_paths, fname)

    return fname, terrained_map, display


@save_map("_pro")
def create_properties(
    fname: str,
    points_amount: int,
    display: bool = False,
) -> Tuple[str, List[List[str]], bool, str]:
    """Creates propertied map from terrained map and saves it into a file.

    Return values are used for the decorator.

    Args:
        fname (str): name of the file that is going to be created
        points_amount (int): amount of destination points to visit (another
            module's task to process)
        display (bool, optional): Option to print created properties into
            console. Defaults to False.

    Raises:
        FileNotFoundError: file does not exist

    Returns:
        Tuple[str, List[List[str]], bool, str]: (
            name of the file that is going to be created,
            2D map that will be saved into file,
            option to print created walls into console,
            spacing format between coordinates
        )
    """

    terrained_map = load_map(fname, "_ter")
    propertied_map = _generate_properties(terrained_map, points_amount)

    return fname, propertied_map, display, "{:^5}"


def _save_map(
    fname: str,
    map_2d: List[List[str]],
    display: bool = False,
    spacing: str = "{:^3}",
) -> None:
    """Saves a map from 2D list into file.

    Saves the map into /data/maps directory. If the directory does not exist,
    it will create one. This function is being used in save_map decorator.

    Args:
        fname (str): name of the file into which the map is going to be saved
        map_2d (List[List[str]]): 2D map that will be saved into file
        display (bool, optional): Option to print saved map into console.
            Defaults to False.
        spacing (str, optional): Spacing between values that are in the map.
            Defaults to "{:^3}".
    """

    src_dir = Path(__file__).parents[0]
    map_dir = Path(f"{src_dir}/data/maps")
    map_dir.mkdir(parents=True, exist_ok=True)

    fname_path = Path(f"{map_dir}/{fname}.txt")
    with open(fname_path, "w", encoding="utf-8") as file:
        for i, row in enumerate(map_2d):
            for j in range(len(row)):
                file.write(spacing.format(map_2d[i][j]))
                if display:
                    print(spacing.format(map_2d[i][j]), end=" ")
            file.write("\n")
            if display:
                print()


def load_map(fname: str, suffix: str) -> List[List[str]]:
    """Loads a map from file into 2D map list.

    The map is being loaded from /data/maps directory.

    Args:
        fname (str): name of the file to load
        suffix (str): Suffix of fname.
            Options: "_wal", "_ter", "_pro"

    Returns:
        List[List[str]]: 2D map loaded from file
    """

    src_dir = Path(__file__).parents[0]
    fname_path = Path(f"{src_dir}/data/maps/{fname}{suffix}.txt")
    map_ = []

    try:
        with open(fname_path, encoding="utf-8") as file:
            line = file.readline().rstrip()
            map_.append(line.split())
            prev_length = len(line)

            for line in file:
                line = line.rstrip()
                if prev_length != len(line):
                    map_ = []
                    break
                map_.append(line.split())
                prev_length = len(line)
    except FileNotFoundError:
        print("Invalid file name!")
        exit()

    return map_


def _evolutionize(
    map_2d: List[List[str]], max_runs: int, print_stats: bool = True
) -> Tuple[List[List[str]], Dict[Tuple[int, int], int]]:
    """Runs evolutionary algorithm on 2D walled map to cover it with terrain.

    Args:
        map_2d (List[List[str]]): 2D walled map that will be terrained
        max_runs (int): max number of attempts to find solution with evo. alg.
        print_stats (bool, optional): Turns on debug mode that prints stats
            and solution into console. Defaults to True.

    Returns:
        Tuple[List[List[str]], Dict[Tuple[int, int], int], str]: (
            2D map filled with terrain (wall being -1, unraked being -2),
            raking paths that will be used for gif visualization
        )
    """

    def print_better_pop():
        """Prints information about better newly found solution"""

        print(f"Generation: {i+1},", end="\t")
        print(f"Raked: {fit_max} (out of {to_rake_amount})", end="\t")
        print(f"Mutation rate: {round(mut_rate, 2)}")

    def print_final_stats():
        """Prints final stats, solution and map"""

        total = round(time() - start_time, 2)
        avg = round(sum(generation_times) / len(generation_times), 2)
        chromo = " ".join(map(str, population[fit_max_index]))
        result_msg = f"Solution {'' if found_solution else 'not '}found!"
        print(result_msg)
        print(f"Total time elapsed is {total}s,", end="\t")
        print(f"each generation took {avg}s in average.")
        print(f"Chromosome: {chromo}")

    found_solution = False
    attempt_number = 1

    while not found_solution and attempt_number <= max_runs:
        print(f"\nAttempt number {attempt_number}.\n{'-' * 60}")

        # simplify 2D map into 1D dict (it is faster in raking)
        map_tuple = {
            (i, j): -int(col)
            for i, row in enumerate(map_2d)
            for j, col in enumerate(row)
        }

        rows, cols = len(map_2d), len(map_2d[0])
        rocks_amount = abs(sum(map_tuple.values()))
        to_rake_amount = rows * cols - rocks_amount
        map_perimeter = (rows + cols) * 2

        # generating chromosome for first generation/population
        population = []
        genes = random.sample(range(map_perimeter - 1), map_perimeter - 1)
        for _ in range(CHROMOSOMES):  # chromosome - solution defined by genes
            random.shuffle(genes)
            chromosome = [num * random.choice([-1, 1]) for num in genes]
            population.append(chromosome)

        start_time = time()
        generation_times = []

        # loop over generations - evolution
        prev_max = 0
        mut_rate = MIN_MUT_RATE
        for i in range(GENERATIONS):  # generation - set of all chromosomes
            generation_time = time()

            # evaluate all chromosomes and save the best one
            fit_vals = []
            fit_max = 0
            fit_max_index = 0
            for j in range(CHROMOSOMES):
                map_tuple_filled, raking_paths = _rake_map(
                    population[j], copy(map_tuple), rows, cols
                )
                fit_val = _calculate_fitness(map_tuple_filled, to_rake_amount)
                fit_vals.append(fit_val)
                if fit_val > fit_max:
                    fit_max = fit_val
                    fit_max_map = map_tuple_filled
                    fit_max_path = raking_paths
                    fit_max_index = j

            if prev_max < fit_max and print_stats:
                print_better_pop()

            # found solution
            if fit_max == to_rake_amount:
                found_solution = True
                generation_times.append(time() - generation_time)
                break

            # increase mutation rate generation to prevent local maximum
            mut_rate = (
                mut_rate + 0.01 if mut_rate < MAX_MUT_RATE else MAX_MUT_RATE
            )

            # create new chromosome for the next generation
            children = _create_next_generation(
                map_perimeter, mut_rate, population, fit_vals
            )

            # keep the best chromosome for next generation
            for i in range(map_perimeter - 1):
                children[0][i] = population[fit_max_index][i]

            population = children
            prev_max = fit_max
            generation_times.append(time() - generation_time)

        if print_stats:
            print_final_stats()
        attempt_number += 1

    map_2d_filled = _fill_map(map_2d, fit_max_map)

    return map_2d_filled, fit_max_path


def _rake_map(
    chromosome: List[int],
    map_tuple: Dict[Tuple[int, int], int],
    rows: int,
    cols: int,
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
    """Rakes the map by chromosome to fill it with terrain.

    Chromosome consists of instructions known as genes. Each gene defines
    the starting position and movement direction choice upon collision with.

    Args:
        chromosome (List[int]): ordered set of genes (instructions)
        map_tuple (Dict[Tuple[int, int], int]): map defined by dict with:
            keys being tuples as coordinates,
            value being values of terrain (0 is unraked)
        rows (int): amount of rows in the map
        cols (int): amount of cols in the map

    Returns:
        Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]: (
            map tuple filled with terrain (map_tuple -> map_tuple_filled),
            raking paths that will be used for gif visualization
        )
    """

    rake_paths = {}  # type: Dict[Tuple[int, int], int]
    terrain_num = 1  # first raking number

    for gene in chromosome:
        pos, move = _get_start_pos(gene, rows, cols)

        # collision check (to rock or raked sand)
        if map_tuple[pos]:
            continue

        parents = {}  # type: Dict[Any, Any]
        parent = (-1, -1)  # type: Tuple[int, int]
        # (-1, -1) (indicates no parrent)
        while _in_bounds(pos, rows, cols):
            if map_tuple[pos]:
                pos = parent
                parent = parents[pos]

                # change moving direction
                if move[0]:
                    move = _get_row_movement(pos, cols, map_tuple, gene)
                else:
                    move = _get_col_movement(pos, rows, map_tuple, gene)

                # cant change direction - remove the path
                if not any(move):
                    terrain_num -= 1
                    while parents[pos] != (-1, -1):
                        map_tuple[pos] = 0
                        pos = parents[pos]
                    map_tuple[pos] = 0
                    break

                # can change direction to just move out of the map
                if all(move):
                    break

            # move to the next pos
            map_tuple[pos] = terrain_num
            parents[pos] = parent
            parent = pos
            pos = pos[0] + move[0], pos[1] + move[1]

        # save paths for gif visualization
        if any(move):
            rake_path = {key: terrain_num for key in parents}
            rake_paths = {**rake_paths, **rake_path}

        terrain_num += 1

    return map_tuple, rake_paths


def _get_start_pos(
    gene: int, rows: int, cols: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Gets starting position and movement direction.

    Args:
        gene (int): instruction that defines the starting position and movement
            (number between (0) and (perimeter -1))
        rows (int): amount of rows in the map
        cols (int): amount of cols in the map

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: (
            starting position coordinate,
            movement direction coordinate
        )
    """

    half_perimeter = rows + cols

    pos_num = abs(gene)
    if pos_num < cols:  # go DOWN
        pos, move = (0, pos_num), (1, 0)
    elif pos_num < half_perimeter:  # go RIGHT
        pos, move = (pos_num - cols, 0), (0, 1)
    elif pos_num < half_perimeter + rows:  # go LEFT
        pos, move = (pos_num - half_perimeter, cols - 1), (0, -1)
    else:  # go UP
        pos, move = (
            (rows - 1, pos_num - half_perimeter - rows),
            (-1, 0),
        )

    return pos, move


def _in_bounds(pos: Tuple[int, int], rows: int, cols: int) -> bool:
    """Checks whether current position is not out of bounds of the map.

    Args:
        pos (Tuple[int, int]): current position we are at
        rows (int): amount of rows in the map
        cols (int): amount of cols in the map

    Returns:
        bool: indicates raking availability
    """

    return 0 <= pos[0] < rows and 0 <= pos[1] < cols


def _get_row_movement(
    pos: Tuple[int, int],
    cols: int,
    map_tuple: Dict[Tuple[int, int], int],
    gene: int,
) -> Tuple[int, int]:
    """Gets next movement coordinate that changes positions between rows.

    Args:
        pos (Tuple[int, int]): current position we are at
        cols (int): amount of cols in the map
        map_tuple (Dict[Tuple[int, int], int]): map defined by dict with:
            keys being tuples as coordinates,
            value being values of terrain (0 is unraked)
        gene (int): determines which movement to take if there are two options
            upon collision

    Returns:
        Tuple[int, int]: movement direction coordinate
    """

    right = pos[0], pos[1] + 1
    left = pos[0], pos[1] - 1
    right_inbound = right[1] < cols
    left_inbound = left[1] >= 0
    right_free = right_inbound and not map_tuple[right]
    left_free = left_inbound and not map_tuple[left]

    # if both ways are free, the gene will decide where to go
    if right_free and left_free:
        move = (0, 1) if gene > 0 else (0, -1)

    elif right_free:
        move = 0, 1
    elif left_free:
        move = 0, -1

    # we are in bounds of the map, but we cannot move anywhere
    elif right_inbound and left_inbound:
        move = 0, 0

    # one movement side is out of bounds
    else:
        move = 1, 1  # indicates the possibility of leaving the map

    return move


def _get_col_movement(
    pos: Tuple[int, int],
    rows: int,
    map_tuple: Dict[Tuple[int, int], int],
    gene: int,
) -> Tuple[int, int]:
    """Gets next movement coordinate that changes positions between columns.

    Args:
        pos (Tuple[int, int]): current position we are at
        rows (int): amount of rows in the map
        map_tuple (Dict[Tuple[int, int], int]): map defined by dict with:
            keys being tuples as coordinates,
            value being values of terrain (0 is unraked)
        gene (int): determines which movement to take if there are two options
            upon collision

    Returns:
        Tuple[int, int]: movement direction coordinate
    """

    down = pos[0] + 1, pos[1]
    upp = pos[0] - 1, pos[1]
    down_inbound = down[0] < rows
    up_inbound = upp[0] >= 0
    down_free = down_inbound and not map_tuple[down]
    up_free = up_inbound and not map_tuple[upp]

    if down_free and up_free:
        move = (1, 0) if gene > 0 else (-1, 0)
    elif down_free:
        move = 1, 0
    elif up_free:
        move = -1, 0
    elif down_inbound and up_inbound:
        move = 0, 0
    else:
        move = 1, 1

    return move


def _calculate_fitness(
    map_tuple_filled: Dict[Tuple[int, int], int], to_rake_amount: int
) -> int:
    """Evaluates the solution by counting spots that were filled/raked.

    Args:
        map_tuple_filled (Dict[Tuple[int, int], int]): 1D terrained map
        to_rake_amount (int): amount of rakeable spots

    Returns:
        int: count of raked spots
    """

    unraked_count = list(map_tuple_filled.values()).count(0)
    raked_count = to_rake_amount - unraked_count

    return raked_count


def _create_next_generation(
    map_perimeter: int,
    mut_rate: float,
    population: List[List[int]],
    fit_vals: List[int],
) -> List[List[int]]:
    """Generates new generation using genetic algorithm.

    Args:
        map_perimeter (int): perimeter of the map
        mut_rate (float): mutation rate - probability of mutating with
            uniform crossover
        population (List[List[int]]): old generation
        fit_vals (List[int]): fitness values for picking better candidates

    Returns:
        List[List[int]]: new generation/chromosome - children
    """

    children = []  # type: List[Any]
    for i in range(CHROMOSOMES):

        # pick a winning chromosomes out of 2
        rand1, rand2 = random.sample(range(CHROMOSOMES), 2)
        win = rand1 if fit_vals[rand1] > fit_vals[rand2] else rand2

        # copying winning chromosome into children
        children.append([])
        for j in range(map_perimeter - 1):
            children[i].append(population[win][j])

        if random.random() > CROSS_RATE:
            continue

        # mutating chromosome with uniform crossover
        # (both inherit the same amount of genetic info)
        for rand_index in range(map_perimeter - 1):
            if random.random() < mut_rate:

                # create random gene and search for it in children
                rand_val = random.randint(0, map_perimeter - 1)
                rand_val *= random.choice([-1, 1])
                found_index = False
                if rand_val in children[i]:
                    found_index = children[i].index(rand_val)

                if found_index:  # swap it with g gene if it was found
                    tmp = children[i][rand_index]
                    children[i][rand_index] = children[i][found_index]
                    children[i][found_index] = tmp
                else:  # replace it with rand_val
                    children[i][rand_index] = rand_val

    return children


def _fill_map(
    map_2d: List[List[str]], map_tuple_filled: Dict[Tuple[int, int], int]
) -> List[List[str]]:
    """Fills walled map with terrain generated by the evolution algorithm.

    Replaces zeros with -2 (unraked spots).

    Args:
        map_2d (List[List[str]]): 2D walled map that will be terrained by
            map_tuple_filled
        map_tuple_filled (Dict[Tuple[int, int], int]): 1D terrained map

    Returns:
        List[List[str]]: 2D terrained map (map_2d -> map_2d_filled)
    """

    for (i, j), terrain_num in map_tuple_filled.items():
        terrain_str = str(terrain_num) if terrain_num else "-2"
        map_2d[i][j] = terrain_str

    return map_2d


def _save_solution(rake_paths: Dict[Tuple[int, int], int], fname: str) -> None:
    """Saves solution - raking paths into pickle file for gif visualization.

    Saves the solution into /data/solutions directory. If the directory does
    not exist, it will create one. This function is being used in save_map
    decorator.

    Args:
        rake_paths (Dict[Tuple[int, int], int]): raking paths that will be used
            for gif visualization
        fname (str): name of pickle file into which the solution will be saved
    """

    src_dir = Path(__file__).parents[0]
    solutions_dir = Path(f"{src_dir}/data/solutions")
    Path(solutions_dir).mkdir(parents=True, exist_ok=True)

    fname_path = Path(f"{solutions_dir}/{fname}_rake.pickle")
    with open(fname_path, "wb") as file:
        pickle.dump(rake_paths, file)


def _generate_properties(
    map_2d: List[List[str]], points_amount: int
) -> List[List[str]]:
    """Adds properties to terrained map.

    Properties are represented with a bracket around the number of terrain.
    {} - starting position, [] - first position to visit, () - point to visit.

    Args:
        map_2d (List[List[str]]): 2D terrained map that will be propertied
        points_amount (int): amount of destination points to visit

    Returns:
        List[List[str]]: 2D propertied map
    """

    def _free_position_finder(
        terrained_map: List[List[str]],
    ) -> Generator[Tuple[int, int], None, None]:
        """Finder of free positions for properties.

        Args:
            terrained_map (List[List[str]]): 2D terrained map

        Yields:
            Generator[Tuple[int, int], None, None]: coordinate of free position
        """

        reserved = set()
        for i, row in enumerate(terrained_map):
            for j, col in enumerate(row):
                if int(col) < 0:
                    reserved.add((i, j))

        while True:
            i = random.randint(0, len(terrained_map) - 1)
            j = random.randint(0, len(terrained_map[0]) - 1)
            if (i, j) not in reserved:
                reserved.add((i, j))
                yield (i, j)

    free_pos = _free_position_finder(map_2d)  # closure (None, None?)

    i, j = next(free_pos)
    map_2d[i][j] = "[" + map_2d[i][j] + "]"

    i, j = next(free_pos)
    map_2d[i][j] = "{" + map_2d[i][j] + "}"

    for _ in range(points_amount):
        i, j = next(free_pos)
        map_2d[i][j] = "(" + map_2d[i][j] + ")"

    return map_2d


if __name__ == "__main__":

    # walls uses:   query, fname, max_runs, points_amount
    # terrain uses:        fname, max_runs, points_amount, evo. consts
    # properties uses:     fname,           points_amount
    BEGIN_FROM = "walls"
    QUERY = "10x12 (1,5) (2,1) (3,4) (4,2) (6,8) (6,9) (6,9)"
    FNAME = "queried"
    MAX_RUNS = 3
    POINTS_AMOUNT = 10

    evo_parameters = dict(
        begin_from=BEGIN_FROM,
        query=QUERY,
        fname=FNAME,
        max_runs=MAX_RUNS,
        points_amount=POINTS_AMOUNT,
    )  # type: Dict[str, Any]

    create_maps(**evo_parameters)
