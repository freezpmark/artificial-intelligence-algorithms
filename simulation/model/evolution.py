import copy
import pickle
import random
import re
import time
from typing import Any, Dict, Generator, List, Tuple


class QueryError(Exception):
    pass


class FileImportError(Exception):
    pass


def listToDictTuple(map_list: List[List[int]]) -> Dict[Tuple[int, int], int]:
    """Transitions map list to dictionary of tuples as keys and values
    as terrain.

    Args:
        map_list (List[List[int]]): 2D map of 0 and 1 (walls) integers

    Returns:
        Dict[Tuple[int, int], int]: map defined by tuples of (x, y)
            being as coordinate keys
    """

    return {
        (i, j): -col
        for i, row in enumerate(map_list)
        for j, col in enumerate(row)
    }


def evolutionize(
    map_list: List[List[int]], max_runs: int, print_stats: bool = True
) -> Tuple[List[List[int]], Dict[Tuple[int, int], int], bool]:
    """Runs evolutionary algorithm on a map with walls to fill it with terrain.

    Args:
        map_list (List[List[int]]): 2D map of 0 and 1 (walls) integers
        max_runs (int): max number of attempts to find a solution with evo alg.
        print_stats (bool, optional): turns on debug mode that prints stats
            and solution

    Returns:
        Tuple[List[List[int]], Dict[Tuple[int, int], int], bool]: (
            2D map of various integers (wall being -1 now),
            chromosome - solution that consist of genes,
            fact announcing if the solution was found or not
        )
    """

    found_solution = False
    attempt_number = 1

    while not found_solution and attempt_number <= max_runs:
        map_tuple = listToDictTuple(map_list)
        shape = len(map_list), len(map_list[0])
        rocks_amount = sum(val != 0 for val in map_tuple.values())
        to_rake_amount = shape[0] * shape[1] - rocks_amount
        genes_amount = (shape[0] + shape[1]) * 2

        CHROMOSOMES = 30  # chromosome - solution defined by genes
        GENERATIONS = 100  # generation - set of all chromosomes
        MIN_MUT_RATE = 0.05
        MAX_MUT_RATE = 0.80
        CROSS_RATE = 0.90

        # generating chromosomes for one population/generation
        population = []
        genes = random.sample(range(1, genes_amount), genes_amount - 1)
        for _ in range(CHROMOSOMES):
            random.shuffle(genes)
            chromosome = [num * random.choice([-1, 1]) for num in genes]
            population.append(chromosome)

        start_time = time.time()
        gen_times = []

        # loop of generations
        prev_max = 0
        mu_rate = MIN_MUT_RATE
        for i in range(GENERATIONS):
            generation_time = time.time()

            # evaluate all chromosomes and save the best one
            fit, fit_max, best_index = [], 0, 0
            for j in range(CHROMOSOMES):
                unraked_amount, filled_map, tmp_rake_paths = rakeMap(
                    population[j], copy.copy(map_tuple), shape
                )
                raked_amount = to_rake_amount - unraked_amount
                fit.append(raked_amount)
                if raked_amount > fit_max:
                    best_index, fit_max = j, raked_amount
                    terr_map, rake_paths = filled_map, tmp_rake_paths

            if prev_max < fit_max:
                print(f"Generation: {i+1},", end="\t")
                print(f"Raked: {fit_max} (out of {to_rake_amount})", end="\t")
                print(f"Mutation rate: {round(mu_rate, 2)}")
            if fit_max == to_rake_amount:
                found_solution = True
                gen_times.append(time.time() - generation_time)
                break

            # increase mutation rate each generation to prevent local maximums
            mu_rate = mu_rate if mu_rate >= MAX_MUT_RATE else mu_rate + 0.01

            # next generation creating, 1 iteration for 2 populations
            children = []  # type: List[Any]
            for i in range(0, CHROMOSOMES, 2):

                # pick 2 better chromosomes out of 4
                pick = random.sample(range(CHROMOSOMES), 4)
                better1 = pick[0] if fit[pick[0]] > fit[pick[1]] else pick[1]
                better2 = pick[2] if fit[pick[2]] > fit[pick[3]] else pick[3]

                # copying better genes to 2 child chromosomes
                children.extend([[], []])
                for j in range(genes_amount - 1):
                    children[i].append(population[better1][j])
                    children[i + 1].append(population[better2][j])

                # mutating 2 chromosomes with uniform crossover
                # (both inherit the same amount of genetic info)
                if random.random() < CROSS_RATE:
                    for c in range(2):
                        for g in range(genes_amount - 1):
                            if random.random() < mu_rate:

                                # search for gene with mut_num number
                                mut_num = random.randint(1, genes_amount)
                                mut_num *= random.choice([-1, 1])
                                f = 0
                                for k, gene in enumerate(children[i + c]):
                                    if gene == mut_num:
                                        f = k

                                # swap it with g gene, else replace g with it
                                if f:
                                    tmp = children[i + c][g]
                                    children[i + c][g] = children[i + c][f]
                                    children[i + c][f] = tmp
                                else:
                                    children[i + c][g] = mut_num

            # keep the best chromosome for next generation
            for i in range(genes_amount - 1):
                children[0][i] = population[best_index][i]

            population = children
            prev_max = fit_max
            gen_times.append(time.time() - generation_time)

        # printing stats, solution and map
        if print_stats:
            total = round(time.time() - start_time, 2)
            avg = round(sum(gen_times) / len(gen_times), 2)
            chromo = " ".join(map(str, population[best_index]))
            answer = "found" if found_solution else "not found"
            print(f"Solution is {answer}!")
            print(f"Total time elapsed is {total}s,", end="\t")
            print(f"each generation took {avg}s in average.")
            print(f"Chromosome: {chromo}")

            attempt_number += 1
            if not found_solution and attempt_number <= max_runs:
                print(f"\nAttempt number {attempt_number}.")

    return terr_map, rake_paths, found_solution


def rakeMap(
    chromosome: List[int],
    map_tuple: Dict[Tuple[int, int], int],
    shape: Tuple[int, int],
) -> Tuple[int, List[List[int]], Dict[Tuple[int, int], int]]:
    """Attempts to fill the map terrain with chromosome that is defined
    by the order of instructions known as genes.

    Args:
        chromosome (List[int]): ordered set of genes (instructions)
        map_tuple (Dict[Tuple[int, int], int]): map defined by tuples
            of (x, y) being as coordinate keys
        shape (Tuple[int, int]): height and width lengths of the map

    Returns:
        Tuple[int, List[List[int]], Dict[Tuple[int, int], int]]:
            (amount of unraked spots, terrained map, raking paths)
    """

    rows, cols = shape
    half_perimeter = sum(shape)
    UNRAKED = 0
    parents = {}  # type: Dict[Any, Any]
    rake_paths = {}  # type: Dict[Tuple[int, int], int]
    pos = 0  # type: Any
    order = 1

    for gene in chromosome:

        # get starting position and movement direction
        pos_num = abs(gene)
        if pos_num <= cols:  # go DOWN
            pos, move = (0, pos_num - 1), (1, 0)
        elif pos_num <= half_perimeter:  # go RIGHT
            pos, move = (pos_num - cols - 1, 0), (0, 1)
        elif pos_num <= half_perimeter + rows:  # go LEFT
            pos, move = (pos_num - half_perimeter - 1, cols - 1), (0, -1)
        else:  # go UP
            pos, move = (
                (rows - 1, pos_num - half_perimeter - rows - 1),
                (-1, 0),
            )

        # checking whether we can enter the garden with current pos
        if map_tuple[pos] == UNRAKED:
            parents = {}
            parent = 0

            # move until we reach end of the map
            while 0 <= pos[0] < rows and 0 <= pos[1] < cols:

                # collision to raked sand/rock
                if map_tuple[pos] != UNRAKED:
                    pos = parent  # get previous pos
                    parent = parents[pos]  # get previous parent

                    # change moving direction
                    if move[0] != 0:  # Y -> X
                        R_pos = pos[0], pos[1] + 1
                        L_pos = pos[0], pos[1] - 1
                        R_inbound = R_pos[1] < cols
                        L_inbound = L_pos[1] >= 0
                        R_free = R_inbound and map_tuple[R_pos] == UNRAKED
                        L_free = L_inbound and map_tuple[L_pos] == UNRAKED

                        if R_free and L_free:
                            move = (0, 1) if gene > 0 else (0, -1)
                        elif R_free:
                            move = 0, 1
                        elif L_free:
                            move = 0, -1
                        elif R_inbound and L_inbound:
                            move = 0, 0
                        else:
                            break  # reached end of the map so we can leave

                    else:  # X -> Y
                        D_pos = pos[0] + 1, pos[1]
                        U_pos = pos[0] - 1, pos[1]
                        D_inbound = D_pos[0] < rows
                        U_inbound = U_pos[0] >= 0
                        D_free = D_inbound and map_tuple[D_pos] == UNRAKED
                        U_free = U_inbound and map_tuple[U_pos] == UNRAKED

                        if D_free and U_free:
                            move = (1, 0) if gene > 0 else (-1, 0)
                        elif D_free:
                            move = 1, 0
                        elif U_free:
                            move = -1, 0
                        elif D_inbound and U_inbound:
                            move = 0, 0
                        else:
                            break

                    # if we cant change direction, remove the path
                    if not any(move):
                        order -= 1
                        while parents[pos] != 0:
                            map_tuple[pos] = 0
                            pos = parents[pos]
                        map_tuple[pos] = 0
                        break

                # save the order num and parent, move by setting new pos
                map_tuple[pos] = order
                parents[pos] = parent
                parent = pos
                pos = pos[0] + move[0], pos[1] + move[1]

            # save paths for visualization
            rake_path = {key: order for key in parents}
            rake_paths = {**rake_paths, **rake_path}
            order += 1

    filled_map = []  # type: List[List[int]]
    unraked_amount = 0
    row_number = -1

    for i, order_num in enumerate(map_tuple.values()):
        if order_num == UNRAKED:
            unraked_amount += 1
        if i % cols == 0:
            row_number += 1
            filled_map.append([])
        filled_map[row_number].append(order_num)

    return unraked_amount, filled_map, rake_paths


def generateProperties(
    map_list: List[List[str]], points_amount: int
) -> List[List[str]]:
    """Adds properties to terrained map.

    Args:
        map_list (List[List[str]]): 2D terrained map that will be propertied
        points_amount (int): amount of destination points to visit

    Returns:
        List[List[str]]: 2D propertied map
    """

    def positionGenerator(
        terrained_map: List[List[str]],
    ) -> Generator[Tuple[int, int], None, None]:
        """Generator of free positions for properties.

        Args:
            terrained_map (List[List[str]]): 2D terrained map

        Yields:
            Generator[Tuple[int, int], None, None]: coordinate of free position
        """

        reserved = set()
        for i, row in enumerate(terrained_map):
            for j, col in enumerate(row):
                if col == "-1":
                    reserved.add((i, j))

        while True:
            x = random.randint(0, len(terrained_map) - 1)
            y = random.randint(0, len(terrained_map[0]) - 1)
            if (x, y) not in reserved:
                reserved.add((x, y))
                yield (x, y)

    pos = positionGenerator(map_list)

    first = next(pos)
    start = next(pos)
    map_list[first[0]][first[1]] = "[" + map_list[first[0]][first[1]] + "]"
    map_list[start[0]][start[1]] = "{" + map_list[start[0]][start[1]] + "}"
    for _ in range(points_amount):
        point = next(pos)
        map_list[point[0]][point[1]] = "(" + map_list[point[0]][point[1]] + ")"

    return map_list


def saveMap(
    map_list: List[List[str]],
    export_name: str,
    show: bool = False,
    spacing: str = "{:^3}",
) -> None:
    """Saves a map into file. It can also print the map into console.

    Args:
        map_list (List[List[str]]): 2D map
        export_name (str): name of the file we're saving the map into
        show (bool): Print saved map into console. Defaults to False.
        spacing (str, optional): spacing between numbers. Defaults to "{:^3}".
    """

    with open("simulation/data/" + export_name + ".txt", "w") as f:
        for i, row in enumerate(map_list):
            for j in range(len(row)):
                f.write(spacing.format(map_list[i][j]))
                if show:
                    print(spacing.format(map_list[i][j]), end=" ")
            f.write("\n")
            if show:
                print()


def loadMap(import_name: str) -> List[List[str]]:
    """Loads a map from a file into 2D list array of strings.

    Args:
        import_name (str): name of the file to load (with _wal/ter/pro suffix)

    Returns:
        List[List[str]]: 2D map
    """

    map_ = []
    try:
        with open("simulation/data/" + import_name + ".txt") as f:
            line = f.readline().rstrip()
            map_.append(line.split())
            prev_length = len(line)

            for line in f:
                line = line.rstrip()
                if prev_length != len(line):
                    map_ = []
                    break
                prev_length = len(line)
                map_.append(line.split())
    except FileNotFoundError:
        map_ = []

    return map_


def saveSolution(
    rake_paths: Dict[Tuple[int, int], int], export_name: str
) -> None:
    """Saves solution (paths) of evolutionary alg. into pickle file.

    Args:
        rake_paths (Dict[Tuple[int, int], int]): order of raking the map
        export_name (str): name of pickle file into which the solution
            will be saved
    """
    with open("simulation/data/" + export_name + "_solution", "wb") as f:
        pickle.dump(rake_paths, f)


def createWalls(query: str, export_name: str, show: bool = False) -> None:
    """Creates a file that represents unterrained map with walls.
    Map is filled with "1" being walls and "0" being walkable places.

    Args:
        query (str): contains size of the map and tuple coordinates of walls
            example: "10x12 (1,5) (2,1) (3,4) (4,2) (6,8) (6,9)"
        export_name (str): root name of the file that is going to be created
        show (bool): Print created walls into console. Defaults to False.
    """

    walled_map = []
    if re.search(r"[0-9]+x[0-9]+(\ \([0-9]+,[0-9]+\))+$", query):
        query_list = query.split()
        rows, cols = map(int, query_list[0].split("x"))
        walls = {eval(coordinate) for coordinate in query_list[1:]}

        walled_map = [["0"] * cols for _ in range(rows)]
        for wall in walls:
            try:
                walled_map[wall[0]][wall[1]] = "1"
            except IndexError:
                walled_map = []

    if not walled_map:
        raise QueryError("Invalid query!")

    export_name += "_wal"
    saveMap(walled_map, export_name, show)


def createTerrain(max_runs: int, import_name: str, show: bool = False) -> str:
    """Creates a file that represents terrained map.
    Map is filled with "-1" being walls and walkable places are filled
    with various numbers generated by evolutionary algorithm.

    Args:
        max_runs (int): max number of attempts to find a solution with evo alg.
        import_name (str): root name of the file that is going to be imported
        show (bool): Print created terrain into console. Defaults to False.

    Returns:
        str: message announcing if the solution was found or not
    """

    walled_map = loadMap(import_name + "_wal")
    if not walled_map:
        raise FileImportError("Invalid import name for creating terrain!")

    map_list = [[int(i) for i in subarray] for subarray in walled_map]
    map_list, rake_paths, solution = evolutionize(map_list, max_runs)
    terrained_map = [[str(i) for i in subarray] for subarray in map_list]

    export_name = import_name + "_ter"
    saveMap(terrained_map, export_name, show)
    saveSolution(rake_paths, export_name)

    return "Solution was found." if solution else "Solution was not found."


def createProperties(
    points_amount: int, import_name: str, show: bool = False,
) -> None:
    """Creates a file that represents terrained map with properties.
    Properties are represented with a bracket around the number of terrain.
    {} - starting position, [] - first position to visit, () - points to visit.

    Args:
        points_amount (int): amount of destination points to visit
        import_name (str): root name of the file that is going to be imported
        show (bool): Print created properties into console. Defaults to False.
    """

    terrained_map = loadMap(import_name + "_ter")
    if not terrained_map:
        raise FileImportError("Invalid import name for creating properties!")

    map_propertied = generateProperties(terrained_map, points_amount)

    export_name = import_name + "_pro"
    saveMap(map_propertied, export_name, show, "{:^5}")


def createMaps(
    begin_create: str,
    query: str,
    export_name: str,
    max_runs: int,
    points_amount: int,
) -> None:
    """Runs evolution algorithm to create and save the maps into text file.

    Args:
        begin_create (str): defines from which part of creating maps to
            start until the ending properties part
            (walls/terrain/properties)
        query (str): contains size of map and tuple coordinates of walls
            example: "10x12 (1,5) (2,1) (3,4) (4,2) (6,8) (6,9)"
        export_name (str): name of the file that is going to be created
        max_runs: (int): max number of attempts to find a solution
        points_amount (int): amount of destination points to visit
    """

    next_ = False

    try:
        if begin_create == "walls":
            createWalls(query, export_name)
            next_ = True
        if begin_create == "terrain" or next_:
            createTerrain(max_runs, export_name)
            next_ = True
        if begin_create == "properties" or next_:
            createProperties(points_amount, export_name, show=True)

    except QueryError as e:
        print(e)
    except FileImportError as e:
        print(e)


if __name__ == "__main__":

    # parameters:

    # walls uses: query, file_name, max_runs, points_amount
    # terrain uses: file_name, max_runs, points_amount
    # properties uses: file_name, points_amount
    begin_create = "walls"
    query = "10x12 (1,5) (2,1) (3,4) (4,2) (6,8) (6,9)"
    file_name = "queried"
    max_runs = 3
    points_amount = 10

    evo_parameters = dict(
        begin_create=begin_create,
        query=query,
        export_name=file_name,
        max_runs=max_runs,
        points_amount=points_amount,
    )  # type: Dict[str, Any]

    createMaps(**evo_parameters)
