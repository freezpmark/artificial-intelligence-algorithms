import copy
import random
import re
import time
from typing import Any, Dict, Generator, List, Tuple


def entityGenerator(
    map_terrained: List[List[int]],
) -> Generator[Tuple[int, int], None, None]:
    """Generator of positions that are used for entities.

    Arguments:
        map_terrained {list} -- 2D map list of terrained map

    Yields:
        tuple -- (x, y) coordinate of generated position
    """
    # Generator[yield_type, send_type, return_type]
    # what is SendType and ReturnType?
    reserved = set()
    for i, row in enumerate(map_terrained):
        for j, x in enumerate(row):
            if x == -1:
                reserved.add((i, j))

    while True:
        x = random.randint(0, len(map_terrained) - 1)
        y = random.randint(0, len(map_terrained[0]) - 1)
        if (x, y) not in reserved:
            reserved.add((x, y))
            yield (x, y)


def generateEntities(
    map_terrained: List[List[str]], amount: int
) -> Dict[str, Any]:
    """Generates position for papers, base and starting location.

    Arguments:
        map_terrained {list} -- 2D map list of terrained map
        amount {int} -- amount of papers we wish to generate

    Returns:
        dict -- has 3 entity keys as names with coordinate values
    """
    if not map_terrained:
        return {}

    map_walled_int = [list(map(int, i)) for i in map_terrained]
    gen = entityGenerator(map_walled_int)

    papers = [next(gen) for _ in range(amount)]
    base = next(gen)
    start = next(gen)

    return {"papers": papers, "base": base, "start": start}


# for above 2 functions, make a better solution maybe?


def evolutionize(
    map_list: List[List[int]], attempts: int, print_stats: bool
) -> List[List[int]]:
    """Runs evolutionary algorithm for map with walls to fill it with terrain.

    Arguments:
        map_list {list} -- 2D map list of walls
        attempts {int} -- number of times algorithm runs
        print_stats {boolean} -- turns on debug mode that prints the solution

    Returns:
        list -- 2D map list that is filled by the most amount of terrain
    """
    if not map_list:
        return map_list

    found_solution = False
    attempt_number = 1
    while not found_solution and attempt_number <= attempts:
        map_tuple = {
            (i, j): -col
            for i, row in enumerate(map_list)
            for j, col in enumerate(row)
        }

        SHAPE = len(map_list), len(map_list[0])
        ROCKS = sum(val != 0 for val in map_tuple.values())
        TO_RAKE = SHAPE[0] * SHAPE[1] - ROCKS
        GENES = (SHAPE[0] + SHAPE[1]) * 2  # gene - an intruction (PERIMETER)
        CHROMOSOMES = 30  # chromosome - solution defined by genes
        GENERATIONS = 100  # generation - set of all chromosomes

        MIN_MUT_RATE = 0.05
        MAX_MUT_RATE = 0.80
        CROSS_RATE = 0.90

        start_time = time.time()
        gen_times = []
        prev_max = 0

        # generating chromosomes for one population/generation
        population = []
        genes = random.sample(range(1, GENES), GENES - 1)
        for _ in range(CHROMOSOMES):
            random.shuffle(genes)
            chromosome = [num * random.choice([-1, 1]) for num in genes]
            population.append(chromosome)

        # loop of generations
        mut_rate = MIN_MUT_RATE
        for i in range(GENERATIONS):
            generation_time = time.time()

            # evaluate all chromosomes and save the best one
            fit, fit_max, j_max = [], 0, 0
            for j in range(CHROMOSOMES):
                unraked, fills = rakeMap(
                    population[j], copy.copy(map_tuple), SHAPE
                )
                raked = TO_RAKE - unraked
                fit.append(raked)
                if raked > fit_max:
                    j_max, fit_max, map_filled = j, raked, fills

            if prev_max < fit_max:
                print(f"Generation: {i+1},", end="\t")
                print(f"Max raked: {fit_max} (out of {TO_RAKE})", end="\t")
                print(f"Mutation rate: {round(mut_rate, 2)}")
            if fit_max == TO_RAKE:
                found_solution = True
                gen_times.append(time.time() - generation_time)
                break

            # increase mutation rate each generation to prevent local maximums
            mut_rate = (
                mut_rate if mut_rate >= MAX_MUT_RATE else mut_rate + 0.01
            )

            # next generation creating, 1 iteration for 2 populations
            children = []  # type: List[Any]
            for i in range(0, CHROMOSOMES, 2):

                # pick 2 better chromosomes out of 4
                pick = random.sample(range(CHROMOSOMES), 4)
                better1 = pick[0] if fit[pick[0]] > fit[pick[1]] else pick[1]
                better2 = pick[2] if fit[pick[2]] > fit[pick[3]] else pick[3]

                # copying better genes to 2 child chromosomes
                children.extend([[], []])
                for j in range(GENES - 1):
                    children[i].append(population[better1][j])
                    children[i + 1].append(population[better2][j])

                # mutating 2 chromosomes with uniform crossover
                # (both inherit the same amount of genetic info)
                if random.random() < CROSS_RATE:
                    for c in range(2):
                        for g in range(GENES - 1):
                            if random.random() < mut_rate:

                                # search for gene with mut_num number
                                mut_num = random.randint(1, GENES)
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
            for i in range(GENES - 1):
                children[0][i] = population[j_max][i]

            population = children

            prev_max = fit_max
            gen_times.append(time.time() - generation_time)

        # printing stats, solution and map
        if print_stats:
            total = round(time.time() - start_time, 2)
            avg = round(sum(gen_times) / len(gen_times), 2)
            chromo = " ".join(map(str, population[j_max]))
            print(
                "{} a solution!".format(
                    "Found" if found_solution else "Couldn't find"
                )
            )
            print(f"Total time elapsed is {total}s,", end="\t")
            print(f"each generation took {avg}s in average.")
            print(f"Chromosome: {chromo}")

            for row in map_filled:
                for col in row:
                    print("{0:2}".format(col), end=" ")
                print()
            attempt_number += 1
            if not found_solution and attempt_number <= attempts:
                print(f"\nAttempt number {attempt_number}.")

    return map_filled if found_solution else []


def rakeMap(
    chromosome: List[int],
    map_tuple: Dict[Tuple[int, int], int],
    SHAPE: Tuple[int, int],
) -> Tuple[int, List[List[int]]]:
    """Attempts to fill the map terrain with chromosome that is defined
    by the order of instructions known as genes.

    Arguments:
        chromosome {list} -- collection of genes (instructions)
        map_tuple {dict} -- map defined by tuples of x, y coordinates as keys
        SHAPE {list} -- consist of height and width length values

    Returns:
        int -- amount of unraked (unfilled) positions
        list -- 2D map list that is filled with terrain
    """
    ROWS, COLS = SHAPE[0], SHAPE[1]
    HALF_PERIMETER = SHAPE[0] + SHAPE[1]
    UNRAKED = 0

    parents = {}  # type: Dict[Any, Any]
    pos = 0  # type: Any
    order = 1
    for gene in chromosome:

        # get starting position and movement direction
        pos_num = abs(gene)
        if pos_num <= COLS:  # go DOWN
            pos, move = (0, pos_num - 1), (1, 0)
        elif pos_num <= HALF_PERIMETER:  # go RIGHT
            pos, move = (pos_num - COLS - 1, 0), (0, 1)
        elif pos_num <= HALF_PERIMETER + ROWS:  # go LEFT
            pos, move = (pos_num - HALF_PERIMETER - 1, COLS - 1), (0, -1)
        else:  # go UP
            pos, move = (
                (ROWS - 1, pos_num - HALF_PERIMETER - ROWS - 1),
                (-1, 0),
            )

        # checking whether we can enter the garden with current pos
        if map_tuple[pos] == UNRAKED:
            parents = {}
            parent = 0

            # move until we reach end of the map
            while 0 <= pos[0] < ROWS and 0 <= pos[1] < COLS:

                # collision to raked sand/rock
                if map_tuple[pos] != UNRAKED:
                    pos = parent  # get previous pos
                    parent = parents[pos]  # get previous parent

                    # change moving direction
                    if move[0] != 0:  # Y -> X
                        R = pos[0], pos[1] + 1
                        L = pos[0], pos[1] - 1
                        R_inbound = R[1] < COLS
                        L_inbound = L[1] >= 0
                        R_free = R_inbound and map_tuple[R] == UNRAKED
                        L_free = L_inbound and map_tuple[L] == UNRAKED

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
                        D = pos[0] + 1, pos[1]
                        U = pos[0] - 1, pos[1]
                        D_inbound = D[0] < ROWS
                        U_inbound = U[0] >= 0
                        D_free = D_inbound and map_tuple[D] == UNRAKED
                        U_free = U_inbound and map_tuple[U] == UNRAKED

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

                map_tuple[pos] = order
                parents[pos] = parent
                parent = pos
                pos = pos[0] + move[0], pos[1] + move[1]

            order += 1

    fills = []  # type: List[List[int]]
    unraked = 0
    j = -1
    for i, fill in enumerate(map_tuple.values()):
        if fill == UNRAKED:
            unraked += 1
        if i % COLS == 0:
            j += 1
            fills.append([])
        fills[j].append(fill)

    return unraked, fills


def saveMap(
    map_terrained: List[List[str]],
    file_name: str,
    entities: Dict[str, Any] = {},
) -> None:

    """Saves terrained map into text file named file_name
    with entities being surrounded with certain brackets.

    Arguments:
        map_terrained {list} -- 2D map list of terrained map
        file_name {str} -- name of the text file we're saving the map into
        entities {dict} -- has 3 entity keys as names with coordinate values
    """
    # specify what is happening.. walls/terrain/objects? to create file suffix
    spacing = "{:^5}" if entities else "{:^3}"

    with open("simulation/maps/" + file_name, "w") as f:
        for i, row in enumerate(map_terrained):
            for j in range(len(row)):
                string = str(map_terrained[i][j])
                if entities:
                    if (i, j) in entities["papers"]:
                        string = "(" + string + ")"
                    elif (i, j) == entities["base"]:
                        string = "[" + string + "]"
                    elif (i, j) == entities["start"]:
                        string = "{" + string + "}"
                f.write(spacing.format(string))
            f.write("\n")


def loadMap(file_name: str) -> List[List[str]]:
    map_walls = []

    try:
        with open("simulation/maps/" + file_name) as f:
            line = f.readline().rstrip()
            map_walls.append(line.split())
            prev_length = len(line)

            for line in f:
                line = line.rstrip()
                if prev_length != len(line):
                    map_walls = []
                    break
                prev_length = len(line)
                map_walls.append(line.split())
    except FileNotFoundError:
        map_walls = []

    return map_walls


def create_walls(new_file_name: str, query: str) -> str:
    # 2D map list with walls that are represented by number 1
    map_walls = []

    if re.search(r"[0-9]+x[0-9]+(\ \([0-9]+,[0-9]+\))+$", query):
        query_list = query.split()
        ROWS, COLS = map(int, query_list[0].split("x"))
        walls = {eval(coordinate) for coordinate in query_list[1:]}
        map_walls = [["0"] * COLS for _ in range(ROWS)]

        for wall in walls:
            try:
                map_walls[wall[0]][wall[1]] = "1"
            except IndexError:
                map_walls = []

    if map_walls:
        new_file_name = new_file_name + "_wal"
        saveMap(map_walls, new_file_name + ".txt")
        return new_file_name
    return ""


def create_terrain(file_name: str, new_file_name: str, attempts: int) -> str:
    # Creates a map using evolutionary algorithm and saves it to a file.

    if not file_name:
        file_name = new_file_name
    map_walled = loadMap(file_name + ".txt")

    map_walled_int = [[int(i) for i in subarray] for subarray in map_walled]
    map_walled_int = evolutionize(map_walled_int, attempts, True)
    map_terrained = [[str(i) for i in subarray] for subarray in map_walled_int]

    if map_terrained:
        if new_file_name.endswith("_wal"):
            new_file_name = new_file_name[:-4]
        new_file_name += "_ter"
        saveMap(map_terrained, new_file_name + ".txt")
        return new_file_name
    return ""


def create_objects(file_name: str, new_file_name: str, papers: int) -> str:

    if not file_name:
        file_name = new_file_name
    map_terrained = loadMap(file_name + ".txt")

    entities = generateEntities(
        map_terrained, papers
    )  # return map with objects (entities)

    if map_terrained:
        if new_file_name.endswith("_ter"):
            new_file_name = new_file_name[:-4]
        new_file_name += "_obj"
        saveMap(
            map_terrained, new_file_name + ".txt", entities
        )  # remove entities argument
        return new_file_name
    return ""


if __name__ == "__main__":

    query = "10x12 (1,5) (2,1) (3,4) (4,2) (6,8) (6,9)"
    file_name = ""
    new_file_name = "default"
    attempts = 3  # number of max times evolutionary algorithm runs
    papers = 3  # amount of papers we wish to create

    new_file_name = create_walls(new_file_name, query)
    new_file_name = create_terrain(file_name, new_file_name, attempts)
    new_file_name = create_objects(file_name, new_file_name, papers)
