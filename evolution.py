import random
import re
import time
import copy


def create(query, attempts, papers):
    """Creates a map using evolutionary algorithm and saves it to a file.

    Arguments:
        query {string} -- file name to load, or command that creates a new map
                          These two examples will have the same outcome:
                            query -- "file_name.txt" < "001\\n000\\n010"
                            query -- "3x3 (0,2) (2,1)"
        attempts {int} -- number of times evolutionary algorithm runs
        papers {int} -- amount of papers we wish to create
    """
    # DECLARATION: Load or create walls
    map_walls, file_name = loadWalls(query)
    if not map_walls:
        print("Invalid query!")
        return

    # INITIALIZATION: Add terrain
    map_terrained = evolutionize(map_walls, attempts, True)
    if not map_terrained:
        print("Evolution algorithm could not find a solution, try again.")
        return False

    # INITIALIZATION: Add entities at random places
    entities = generateEntities(map_terrained, papers)
    saveMap(map_terrained, 'evo_' + file_name, entities)

    return True


def entityGenerator(map_terrained):
    """Generator of positions that are used for entities.

    Arguments:
        map_terrained {list} -- 2D map list of terrained map

    Yields:
        tuple -- (x, y) coordinate of generated position
    """
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


def generateEntities(map_terrained, amount):
    """Generates position for papers, base and starting location.

    Arguments:
        map_terrained {list} -- 2D map list of terrained map
        amount {int} -- amount of papers we wish to generate

    Returns:
        dict -- has 3 entity keys as names with coordinate values
    """
    gen = entityGenerator(map_terrained)

    papers = [next(gen) for _ in range(amount)]
    base = next(gen)
    start = next(gen)

    return {'papers': papers, 'base': base, 'start': start}


def saveMap(map_terrained, file_name, entities):
    """Saves terrained map into text file named file_name
    with entities being surrounded with certain brackets.

    Arguments:
        map_terrained {list} -- 2D map list of terrained map
        file_name {str} -- name of the text file we're saving the map into
        entities {dict} -- has 3 entity keys as names with coordinate values
    """
    with open(file_name, 'w') as f:
        for i, row in enumerate(map_terrained):
            for j in range(len(row)):
                string = str(map_terrained[i][j])
                if (i, j) in entities['papers']:
                    string = '(' + string + ')'
                elif (i, j) == entities['base']:
                    string = '[' + string + ']'
                elif (i, j) == entities['start']:
                    string = '{' + string + '}'
                f.write("{:^5}".format(string))
            f.write('\n')


def loadWalls(query):
    """ Loads or creates a map of walls that is represented by 2D list.
    Query can consist of either a file name, or command that creates a new map.
    These two examples will have the same outcome:

    query -- "file_name.txt" < "001\\n000\\n010"\n
    query -- "3x3 (0,2) (2,1)"

    Arguments:
        query {string} -- file name to load, or command that creates a new map

    Returns:
        list -- 2D map list with walls that are represented by number 1
        str -- file name into which we will save finished map
    """
    map_walls = []

    # Load from string
    if re.search(r'[0-9]+x[0-9]+(\ \([0-9]+,[0-9]+\))+$', query):
        query = query.split()
        ROWS, COLS = map(int, query[0].split('x'))
        walls = {eval(coordinate) for coordinate in query[1:]}
        map_walls = [[0] * COLS for _ in range(ROWS)]

        for wall in walls:
            try:
                map_walls[wall[0]][wall[1]] = 1
            except IndexError:
                map_walls = None

        file_name = 'commanded.txt'

    # Load from file
    elif re.search(r'\.txt', query):
        with open(query) as f:
            line = f.readline().rstrip()
            map_walls.append([int(column) for column in line])
            prev_length = len(line)

            for line in f:
                line = line.rstrip()
                if prev_length != len(line):
                    map_walls = None
                    break
                prev_length = len(line)
                map_walls.append([int(column) for column in line])

        file_name = query

    return map_walls, file_name


def listToTuple(map_list):
    """Converts a 2D map list to dictionary of tuples as keys with x, y coors
    """
    map_tuple = {}
    for i, row in enumerate(map_list):
        for j in range(len(row)):
            map_tuple[i, j] = map_list[i][j]

    return map_tuple


def evolutionize(map_list, attempts, print_stats):
    """Runs evolutionary algorithm for map with walls to fill it with terrain.

    Arguments:
        map_list {list} -- 2D map list of walls
        attempts {int} -- number of times algorithm runs
        print_stats {boolean} -- turns on debug mode that prints the solution

    Returns:
        list -- 2D map list that is filled by the most amount of terrain
    """
    found_solution = False
    attempt_number = 1
    while not found_solution and attempt_number <= attempts:
        map_tuple = listToTuple(map_list)
        map_tuple = {key: -map_tuple[key] for key in map_tuple.keys()}

        SHAPE = len(map_list), len(map_list[0])
        ROCKS = sum(val != 0 for val in map_tuple.values())
        TO_RAKE = SHAPE[0] * SHAPE[1] - ROCKS
        GENES = (SHAPE[0] + SHAPE[1]) * 2    # gene - an intruction (PERIMETER)
        CHROMOSOMES = 30               # chromosome - solution defined by genes
        GENERATIONS = 100              # generation - set of all chromosomes

        MIN_MUT_RATE = 0.05
        MAX_MUT_RATE = 0.80
        CROSS_RATE = 0.90

        start_time = time.time()
        gen_times = []
        prev_max = 0

        # generating chromosomes for one population/generation
        population = []
        genes = random.sample(range(1, GENES), GENES-1)
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
                    population[j], copy.copy(map_tuple), SHAPE)
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
            mut_rate = mut_rate if mut_rate >= MAX_MUT_RATE else mut_rate + .01

            # next generation creating, 1 iteration for 2 populations
            children = []
            for i in range(0, CHROMOSOMES, 2):

                # pick 2 better chromosomes out of 4
                pick = random.sample(range(CHROMOSOMES), 4)
                better1 = pick[0] if fit[pick[0]] > fit[pick[1]] else pick[1]
                better2 = pick[2] if fit[pick[2]] > fit[pick[3]] else pick[3]

                # copying better genes to 2 child chromosomes
                children.extend([[], []])
                for j in range(GENES-1):
                    children[i].append(population[better1][j])
                    children[i+1].append(population[better2][j])

                # mutating 2 chromosomes with uniform crossover
                # (both inherit the same amount of genetic info)
                if random.random() < CROSS_RATE:
                    for c in range(2):
                        for g in range(GENES-1):
                            if random.random() < mut_rate:

                                # search for gene with mut_num number
                                mut_num = random.randint(1, GENES)
                                mut_num *= random.choice([-1, 1])
                                f = 0
                                for k, gene in enumerate(children[i+c]):
                                    if gene == mut_num:
                                        f = k

                                # swap it with g gene, else replace g with it
                                if f:
                                    tmp = children[i+c][g]
                                    children[i+c][g] = children[i+c][f]
                                    children[i+c][f] = tmp
                                else:
                                    children[i+c][g] = mut_num

            # keep the best chromosome for next generation
            for i in range(GENES-1):
                children[0][i] = population[j_max][i]

            population = children

            prev_max = fit_max
            gen_times.append(time.time() - generation_time)

        # printing stats, solution and map
        if print_stats:
            total = round(time.time() - start_time, 2)
            avg = round(sum(gen_times) / len(gen_times), 2)
            chromo = " ".join(map(str, population[j_max]))
            print("{} a solution!".format(
                "Found" if found_solution else "Couldn't find"))
            print(f"Total time elapsed is {total}s,", end="\t")
            print(f"each generation took {avg}s in average.")
            print(f"Chromosome: {chromo}")

            for row in map_filled:
                for col in row:
                    print("{0:2}".format(col), end=' ')
                print()
            attempt_number += 1
            if not found_solution and attempt_number <= attempts:
                print(f"\nAttempt number {attempt_number}.")

    return map_filled if found_solution else []


def rakeMap(chromosome, map_tuple, SHAPE):
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

    order = 1
    for gene in chromosome:

        # get starting position and movement direction
        pos_num = abs(gene)
        if pos_num <= COLS:                         # go DOWN
            pos, move = (0, pos_num-1), (1, 0)
        elif pos_num <= HALF_PERIMETER:             # go RIGHT
            pos, move = (pos_num-COLS-1, 0), (0, 1)
        elif pos_num <= HALF_PERIMETER + ROWS:      # go LEFT
            pos, move = (pos_num-HALF_PERIMETER-1, COLS-1), (0, -1)
        else:                                       # go UP
            pos, move = (ROWS-1, pos_num-HALF_PERIMETER-ROWS-1), (-1, 0)

        # checking whether we can enter the garden with current pos
        if map_tuple[pos] == UNRAKED:
            parents = {}
            parent = 0

            # move until we reach end of the map
            while 0 <= pos[0] < ROWS and 0 <= pos[1] < COLS:

                # collision to raked sand/rock
                if map_tuple[pos] != UNRAKED:
                    pos = parent            # get previous pos
                    parent = parents[pos]   # get previous parent

                    # change moving direction
                    if move[0] != 0:    # Y -> X
                        R = pos[0], pos[1]+1
                        L = pos[0], pos[1]-1
                        R_inbound = R[1] < COLS
                        L_inbound = L[1] >= 0
                        R = R_inbound and map_tuple[R] == UNRAKED
                        L = L_inbound and map_tuple[L] == UNRAKED

                        if R and L:
                            move = (0, 1) if gene > 0 else (0, -1)
                        elif R:
                            move = 0, 1
                        elif L:
                            move = 0, -1
                        elif R_inbound and L_inbound:
                            move = False
                        else:
                            break   # reached end of the map so we can leave

                    else:               # X -> Y
                        D = pos[0]+1, pos[1]
                        U = pos[0]-1, pos[1]
                        D_inbound = D[0] < ROWS
                        U_inbound = U[0] >= 0
                        D = D_inbound and map_tuple[D] == UNRAKED
                        U = U_inbound and map_tuple[U] == UNRAKED

                        if D and U:
                            move = (1, 0) if gene > 0 else (-1, 0)
                        elif D:
                            move = 1, 0
                        elif U:
                            move = -1, 0
                        elif D_inbound and U_inbound:
                            move = False
                        else:
                            break

                    # if we cant change direction, remove the path
                    if not move:
                        order -= 1
                        while parents[pos] != 0:
                            map_tuple[pos] = 0
                            pos = parents[pos]
                        map_tuple[pos] = 0
                        break

                map_tuple[pos] = order
                parents[pos] = parent
                parent = pos
                pos = pos[0]+move[0], pos[1]+move[1]

            order += 1

    fills = []
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
