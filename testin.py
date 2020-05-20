with open('mapa10.txt') as f:
    mapa = []
    length = 0
    error = False
    for i, line in enumerate(f):
        line = line.rstrip()
        if length and length != len(line):
            error = True
            break

        mapa.append([])
        for column in line:
            mapa[i].append(column)

        length = len(line)

    print(error, mapa)