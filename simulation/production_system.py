import collections as col
from itertools import islice


def runProduction():
    Rules = col.namedtuple("Rule", "name conds acts")

    # ToDo: check whether file is in correct format with regexp
    # we've loaded the files
    with open("simulation/rules.txt") as f1:
        rules = []
        while rule := [line.rstrip("\n:") for line in islice(f1, 3)]:
            rules.append(Rules(*rule))
            f1.readline()
    with open("simulation/facts.txt") as f2:
        facts = [fact.rstrip() for fact in f2]

    while True:
        for rule in rules:
            mapping = {}
            conds = rule.conds.split(",")
            for cond in conds:
                for fact in facts:
                    pot_mapping = {}
                    upd = True
                    csplit = cond.split()
                    fsplit = fact.split()
                    for i, (c, f) in enumerate(zip(csplit, fsplit)):
                        if c[0] == "?" and c not in mapping:
                            pot_mapping[c] = f
                        elif c != f:
                            upd = False
                            break
                        elif c[0] == "<":
                            if mapping[csplit[i+1]] == mapping[csplit[i+2]]:
                                upd = False
                            break
                    if upd:
                        mapping = {**mapping, **pot_mapping}
                        break

            print(mapping)
        break


if __name__ == "__main__":
    runProduction()

    # save rules: rule -> rule -> rule -> rule: struct(name, cond, action)
    # save facts: fact -> fact -> fact -> fact: struct(<cond>)

    # ITERATE through to-be FACTS
    #     ITERATE through RULES
    #         ITERATE through RULES' BRACKETS (recursively)
    #             ITERATE through FACTS to fill ?X's (everywhere) and ?Y etc.
    #         Save action to be processed (actions are reseted after action)

    # () with <> means they have to be different
    # (they can be the same among siblings! Parent can have 2x the same child)
