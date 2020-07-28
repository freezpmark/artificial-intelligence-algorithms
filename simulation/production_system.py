import collections as col
from itertools import islice


def expand(conds, facts, mapping):

    acts = []
    for fact in facts:
        fact = fact.split()  # uz sme daco appendli for act!
        tmp_mapping = {}
        upd = True
        for i, (c, f) in enumerate(zip(conds, fact)):
            c_key = c.rstrip(",")
            if c_key.startswith("?") and f[0].isupper():  # new entity
                if c_key not in mapping:
                    if f not in mapping.values():
                        tmp_mapping[c_key] = f  # # neexistuje ani jedno
                    else:
                        upd = False
                        break  # f existuje, a c neexistuje
                elif mapping[c_key] != f:
                    upd = False
                    break  # c existuje, f v nom nieje -> break
                else:
                    continue  # ak existuju oba, continue
            elif c_key.startswith("<"):  # special cond at the end
                if mapping[conds[i + 1]] == mapping[conds[i + 2]]:
                    upd = False
                break
            elif c_key != f:  # difference
                upd = False
                break

            if c.endswith(","):  # next cond -> recursive call
                new_act = expand(
                    conds[i + 1 :], facts, {**mapping, **tmp_mapping}
                )
                if new_act:
                    if isinstance(new_act, list):  # nesting conds
                        acts += new_act
                    else:
                        acts.append(new_act)

        if upd and not c.endswith(","):  # found a match for action!
            # return {**mapping, **tmp_mapping}
            acts.append({**mapping, **tmp_mapping})
            if c == "<>":  # facts are irelevant, we would do duplicates
                break

    return acts


def runProduction():

    # ToDo: check whether file is in correct format with regexp
    # Loading files
    Rules = col.namedtuple("Rule", "name conds acts")
    with open("simulation/rules.txt") as f1:
        rules = []
        while rule := [line.rstrip("\n:") for line in islice(f1, 3)]:
            rules.append(Rules(*rule))
            f1.readline()
    with open("simulation/facts.txt") as f2:
        facts = [fact.rstrip() for fact in f2]


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
