import collections as col
from itertools import islice


def expand(conds, facts, mapping):
    # ToDo: beautify this function!
    # LOOP over rule's conditions recursively

    acts = []
    # LOOP over facts
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
            # () with <> means they have to be different
            elif c_key.startswith("<"):  # special cond at the end,,
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
            acts.append({**mapping, **tmp_mapping})  # save action
            if c == "<>":  # facts are irelevant, we would do duplicates
                break

    return acts


def loadFiles():
    Rules = col.namedtuple("Rule", "name conds acts")
    with open("simulation/rules.txt") as f1:
        rules = []
        while rule := [line.rstrip("\n:") for line in islice(f1, 3)]:
            rules.append(Rules(*rule))
            f1.readline()

    with open("simulation/facts.txt") as f2:
        facts = [fact.rstrip() for fact in f2]

    return rules, facts


def findActions(rules, facts):

    actions_found = []
    for rule in rules:  # LOOP over rules
        rule_acts = [act.split(" ", 1) for act in rule.acts.split(", ")]
        rule_acts_label = expand(rule.conds.split(), facts, {})
        for label in rule_acts_label:

            actions = []
            for act_type, action in rule_acts:
                for key in [key.rstrip(",") for key in action.split()]:
                    if key.startswith("?"):
                        action = action.replace(key, label[key])
                actions.append(act_type + " " + action)

            actions_found.append(actions)

    return actions_found


def removeDuplicates(actions_found, facts):

    i = 0
    for _ in range(len(actions_found)):
        message = True  # happens when there wasnt a duplicate in prev acts
        j = 0
        for _ in range(len(actions_found[i])):
            act_type, act = actions_found[i][j].split(" ", 1)
            if (
                (act_type == "pridaj" and act in facts)
                or (act_type == "vymaz" and act not in facts)
                or (act_type == "sprava" and not message)
            ):
                del actions_found[i][j]
                message = False
            else:
                j += 1
        if not actions_found[i]:
            del actions_found[i]
        else:
            i += 1

    return actions_found


def applyActions(actions_appliable, facts):

    messages = []
    for action in actions_appliable[0]:
        act_type, act = action.split(" ", 1)
        if act_type == "pridaj":
            facts.append(act)
        elif act_type == "vymaz":
            facts.remove(act)
        elif act_type == "sprava":
            messages.append(act)

    return facts, messages


def runProduction():

    # ToDo: check whether file is in correct format with regexp
    rules, facts = loadFiles()

    # LOOP over to-be FACTS
    while True:
        actions_found = findActions(rules, facts)
        actions_appliable = removeDuplicates(actions_found, facts)

        if not actions_appliable:
            break

        facts, msgs = applyActions(actions_appliable, facts)
        for fact in facts:
            print(fact)
        for msg in msgs:
            print("MESSAGE:", msg)
        print()


if __name__ == "__main__":
    runProduction()
