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

    while True:
        all_actions = []
        for rule in rules:
            for actions in expand(rule.conds.split(), facts, {}):
                line_actions = []
                for rule_act in rule.acts.split(", "):
                    act_type, filling_act = rule_act.split(" ", 1)
                    for key in filling_act.split():
                        c_key = key.rstrip(",")
                        if c_key.startswith("?"):
                            filling_act = filling_act.replace(
                                c_key, actions[c_key]
                            )
                    line_actions.append(act_type + " " + filling_act)
                all_actions.append(line_actions)

        # remove duplicates
        i = 0
        for _ in range(len(all_actions)):
            message = True
            j = 0
            for _ in range(len(all_actions[i])):
                act_type, act = all_actions[i][j].split(" ", 1)
                if (
                    (act_type == "pridaj" and act in facts)
                    or (act_type == "vymaz" and act not in facts)
                    or (act_type == "sprava" and not message)
                ):
                    del all_actions[i][j]
                    message = False
                else:
                    j += 1
            if not all_actions[i]:
                del all_actions[i]
            else:
                i += 1

        # message must be positioned as last action in line_actions
        # message will be printed only when all acts were added

        # apply action if exist
        if not all_actions:
            break
        messages = []
        for action in all_actions[0]:
            act_type, act = action.split(" ", 1)
            if act_type == "pridaj":  # messages should be after actions
                facts.append(act)
            elif act_type == "vymaz":
                facts.remove(act)
            elif act_type == "sprava":
                messages.append(act)

        for fact in facts:
            print(fact)
        for msg in messages:
            print("MESSAGE:", msg)
        print()


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
