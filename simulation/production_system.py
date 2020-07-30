import collections as col
from itertools import islice
from typing import Any, Dict, List, Tuple


def loadFiles() -> Tuple[List[Any], List[str]]:

    Rules = col.namedtuple("Rule", "name conds labels")
    rules = []
    with open("simulation/rules.txt") as f1:
        while rule := [line.rstrip("\n:") for line in islice(f1, 3)]:
            rules.append(Rules(*rule))
            f1.readline()

    with open("simulation/facts.txt") as f2:
        facts = [fact.rstrip() for fact in f2]

    return rules, facts


def findActions(rules: List[Any], facts: List[str]) -> List[List[str]]:

    actions_found = []
    for rule in rules:  # LOOP over rules
        rule_acts = [act.split(" ", 1) for act in rule.labels.split(", ")]
        rule_acts_label = expand(rule.conds.split(), facts, {})
        for label in rule_acts_label:

            actions = []
            for type_, action in rule_acts:
                for key in [key.rstrip(",") for key in action.split()]:
                    if key.startswith("?"):
                        action = action.replace(key, label[key])
                actions.append(type_ + " " + action)

            actions_found.append(actions)

    return actions_found


def removeDuplicates(
    actions_found: List[List[str]], facts: List[str]
) -> List[List[str]]:

    i = 0
    for _ in range(len(actions_found)):
        message = True  # happens when there wasnt a duplicate in prev labels
        j = 0
        for _ in range(len(actions_found[i])):
            type_, act = actions_found[i][j].split(" ", 1)
            if (
                (type_ == "pridaj" and act in facts)
                or (type_ == "vymaz" and act not in facts)
                or (type_ == "sprava" and not message)
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


def applyActions(
    actions_appliable: List[List[str]], facts: List[str]
) -> Tuple[List[str], List[str]]:

    messages = []
    for action in actions_appliable[0]:
        type_, act = action.split(" ", 1)
        if type_ == "pridaj":
            facts.append(act)
        elif type_ == "vymaz":
            facts.remove(act)
        elif type_ == "sprava":
            messages.append(act)

    return facts, messages


def expand(
    conds: List[str], facts: List[str], label: Dict[str, str]
) -> List[Dict[str, str]]:
    # LOOP over rule's conditions recursively

    # LOOP over facts
    labels = []
    for fact_str in facts:
        fact_list = fact_str.split()
        tmp_label = {}
        continue_ = True
        for i, (c, f) in enumerate(zip(conds, fact_list)):
            c_key = c.rstrip(",")
            # label checking for "?"
            if c_key.startswith("?") and f[0].isupper():  # new entity
                if c_key not in label:
                    if f not in label.values():
                        tmp_label[c_key] = f
                    else:
                        continue_ = False
                elif label[c_key] != f:
                    continue_ = False
            # key identity checking with <> special cond
            elif c_key.startswith("<"):
                if label[conds[i + 1]] == label[conds[i + 2]]:
                    continue_ = False
            # unmatched condition with fact
            elif c_key != f:
                continue_ = False

            if not continue_:
                break

            # next condition -> recursive call
            if c.endswith(","):
                labels += expand(conds[i + 1 :], facts, {**label, **tmp_label})

        # label match found for action
        if continue_ and not c.endswith(","):
            labels.append({**label, **tmp_label})
            if c == "<>":  # iterating facts are irelevant as its independent
                break

    return labels


def runProduction() -> None:

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
