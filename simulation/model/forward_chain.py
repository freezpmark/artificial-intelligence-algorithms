import collections as col
import random
import re
from itertools import islice
from typing import Any, Dict, List, Tuple


def loadRules(fname_rules: str) -> List[Any]:
    """Loads rules from the file.

    Args:
        fname_rules (str): name of file from which we load rules

    Returns:
        List[Any]: namedtuples with these attributes:
            name - name of the rule (unused)
            conds - conditions to fulfil actions
            acts - actions (message, add or remove fact from the set of facts)
    """

    Rules = col.namedtuple("Rule", "name conds acts")
    rules = []

    # non-whitespace character
    # ?X in each comma seperated statement
    # remove/add/message and ?<any word char> in each comma seperated statement
    # (instead of ".*?," we could use "[^,]*,", or combine it "[^,]*?,")
    patterns = [
        re.compile(r"\S+"),
        re.compile(r"((\?[A-Z]+)[^,]*, )*.*\?[A-Z].*"),
        re.compile(
            r"((add|remove|message).*\?\w.*?, )*(add|remove|message).*\?\w.*"
        ),
    ]
    with open("simulation/knowledge/" + fname_rules + ".txt") as f:
        while rule := [line.rstrip("\n:") for line in islice(f, 4)]:
            if rule.pop():
                print("There is no empty line after rule!")
            for i in range(len(Rules._fields)):
                if not patterns[i].match(rule[i]):
                    print(Rules._fields[i], "field is set wrong!")
                    return []
            rules.append(Rules(*rule))

    return rules


def loadFacts(fname_facts: str) -> List[str]:
    """Loads facts from the file.

    Args:
        fname_facts (str): name of file from which we load facts

    Returns:
        List[str]: fact sentences
    """

    with open("simulation/knowledge/" + fname_facts + ".txt") as f:
        facts = [fact.rstrip() for fact in f]

    return facts


def findActions(rules: List[Any], facts: List[str]) -> List[List[str]]:
    """Finds all actions from given facts according to given rules.

    Args:
        rules (List[Any]): namedtuples with these attributes:
            name - name of the rule (unused)
            conds - conditions to fulfil actions
            acts - actions (message, add or remove fact from the set of facts)
        facts (List[str]): known fact sentences

    Returns:
        List[List[str]]: lists of actions that have been found from each rule
    """

    found_actions = []
    for rule in rules:  # loop over rules
        rule_acts = [act.split(" ", 1) for act in rule.acts.split(", ")]
        rule_acts_label = expand(rule.conds.split(), facts, {})
        for label in rule_acts_label:

            actions = []
            for type_, action in rule_acts:
                for key in [key.rstrip(",") for key in action.split()]:
                    if key.startswith("?"):
                        action = action.replace(key, label[key])
                actions.append(type_ + " " + action)

            found_actions.append(actions)

    return found_actions


def removeDuplicates(
    actions: List[List[str]], facts: List[str]
) -> List[List[str]]:
    """Removes the outcome of actions that were already present in the facts.

    Args:
        actions (List[List[str]]): lists of actions that have been
            found from each rule
        facts (List[str]): known fact sentences

    Returns:
        List[List[str]]: lists of appliable actions
    """

    i = 0
    # loop over each rule
    for _ in range(len(actions)):
        message = True
        j = 0

        # loop over actions found from each rule
        for _ in range(len(actions[i])):
            type_, act = actions[i][j].split(" ", 1)
            if (
                (type_ == "add" and act in facts)
                or (type_ == "remove" and act not in facts)
                or (type_ == "message" and not message)
            ):
                del actions[i][j]
                message = False  # remove msg act if prev. act was deleted
            else:
                j += 1

        # remove empty set of actions
        if not actions[i]:
            del actions[i]
        else:
            i += 1

    return actions


def applyActions(
    appliable_actions: List[List[str]], facts: List[str]
) -> Tuple[str, List[str], List[str]]:
    """Applies list of actions that is first in the queue.

    Args:
        appliable_actions (List[List[str]]): lists of appliable actions
        facts (List[str]): known fact sentences

    Returns:
        Tuple[str, List[str], List[str]]: (applied action,
            known fact sentences, messages)
    """

    messages = []
    for action in appliable_actions[0]:
        type_, act = action.split(" ", 1)
        if type_ == "add":
            facts.append(act)
        elif type_ == "remove":
            facts.remove(act)
        elif type_ == "message":
            messages.append(act)

    return action, facts, messages


def expand(
    conds: List[str], facts: List[str], label: Dict[str, str]
) -> List[Dict[str, str]]:
    """Loops over conditions of rule recursively and finds all
    condition-matching labels from given facts.

    Args:
        conds (List[str]): conditions for fulfilling rule's actions
        facts (List[str]): known fact sentences
        label (Dict[str, str]): represent entities (?X -> <entity from fact>)

    Returns:
        List[Dict[str, str]]: labels
    """

    if conds[0] == "<>":  # identity checking is included in label checking
        return [label]

    labels = []
    # loop over facts
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
                        continue_ = False  # f already exist
                elif label[c_key] != f:
                    continue_ = False  # c and f does not match
            elif c_key != f:
                continue_ = False  # unmatched condition with fact

            if not continue_:
                break

            # next condition -> recursive call
            if c.endswith(","):
                labels += expand(conds[i + 1 :], facts, {**label, **tmp_label})

        # label match found for action
        if continue_ and not c.endswith(","):
            labels.append({**label, **tmp_label})

    return labels


def saveFacts(facts: List[str], save_fname_facts: str) -> None:
    """Saves all facts into text file.

    Args:
        facts (List[List[str]]): list of new and old facts
        save_fname_facts (str): name of the file
    """

    with open("simulation/knowledge/" + save_fname_facts + ".txt", "w") as f:
        f.write("\n".join(facts))


def runProduction(pars: Dict[str, Any]) -> None:
    """Sets parameters for running rule-based system with forward chaining.

    Args:
        pars (Dict[str, Any]): parameters:
            save_fname_facts (str): name of file into which facts will be saved
            load_fname_facts (str): name of file from which we load facts
            load_fname_rules (str): name of file from which we load rules
            step_by_step (bool): entering one fact by each production run
            facts_amount (int): number of facts we want to load (points)
            facts_random_order (bool): shuffle loaded facts
    """

    rules = loadRules(pars["load_fname_rules"])
    facts = loadFacts(pars["load_fname_facts"])
    if pars["facts_random_order"]:
        random.shuffle(facts)
    if pars["facts_amount"] < len(facts):
        facts = facts[: pars["facts_amount"]]

    if pars["step_by_step"]:
        new_facts = []  # type: List[str]
        stepped_facts = {}
        for i, key_fact in enumerate(facts):
            applied_facts, new_facts = runForwardChain(
                new_facts + [key_fact], rules, pars["save_fname_facts"]
            )
            stepped_facts[key_fact] = applied_facts
    else:
        applied_facts, new_facts = runForwardChain(
            facts, rules, pars["save_fname_facts"]
        )
        stepped_facts = {"All steps at once": applied_facts}

    for i, fact in enumerate(stepped_facts, 1):
        print(f"{str(i)}:  {fact} -> " + ", ".join(stepped_facts[fact]))


def runForwardChain(
    facts: List[str], rules: List[Any], save_fname_facts: str
) -> Tuple[List[str], List[str]]:
    """Runs forward chaining to discover all possible facts. Discovered
    new facts along with already known facts will be saved to text file.

    Args:
        facts (List[str]): known fact sentences
        rules (List[Any]): namedtuples with these attributes:
            name - name of the rule (unused)
            conds - conditions to fulfil actions
            acts - actions (message, add or remove fact from the set of facts)
        save_fname_facts (str): name of the file into which facts will be saved

    Returns:
        Tuple[List[str], List[str]]: (applied facts, all facts)
    """

    # loop over applied_facts (to-be facts)
    applied_facts = []
    while True:
        found_actions = findActions(rules, facts)
        appliable_actions = removeDuplicates(found_actions, facts)

        if not appliable_actions:
            saveFacts(facts, save_fname_facts)
            break

        applied_fact, facts, msgs = applyActions(appliable_actions, facts)
        applied_facts.append(applied_fact)

    return applied_facts, facts


if __name__ == "__main__":

    chain_parameters = dict(
        save_fname_facts="facts",
        load_fname_facts="facts_init",
        load_fname_rules="rules",
        step_by_step=True,
        facts_amount=11,
        facts_random_order=True,
    )

    runProduction(chain_parameters)
