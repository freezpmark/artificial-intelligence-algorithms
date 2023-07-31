"""This module serves to run 3. stage (out of 3) of creating simulation -
Rule Based Production System.

Production system belongs to knowledge systems that use data to create new
knowledge. In this case, it deduces new facts from facts that are being
collected in each visited point (node). Deduction is defined by set of rules
that are loaded from the text file.

Function hierarchy:
run_production                  - main function
    init_rules                  - loads rules from file
        _get_4_lines_from_file  - load one rule from 4 lines
    init_facts                  - loads facts from file
    run_forward_chain           - solution finder
        _find_actions           - finds all possible actions
            _expand             - labelling entities
        _remove_duplicates      - remove duplicate actions
        _apply_actions          - apply actions
    _save_facts                 - saves all facts into file
    _print_solution             - prints the solution
    _save_solution              - save the solution into file
"""

import json
import random
import re
from collections import namedtuple
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Tuple


def run_production(
    fname_save_facts: str,
    fname_load_facts: str,
    fname_load_rules: str,
    step_by_step: bool,
    facts_amount: int,
    randomize_facts_order: bool,
    fname: str,
) -> None:
    """Runs rule based production system with forward chaining.

    Runs forward chain to get new facts given by the rules and already known
    facts that we read from text files. At the end, it prints the solution into
    console, saves it into json file and also saves facts into text file.

    Args:
        fname_save_facts (str): name of file into which facts will be saved
        fname_load_facts (str): name of file from which we load facts
        fname_load_rules (str): name of file from which we load rules
        step_by_step (bool): enters one loaded fact by each production run
        facts_amount (int): number of facts we want to load (points)
        randomize_facts_order (bool): option to shuffle loaded facts
        fname (str): name of json file into which the solution will be saved
    """

    rules = init_rules(fname_load_rules)
    known_facts = init_facts(
        fname_load_facts, facts_amount, randomize_facts_order
    )

    if step_by_step:
        using_facts = []  # type: List[str]
        fact_discovery_flow = {}

        for known_fact in known_facts:
            using_facts += [known_fact]
            new_facts = run_forward_chain(using_facts, rules)
            fact_discovery_flow[known_fact] = new_facts

    else:
        using_facts = known_facts
        new_facts = run_forward_chain(using_facts, rules)
        fact_discovery_flow = {"All steps at once": new_facts}

    _save_facts(using_facts, fname_save_facts)
    _print_solution(fact_discovery_flow)
    _save_solution(fact_discovery_flow, fname)


def init_rules(fname_rules: str) -> List[Any]:
    """Loads rules from the file and initializes them to namedtuple structure.

    The rules are being loaded from /data/knowledge directory.

    Args:
        fname_rules (str): name of the file from which we load rules

    Returns:
        List[Any]: namedtuples with these attributes:
            name - name of the rule (unused)
            conds - conditions to fulfill actions
            acts - actions (message, add or remove fact from the set of facts)
    """

    Rules = namedtuple("Rules", "name conds acts")

    # regexp for loading a rule - name, conds and acts, each in one line
    patterns = [
        re.compile(r"\S+"),
        re.compile(r"((\?[A-Z]+)[^,]*, )*.*\?[A-Z].*"),
        re.compile(
            r"((add|remove|message) \?\w.*?, )*(add|remove|message).*\?\w.*"
        ),  # instead of ".*?," we could use "[^,]*,", or combine it "[^,]*?,"
    ]

    src_dir = Path(__file__).parents[0]
    fname_path = Path(f"{src_dir}/data/knowledge/{fname_rules}.txt")

    rules = []
    with open(fname_path, encoding="utf-8") as file:
        while rule := _get_4_lines_from_file(file):
            fields_amount = len(Rules._fields)
            for i in range(fields_amount):
                if not patterns[i].match(rule[i]):
                    print(Rules._fields[i], "field is set wrong!")
                    exit()

            rules.append(Rules(*rule))

    return rules


def init_facts(
    fname_facts: str, facts_amount: int, randomize_facts_order: bool
) -> List[str]:
    """Loads facts from the file, initializes amount and order of them.

    The facts are being loaded from /data/knowledge directory.

    Args:
        fname_facts (str): name of the file from which we load facts
        facts_amount (int): number of facts we want to load (points)
        randomize_facts_order (bool): option to shuffle loaded facts

    Returns:
        List[str]: known facts in sentences
    """

    src_dir = Path(__file__).parents[0]
    fname_path = Path(f"{src_dir}/data/knowledge/{fname_facts}.txt")

    with open(fname_path, encoding="utf-8") as file:
        facts = [fact.rstrip() for fact in file][:facts_amount]

    if randomize_facts_order:
        random.shuffle(facts)

    return facts


def _get_4_lines_from_file(file: Any) -> Tuple[Any, ...]:
    """Reads and prepares 3 lines from the file to get one rule.

    Args:
        file (file): opened file

    Returns:
        Tuple[str]: 3 lines that represent single rule
    """

    rule = tuple(line.rstrip("\n:") for line in islice(file, 4))

    # last line of file is not being read for some reason
    # need to skip last element (empty string) except for the last line
    if len(rule) == 4:
        rule = rule[:-1]

    return rule  # -> Tuple[str, str, str]


def run_forward_chain(known_facts: List[str], rules: List[Any]) -> List[str]:
    """Discovers new facts from the given known facts and rules.

    Runs forward chaining to find actions that updates our facts collection
    until there are no more actions to do.

    Args:
        known_facts (List[str]): facts that are going to be used
        rules (List[Any]): namedtuples with these attributes:
            name - name of the rule (unused)
            conds - conditions to fulfill actions
            acts - actions (message, add or remove fact from the set of facts)

    Returns:
        List[str]: newly found facts that have been added by action
    """

    new_facts = []
    while True:
        found_acts = _find_actions(rules, known_facts)
        acts = _remove_duplicates(found_acts, known_facts)
        if not acts:
            break

        new_facts += _apply_actions(acts, known_facts)

    return new_facts


def _find_actions(rules: List[Any], known_facts: List[str]) -> List[str]:
    """Finds all actions that can be done from the given rules and facts.

    Args:
        rules (List[Any]): namedtuples with these attributes:
            name - name of the rule (unused)
            conds - conditions to fulfill actions
            acts - actions (message, add or remove fact from the set of facts)
        known_facts (List[str]): facts that are going to be used

    Returns:
        List[str]: all actions that can be done
    """

    found_acts = []
    for rule in rules:
        cond_words = rule.conds.split()
        labelled_conds = _expand(cond_words, known_facts, {})
        if labelled_conds:
            acts = rule.acts.split(", ")

            for labelled_cond in labelled_conds:
                for act in acts:
                    act_type, act = act.split(" ", maxsplit=1)
                    for key, value in labelled_cond.items():
                        act = act.replace(key, value)
                    found_acts.append(act_type + " " + act)

    return found_acts


def _remove_duplicates(
    found_acts: List[str], known_facts: List[str]
) -> List[str]:
    """Removes those actions whose outcomes are already present in the facts.

    Args:
        found_acts (List[List[str]]): actions that were found
        known_facts (List[str]): facts that were used

    Returns:
        List[str]: applicable actions
    """

    applicable_acts = []
    acts = set()
    for found_act in found_acts:
        type_, act = found_act.split(" ", 1)
        if (
            (type_ == "add" and act not in known_facts)
            or (type_ == "remove" and act in known_facts)
        ) and (act not in acts):
            applicable_acts.append(found_act)
            acts.add(act)

    return applicable_acts


def _apply_actions(acts: List[str], known_facts: List[str]) -> List[str]:
    """Applies actions to update facts collection.

    Args:
        acts (List[str]): actions that we are about to perform
        known_facts (List[str]): known facts in sentences

    Returns:
        List[str]: updated facts collection
    """

    newly_found_facts = []
    for act in acts:
        type_, act = act.split(" ", 1)
        if type_ == "add":
            known_facts.append(act)
        elif type_ == "remove":
            known_facts.remove(act)
        newly_found_facts.append(act)

    return newly_found_facts


def _expand(
    cond_words: List[str], facts: List[str], labels: Dict[str, str]
) -> List[Dict[str, str]]:
    """Labels the entities from the given facts and conditions in a rule.

    Runs through the rule's conditions recursively and tries to label all
    entities. Entities must start with capitalized characters!

    Args:
        cond_words (List[str]): words of condition(s) in a rule
        facts (List[str]): known facts in sentences
        labels (Dict[str, str]): labels of entities (?X -> entity)

    Returns:
        List[Dict[str, str]]: found labels of entities for whole condition
    """

    if cond_words[0] == "<>":  # labels must be different
        if labels[cond_words[1]] == labels[cond_words[2]]:
            return []
        return [labels]

    found_labels = []
    for fact in facts:
        fact_words = fact.split()
        tmp_label = {}
        continue_ = True

        for i, (c_word, f_word) in enumerate(zip(cond_words, fact_words)):
            next_condition = c_word.endswith(",")
            c_word = c_word.rstrip(",") if next_condition else c_word

            # encountering label with entity
            if c_word.startswith("?") and f_word[0].isupper():
                if c_word not in labels:
                    if f_word not in labels.values():
                        tmp_label[c_word] = f_word  # saving label for entity

                    else:
                        continue_ = False  # entity already exist in labels
                elif labels[c_word] != f_word:
                    continue_ = False  # label already exist for other entity
            elif c_word != f_word:
                continue_ = False  # words in the sentence stopped matching

            if not continue_:
                break

            if next_condition:
                next_cond_words = cond_words[i + 1 :]
                found_labels += _expand(
                    next_cond_words, facts, {**labels, **tmp_label}
                )

        # new label found
        if continue_ and not next_condition:
            found_labels.append({**labels, **tmp_label})

    return found_labels


def _save_facts(facts: List[str], fname_save_facts: str) -> None:
    """Saves all facts into text file.

    Saves facts into /data/knowledge directory. If the directory does not
    exist, it will create one.

    Args:
        facts (List[List[str]]): all (known and found) facts that will be saved
        fname_save_facts (str): name of file into which we save facts
    """

    src_dir = Path(__file__).parents[0]
    knowledge_dir = Path(f"{src_dir}/data/knowledge")
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    fname_path = Path(f"{knowledge_dir}/{fname_save_facts}.txt")
    with open(fname_path, "w", encoding="utf-8") as file:
        file.write("\n".join(facts))


def _print_solution(fact_discovery_flow: Dict[str, List[str]]) -> None:
    """Prints the flow of finding out new facts into console.

    Args:
        fact_discovery_flow (Dict[str, List[str]]): path of discovering facts
    """

    for i, fact in enumerate(fact_discovery_flow, 1):
        print(f"{str(i)}:  {fact} -> " + ", ".join(fact_discovery_flow[fact]))


def _save_solution(
    fact_discovery_flow: Dict[str, List[str]], fname: str
) -> None:
    """Saves the flow of finding out new facts into json file.

    Saves the solution into /data/solutions directory. If the directory does
    not exist, it will create one.
    It is being used for gif visualization and viewing.

    Args:
        fact_discovery_flow (Dict[str, List[str]]): path of discovering facts
        fname (str): name of json file into which the solution will be saved
    """

    src_dir = Path(__file__).parents[0]
    solutions_dir = Path(f"{src_dir}/data/solutions")
    solutions_dir.mkdir(parents=True, exist_ok=True)

    fname_path = f"{solutions_dir}/{fname}_rule.json"
    with open(fname_path, "w", encoding="utf-8") as file:
        json.dump(fact_discovery_flow, file, indent=4)


if __name__ == "__main__":

    FNAME_SAVE_FACTS = "facts"
    FNAME_LOAD_FACTS = "facts_init"
    FNAME_LOAD_RULES = "rules"
    STEP_BY_STEP = True
    FACTS_AMOUNT = 10
    RANDOMIZE_FACTS_ORDER = False
    FNAME = "queried"

    chain_parameters = dict(
        fname_save_facts=FNAME_SAVE_FACTS,
        fname_load_facts=FNAME_LOAD_FACTS,
        fname_load_rules=FNAME_LOAD_RULES,
        step_by_step=STEP_BY_STEP,
        facts_amount=FACTS_AMOUNT,
        randomize_facts_order=RANDOMIZE_FACTS_ORDER,
        fname=FNAME,
    )  # type: Dict[str, Any]

    run_production(**chain_parameters)
