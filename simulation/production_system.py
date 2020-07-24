def runProduction():
    with open("rules.txt") as f1, open("facts.txt") as f2:
        rules = []
        for i, line in enumerate(f1):
            if i % 3 == 0:
                pass  # append

        facts = []
        for line in f2:
            line = line.rstrip()
            # append..


if __name__ == "__main__":
    runProduction()

    # save rules: rule -> rule -> rule -> rule: struct(name, cond, action) 3 sentences
    # save facts: fact -> fact -> fact -> fact: struct(<cond>) 1 sentence

    # ITERATE through to-be FACTS
    #     ITERATE through RULES
    #         ITERATE through RULES' BRACKETS (recursively)
    #             ITERATE through FACTS to fill ?X's (everywhere) and ?Y etc.
    #         Save action to be processed (actions are reseted after each action)

    # () with <> means they have to be different
    # (they can be the same among siblings! Parent can have 2x the same child)
    pass
