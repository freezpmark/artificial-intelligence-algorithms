import collections as col
from itertools import islice

# USE REGEXP TO VALIDATE TEXT FILES
# while True:
#     line = f1.readline()
#     if line.endswith(":"):
#         name = line
#         line = f1.readline()
#         if line.startswith("AK"):
#             cond = line
#         line = f1.readline()
#         if line.startswith("POTOM"):
#             act = line

#     name = f1.readline()
#     cond = f1.readline()
#     act = f1.readline()

#     rule = islice(infile, N)

# f1.readline()
# for i, line in enumerate(f1, 1):
#     if line.endswith(":"):
#         name = line
#     elif line.startswith("AK"):
#         cond = line
#     elif line.startswith("POTOM"):
#         act = line
#     elif not line:
#         if name and cond and act:
#             rules.append(Rules(name, cond, act))
#         else:
#             print("WRONG!")
#         # create rule
# rule = []
# for i, line in enumerate(f1, 1):
#     rule.append(line.rstrip())
#     if i % 3 == 0:
#         rules.append(Rules(*rule))
#         rule = []

# a=1

# while f1:
#     rules.append(Rule(*[line for line in f1][:3]))

# for rule in f1[:2:3]:
# for f1[::3]

# lines_gen = islice(f1, 3)
# lines = [line.rstrip() for line in lines_gen]

# line = f2.readline().rstrip()
# rules = []
# for i, line in enumerate(f1):
#     if i % 3 == 0:
#         pass  # append

# facts = []
# for line in f2:
#     line = line.rstrip()
#     # append..

# rule = [
#     part[1] if part[0].isupper() else part[0][:-1]
#     for line in islice(f1, 3)
#     for part in line.split(' \n', 1)
# ]


def runProduction():
    Rules = col.namedtuple("Rule", "name cond act")

    # ToDo: check whether file is in correct format with regexp
    with open("simulation/rules2.txt") as f1:
        rules = []
        while rule := [line.rstrip("\n:") for line in islice(f1, 3)]:
            rules.append(Rules(*rule))
            f1.readline()

    with open("simulation/facts.txt") as f2:
        facts = [fact.rstrip() for fact in f2]

    # we've loaded the files
    print(rules)
    print(facts)

    # iterate through facts that match current rule (strip names and then use in)


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
