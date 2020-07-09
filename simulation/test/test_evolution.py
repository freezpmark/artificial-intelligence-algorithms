from simulation import evolution as evo

def test_listToTuple():
    assert evo.listToTuple([[0, 1], [1, 1]]) == {(0, 0): 0, (0, 0): 1, (1, 0): 1, (1, 1): 1}