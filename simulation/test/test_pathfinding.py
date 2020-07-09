from simulation import pathfinding as pat

def test_getMoves():
    assert len(pat.getMoves('DSDS')) == 8
    assert len(pat.getMoves('ASDS')) == 4
    assert len(pat.getMoves('D')) == 2