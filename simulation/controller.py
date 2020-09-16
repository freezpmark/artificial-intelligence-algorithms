import model.evolution as evo
import model.forward_chain as chain
import model.pathfinding as path
import view

if __name__ == "__main__":

    # map creation parameters
    begin_create = "walls"
    # walls uses: query, fname, max_runs, points_amount
    # terrain uses: fname, max_runs, points_amount
    # properties uses: fname, points_amount
    query = "10x12 (1,5) (2,1) (3,4) (4,2) (6,8) (6,9)"
    fname = "queried"
    max_runs = 1
    points_amount = 10

    # pathfinding parameters
    movement = "M"
    climb = False
    algorithm = "HK"
    subset_size = None

    # production parameters
    save_fname_facts = "facts"
    load_fname_facts = "facts_init"
    load_fname_rules = "rules"
    step_by_step = True
    facts_random_order = True

    # view parameters
    skip_rake = False

    evo_parameters = dict(
        begin_create=begin_create,
        query=query,
        fname=fname,
        max_runs=max_runs,
        points_amount=points_amount,
    )

    path_parameters = dict(
        fname=fname,
        movement=movement,
        climb=climb,
        algorithm=algorithm,
        subset_size=subset_size,
    )

    chain_parameters = dict(
        save_fname_facts=save_fname_facts,
        load_fname_facts=load_fname_facts,
        load_fname_rules=load_fname_rules,
        step_by_step=step_by_step,
        facts_amount=points_amount + 1,
        facts_random_order=facts_random_order,
        fname=fname,
    )

    evo.createMaps(**evo_parameters)
    path.findShortestPath(**path_parameters)
    chain.runProduction(**chain_parameters)

    view.createGif(fname, skip_rake, climb)

# TODO: remember previous fitness attempts
# TODO: optimize creating props
# TODO: create tests
