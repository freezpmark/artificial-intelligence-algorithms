import stage_1_ai_evolution
import stage_2_ai_pathfinding
import stage_3_ai_forward_chain
import stage_4_view


if __name__ == "__main__":

    # walls:       fname
    # terrain:     fname, points_amount, climb
    # properties:  fname, points_amount
    # view:        fname,                climb
    shared_fname = "queried"
    shared_points_amount = 10
    shared_climb = False

    # harder variant: "10x12 (1,5) (2,1) (3,4) (4,2) (6,8) (6,9) (6,7)"
    evo_query = "10x12 (1,5) (2,1) (3,4) (4,2) (6,8) (6,9) (6,9)"
    evo_begin_create = "walls"
    evo_max_runs = 3

    path_movement_type = "M"
    path_algorithm = "HK"

    chain_fname_save_facts = "facts"
    chain_fname_load_facts = "facts_init"
    chain_fname_load_rules = "rules"
    chain_step_by_step = True
    chain_randomize_facts_order = False

    view_skip_rake = False

    evo_parameters = dict(
        fname=shared_fname,
        begin_from=evo_begin_create,
        query=evo_query,
        max_runs=evo_max_runs,
        points_amount=shared_points_amount,
    )

    path_parameters = dict(
        fname=shared_fname,
        movement_type=path_movement_type,
        climb=shared_climb,
        algorithm=path_algorithm,
        visit_points_amount=shared_points_amount,
    )

    chain_parameters = dict(
        fname_save_facts=chain_fname_save_facts,
        fname_load_facts=chain_fname_load_facts,
        fname_load_rules=chain_fname_load_rules,
        step_by_step=chain_step_by_step,
        facts_amount=shared_points_amount,
        randomize_facts_order=chain_randomize_facts_order,
        fname=shared_fname,
    )

    view_parameters = dict(
        fname=shared_fname,
        skip_rake=view_skip_rake,
        climb=shared_climb,
    )

    stage_1_ai_evolution.create_maps(**evo_parameters)
    stage_2_ai_pathfinding.find_shortest_path(**path_parameters)
    stage_3_ai_forward_chain.run_production(**chain_parameters)
    stage_4_view.create_gif(**view_parameters)
