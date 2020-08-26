import model.evolution as evo
import model.forward_chain as chain
import model.pathfinding as path

if __name__ == "__main__":

    points = 10
    evo_parameters = dict(
        max_runs=3,
        points_amount=points,
        export_name="queried",
        query="10x12 (1,5) (2,1) (3,4) (4,2) (6,8) (6,9)",
    )

    path_parameters = dict(
        fname="queried",
        movement="M",
        climb=True,
        algorithm="HK",
        subset_size=None,
    )

    chain_parameters = dict(
        save_fname_facts="facts",
        load_fname_facts="facts_init",
        load_fname_rules="rules",
        step_by_step=True,
        facts_amount=points + 1,
        facts_random_order=True,
    )

    evo.runEvolution(evo_parameters)
    paths, map_properties = path.runPathfinding(path_parameters)
    chain.runProduction(chain_parameters)

# ToDo: Pytest tests (validations types for parameters)
# ToDo: Gif visualization
