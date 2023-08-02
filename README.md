# artificial-inteligence-algorithms

<div>
 <img width="650" src="https://github.com/freezpmark/artificial-intelligence-algorithms/blob/00a8b5c6a6641d228324dc4622620d46034493eb/data/last_frame_162.png"/>
 <img align="top" src="https://github.com/freezpmark/artificial-intelligence-algorithms/blob/00a8b5c6a6641d228324dc4622620d46034493eb/data/queried.gif" width="350"/>
</div>


Produces GIF animation that visualizes: creation of map in Zen garden approach, shortest path visiting all destinations, deducing new facts from facts that are being collected at certain destinations. The intention behind this project was to improve Python coding skills while practising implementation of some of the most widely mentioned algorithms in AI that don't use machine learning techniques.

## Usage:
 - `git clone https://github.com/freezpmark/artificial-intelligence-algorithms`
 - `cd artificial-intelligence-algorithms`
 - `pip install -r requirements.txt`
 - `python main.py`

You can also run this on Discord. It uses `ai_algo.py` as cog file. To run it, call `<yourPrefix>run_ai`. The GIF file will be shown in the text channel where the command has ran.  
Running the `main.py` or `ai_algo.py` causes to execute feature scripts below in order.

## Features:
 - [stage_1_ai_evolution](#stage_1_ai_evolution)
 - [stage_2_ai_pathfinding](#stage_2_ai_pathfinding)
 - [stage_3_ai_forward_chain](#stage_2_ai_pathfinding)
 - [stage_4_view](#stage_4_view)

### stage_1_ai_evolution
The task is to rake the sand on the entire Zen garden. Character always starts at the edge of the garden and leaves straight shaping path until it meets another edge or obstacle. On the edge, character can walk as he pleases. If it comes to an obstacle - wall or raked sand - he has to turn around, if he is not able to, raking is over.  

#### <b>Parameters:</b>
 - `begin_create` - defines whether to start creating new maps from query, walled map or terrained to propertied map
 - `query` - defines map size with walls
 - `fname` - name of the text file into which the map will be created
 - `max_runs` - max number of times to run evolutionary algorithm to find a solution
 - `points_amount` - amount of blue nodes (facts) we wish to create randomly in the map

### stage_2_ai_pathfinding
Finds the shortest path to visit all nodes. First node to visit is the black node for which A* algorithm is used. Then, we run Dijkstra's algorithm for each blue node to find the shortest distances to all other blue nodes. To find the shortest path between all blue nodes we can use either greedy Naive permutation or Held–Karp algorithm which is alot faster.

#### <b>Parameters:</b>
 - `movement` - defines options of movement
 - `climb` - defines the way how the distance between adjacent nodes is calculated
 - `algorithm` - choses which algorithm to use to find the shortest path between blue nodes
 - `subset_size` - amount of blue nodes we wish to visit in the map

### stage_3_ai_forward_chain (Production rule system)
Production system belongs to knowledge systems that use data to create new knowledge. In this case, it deduces new facts from facts that are being collected in each blue node. Deduction is defined by set of rules that are loaded from the text file.

#### <b>Parameters:</b>
 - `save_fname_facts` - name of file into which facts will be saved
 - `load_fname_facts` - name of file from which we load facts
 - `load_fname_rules` - name of file from which we load rules
 - `step_by_step` - defines whether we want to run production for each collected fact
 - `facts_random_order` - shuffles the order of loaded facts

### stage_4_view (GIF file creation)
##### <b>Parameters:</b>
  - `skip_rake` - defines whether we want to skip raking part in the animation
