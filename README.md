<img src="https://github.com/FrizzLi/Artificial-Intelligence/blob/master/animation.gif"/>

## Artificial-intelligence
Created an app that produces GIF animation that visualizes: creation of map in Zen garden approach, shortest path visiting all destinations, deducing new facts from facts that are being collected at certain destinations. The intention behind this project was to improve Python programming skills while practising implementation of some of the most widely mentioned algorithms in AI that don't use machine learning techniques. (Python)

## Features
### Evolutionary algorithm - evolution.py
The task is to rake the sand on the entire Zen garden. Character always starts at the edge of the garden and leaves straight shaping path until it meets another edge or obstacle. On the edge, character can walk as he pleases. If it comes to an obstacle - wall or raked sand - he has to turn around, if he is not able to, raking is over.  

##### <b>Parameters:</b>
 - begin_create - defines whether to start creating new maps from query, walled map or terrained to propertied map
 - query - defines map size with walls
 - fname - name of the text file into which the map will be created
 - max_runs - max number of times to run evolutionary algorithm to find a solution
 - points_amount - amount of blue nodes (facts) we wish to create randomly in the map

### Pathfinding - pathfinding.py
Finds the shortest path to visit all nodes. First node to visit is the black node for which A* algorithm is used. Then, we run Dijkstra's algorithm for each blue node to find the shortest distances to all other blue nodes. To find the shortest path between all blue nodes we can use either greedy Naive permutation or Held–Karp algorithm which is alot faster.

##### <b>Parameters:</b>
 - movement - defines options of movement
 - climb - defines the way how the distance between adjacent nodes is calculated
 - algorithm - choses which algorithm to use to find the shortest path between blue nodes
 - subset_size - amount of blue nodes we wish to visit in the map

### Production rule system - forward_chain.py
Production system belongs to knowledge systems that use data to create new knowledge. In this case, it deduces new facts from facts that are being collected in each blue node. Deduction is defined by set of rules that are loaded from the text file.

##### <b>Parameters:</b>
 - save_fname_facts - name of file into which facts will be saved
 - load_fname_facts - name of file from which we load facts
 - load_fname_rules - name of file from which we load rules
 - step_by_step - defines whether we want to run production for each collected fact
 - facts_random_order - shuffles the order of loaded facts

### GIF file creation - view.py
##### <b>Parameters:</b>
  - skip_rake - defines whether we want to skip raking part in the animation

## Installation
 - pillow (to create the animation)
```
$ pip install pillow
```
 - OR if you wish to install the whole environment I use, just execute setup.bat. It creates venv, activates it, install environment stuff and opens up Visual Studio Code.


<!--- old description
((In progress) App in which the intention is to cover all fundamental functionalities and principles of Python. Serves me for learning and practising Python and algorithms.)
### Production rule system - forward_chain.py
Production system belongs to knowledge systems that use data to create new knowledge. Knowledge is not only expressing information about an object, but also links between objects, properties of selected problems and ways to ﬁnd solutions. Therefore, the knowledge system is in the simplest case a pair - program that can generally manipulate with knowledge and knowledge base that describes the problem and relationships that apply there.
-->
