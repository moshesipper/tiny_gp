# Tiny Genetic Programming in Python

A minimalistic program implementing Koza-style (tree-based) genetic programming to solve a symbolic regression problem. 

**tiny-gp.py** is a basic (and fully functional) version, which produces textual output of the evolutionary progression and evolved trees.

**tiny-gp-plus.py** displays dynamic graphs of error and mean tree size (size = number of nodes), has a bloat-control option, and produces nicer, graphic output (you'll need to install https://pypi.org/project/graphviz/).

If you wish to cite this:
```
@misc{Sipper2019tinyGP,
  author = {Sipper, M.},
  title = {Tiny Genetic Programming in Python},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/moshesipper/tiny_gp} }
}
```

| Symbolic | Regression using GP  |
|-------------:|:-------------| 
| Objective | Find an expression with one input (independent variable x), whose output equals the value of the quartic function  x<sup>4</sup> + x<sup>3</sup> + x<sup>2</sup> + x + 1 |
| Function set | add, sub, mul |   
| Terminal set | x, -2, -1, 0, 1, 2  |   
| Fitness | Inverse mean absolute error over a dataset of 101 target values, normalized to [0,1]
| Paremeters | POP_SIZE (population size), MIN_DEPTH (minimal initial random tree depth), MAX_DEPTH (maximal initial random tree depth), GENERATIONS (maximal number of generations), TOURNAMENT_SIZE (size of tournament for tournament selection), XO_RATE (crossover rate), PROB_MUTATION (per-node mutation probability) |
| Termination | Maximal number of generations reached or an individual with fitness = 1.0 found |

| Evolved solution | Another evolved solution |
|-------------|-------------| 
| ![GPTree](https://github.com/moshesipper/tiny-gp/blob/master/Figures/GPTree.png) | ![GPTree2](https://github.com/moshesipper/tiny-gp/blob/master/Figures/GPTree2.png) |

| Bloat control  | No bloat control |
|-------------|-------------| 
|![GP run](https://github.com/moshesipper/tiny-gp/blob/master/Figures/with_bloat_control.png) | ![GP run](https://github.com/moshesipper/tiny-gp/blob/master/Figures/without_bloat_control.png) |
