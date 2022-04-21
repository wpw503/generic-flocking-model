# Generic Flocking Model
A generic flocking model based upon Reynolds' boids. 

This repo has been optimised to replicate the behaviours of the [MTI](https://doi.org/10.1371/journal.pone.0035615) and [DMBSMF](https://doi.org/10.1088/1361-6463/aa942f) models, but it can be modified to replicate other models.

## Optimising for different models

Implement a fitness function for a new model in the `evaluation` directory, modify the file `models/generic_multi_MLP_boid.py` to use the relevant inputs, and modify `train_model.py` to use your new fitness function and model. (You may also need to implement different definitions of neighbourhoods in the `neighbourhoods` directory.)

Finally, run `train_model.py` until a satisfactory level of optimisation is reached. The final outcome can be displayed by editing the `display.py` function to use your new model.

## Requirements
- DEAP
- pygame
- numpy
- scipy
- sklearn