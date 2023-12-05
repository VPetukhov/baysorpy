To compile Cython program:
```
python setup.py build_ext -i 
```
To estimate confidence or clusters, use the following imports.


Note: first, you need to compile the cython part before importing the modules. If the error "module X not found", you need to run setup.py first.

```
from processing.molecule_clustering import estimate_molecule_clusters
from processing.noise_estimation import estimate_confidence
```

A usage example can be accessed by running ```simple_test_mol_clust.ipynb```. You have to adjust your ```test_data_path``` to a spatial dataset containing x,y,z coordinates.

