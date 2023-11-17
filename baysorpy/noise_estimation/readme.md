To compile Cython program:
```
python setup.py build_ext -i 
```

After compiling, run 
```
python _test_noise_estimation.py
```

If all the test passed, you can further use the library for your tasks. Usage example:

```
from noise_estimation import fit_noise_probabilities, cython_bincount, build_molecule_graph
```

If you modify the original .pyx code, please make sure that the tests are passed. 


