import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from noise_estimation import adjacency_list, position_data, fit_noise_probabilities, cython_bincount, split, build_molecule_graph



if __name__ == '__main__':
    print('Testing split, bicount using simple test:')
    print('Testing bincount...')
    test_cases = [
    (np.array([0, 1, 2, 2, 3, 3, 3]), 3),
    (np.array([1, 1, 1, 1]), 1),
    (np.array([0, 0, 0, 0]), 0),
    (np.array([2, 2, 3, 3, 3, 3]), 3)]

    for values, max_value in test_cases:
        expected_counts = np.bincount(values, minlength=max_value + 1)
        actual_counts = cython_bincount(values)
        print(actual_counts)
        assert np.array_equal(actual_counts, expected_counts), f"Counts do not match for values {values} with max_value {max_value}"

    print('Bincount ok')

    print('Testing split...')

    array = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    factor = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    max_factor = factor.max()

    splitted = split(array, factor, max_factor)
    expected = [
        np.array([], dtype=array.dtype),
        np.array([10, 20]),
        np.array([30, 40]),
        np.array([50, 60]),
        np.array([70, 80]),
    ]

    for i, (split_subarray, expected_subarray) in enumerate(zip(splitted, expected), start=0):
        if not np.array_equal(split_subarray, expected_subarray):
            print(f"Test failed for group {i}: expected {expected_subarray}, got {split_subarray}")
        else:
            print(f"Test passed for group {i}")
    print('Split ok')
    print("Testing using real data:")


    nn_id = 15
    test_data_path =  "/home/viktor_petukhov/mh/spatial/CellSegmentation/data/merfish_moffit/merfish_coords_adj.csv"
    df_spatial = pd.read_csv(test_data_path)

    pos_data = position_data(df_spatial).T
    tree = KDTree(pos_data.T)
    dists, indices = tree.query(pos_data.T, k=nn_id + 1, sort_results=True)
    mean_dists = dists[:, nn_id]
    adj_list = build_molecule_graph(pos_data)

    print('- testing adjacency_list ....')
    edge_list, adjacent_dists = adjacency_list(pos_data) # ok
    print('Adjacency list: ok')

    print('- testing split ....')
    split(edge_list[1, :], edge_list[0, :], pos_data.shape[1])
    print('Split: ok')

    print('- [MAIN] Testing noise estimation ....')
    assignment_probs_python, assignment_python, (d1, d2), max_diffs = fit_noise_probabilities(mean_dists, adj_list)
    print('- Noise estimation: ok')

