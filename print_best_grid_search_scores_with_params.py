"""
Prints the best results of a sklearn grid search along with the standard deviation and the parameter settings.

Args:
    grid_search: This is an instance of GridSearchCV from sklearn.model_selection which is already fitted.
    n: This is the number of how many best scores should be displayed, by default 5.

Raises:
    KeyError: Raises an exception if the grid_search argument is not fitted.
"""
def print_best_grid_search_scores_with_params(grid_search, n=5):
    if not hasattr(grid_search, 'best_score_'):
        raise KeyError('grid_search is not fitted.')
    print(grid_search.best_score_)
    print("Grid scores on validation set:")
    indexes = np.argsort(grid_search.cv_results_['mean_test_score'])[::-1][:n]
    means = grid_search.cv_results_['mean_test_score'][indexes]
    stds = grid_search.cv_results_['std_test_score'][indexes]
    params = np.array(grid_search.cv_results_['params'])[indexes]
    for mean, std, params in zip(means, stds, params):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
