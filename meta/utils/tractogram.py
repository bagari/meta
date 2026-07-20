import numpy as np
from copy import deepcopy
from dipy.tracking.streamlinespeed import set_number_of_points


def reorient_streamlines(streamlines, model_streamline, n_points=15):
    """
    Reorients a bundle of streamlines to match a model streamlines.

    Parameters:
        streamlines : subject streamlines
        model_streamline : Model streamlines
        n_points (int): Number of points to resample streamlines for comparison

    Returns:
        reoriented (list): List of reoriented subject streamlines
    """

    # Ensure inputs are lists of arrays if we have single streamline
    if isinstance(streamlines, np.ndarray) and streamlines.ndim == 2:
        streamlines = [streamlines]
    if isinstance(model_streamline, np.ndarray) and model_streamline.ndim == 2:
        model_streamline = [model_streamline]

    reoriented = deepcopy(streamlines)
    subject_array = set_number_of_points(streamlines, nb_points=n_points)
    model_array   = set_number_of_points(model_streamline, nb_points=n_points)

    for idx, sl in enumerate(subject_array):
        dist_direct = np.sum(np.linalg.norm(sl - model_array, axis=1))
        dist_flipped = np.sum(np.linalg.norm(sl[::-1] - model_array, axis=1))
        if dist_direct > dist_flipped:
            reoriented[idx] = reoriented[idx][::-1]

    return reoriented

