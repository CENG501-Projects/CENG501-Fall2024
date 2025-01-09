def compute_salience_scores(salience_map, num_subsets=10):
    """
    Compute salience scores for subsets of pixels in the salience map.

    Args:
        salience_map (numpy array): The salience map as a 2D numpy array.
        num_subsets (int): The number of subsets to divide the pixels into.

    Returns:
        list: Salience scores for each subset.
    """
    
    flattened_map = salience_map.flatten()
    
    
    sorted_indices = np.argsort(-flattened_map)  
    sorted_salience = flattened_map[sorted_indices]
    
    
    total_pixels = len(sorted_salience)
    subset_size = total_pixels // num_subsets
    salience_scores = []

    for i in range(num_subsets):
        
        start_idx = i * subset_size
        end_idx = (i + 1) * subset_size if i < num_subsets - 1 else total_pixels
        subset = sorted_salience[start_idx:end_idx]

        
        salience_score = subset.sum()
        salience_scores.append(float(salience_score))  

    return salience_scores
