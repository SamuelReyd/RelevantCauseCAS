def get_rouded_values(data, bins=20):
    bins = np.logspace(np.log10(data.min()), np.log10(data.max()), num=bins + 1)
    bin_indices = np.digitize(data, bins)
    bin_means = np.array([data[bin_indices == i+1].max() if i+1 in np.unique(bin_indices) else 0 for i in range(bin_indices.max())])
    return bin_means[bin_indices - 1]

def digitalize_cost(scores, n_bins=10):
    s = scores.copy()
    bins = np.logspace(np.log10(s[0].min()), np.log10(s[0].max()), num=n_bins + 1)
    bin_indices = np.digitize(s[0], bins)
    values = range(bin_indices.max())
    bin_edges = np.array([
        s[0][bin_indices == i+1].max() 
        if i+1 in np.unique(bin_indices) 
        else bins[i+1]
        for i in values
        ])
    bin_counts = np.array([
        s[0][bin_indices == i+1].size
        for i in values
    ])
    # print(bin_edges.shape, bin_count.shape)
    s[0] = bin_edges[bin_indices - 1]
    return s, bin_edges, bin_counts

def sort_causes_priority(scores, priorities):
    scores = scores.copy()
    # scores[0] = get_rouded_values(scores[0], bins=50)
    # order = [priorities[m] for m in Metrics]
    key = lambda i: (
        scores[priorities[0],i], 
        scores[priorities[1],i], 
        scores[priorities[2],i], 
    )
    # return sorted(range(scores.shape[1]), key=lambda i: (scores[0,i],test_scores[1,i], test_scores[2,i]))
    return sorted(range(scores.shape[1]), key=key)