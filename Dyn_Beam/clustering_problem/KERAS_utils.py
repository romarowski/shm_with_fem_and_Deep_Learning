def prepare_data(dataset, loc, start_index, end_index, 
                 history_size, target_size):
    #Prepares time-series data from dataset, where dataset is a matrix whose 
    #lines represent time and columns are feature (sensor). 
    import numpy as np
    
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
      end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
      indices = range(i-history_size, i)
      # Reshape data from (history_size,) to (history_size, 1)
      data.append(dataset[np.ix_(indices, loc)])
      labels.append(dataset[i+target_size, -1])
    return np.array(data), np.array(labels)
