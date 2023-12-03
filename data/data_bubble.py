import numpy as np

raw = np.load('data/Matriz_Snapshots_81.npz')

data = raw['Matrix']

# save

np.savez('data/data_bubble.npz', data = data)