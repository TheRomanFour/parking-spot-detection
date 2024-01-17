import pickle

pickle_file_path = 'regions_parking2.p'

with open(pickle_file_path, 'rb') as file:
    pickle_data = pickle.load(file)



#print(pickle_data)
print(pickle_data)