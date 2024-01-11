import pickle

# Replace 'D:\\Projekti\\parking spot detection\\regions.p' with your actual pickle file path
pickle_file_path = 'D:\\Projekti\\parking spot detection\\regions.p'

# Load data from the pickle file
with open(pickle_file_path, 'rb') as file:
    pickle_data = pickle.load(file)



#print(pickle_data)
print(pickle_data[0])