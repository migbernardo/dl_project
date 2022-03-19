import pickle


def read_data(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
