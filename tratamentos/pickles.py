import pickle

def criaPickle(data,name):
    data.to_pickle("pickles/"+name+".pkl")
    
def carregaPickle(name):
    with open('pickles/'+name+'.pkl', 'rb') as pickle_file: data = pickle.load(pickle_file)
    return data

def criarModelo(modelo,name):
    pickle.dump(modelo, open('pickles/'+name+'.sav', 'wb'))
    
def carregarModelo(name):
    return pickle.load(open('pickles/'+name+'.sav', 'rb'))