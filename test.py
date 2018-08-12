# -*- coding:utf-8 -*-

import tldextract
import pickle
from zipfile import ZipFile
from keras.models import load_model


def get_alexa(num, filename='top-1m.csv'):

    zipfile = ZipFile('top-1m.csv.zip')
    return [tldextract.extract(x.split(',')[1]).domain for x in \
            zipfile.read(filename).split()[:num]]
    

 
results = pickle.load(open('traindata.pkl', 'rb'))

'''
for i in results:
    with open('traindata.txt', 'a') as f:
        f.write(i[0] + '  ' + i[1] + '\n')
    
'''

model = load_model('model.h5')

print(model)
