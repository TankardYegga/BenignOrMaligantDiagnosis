
import pickle
from re import T
import numpy as np
from regex import P

preds = np.asarray([1,0,1,1,0])
labels = np.asarray([1,0,0,0,1])

dict = {'preds': preds, 'labels': labels}

save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/sklearn_model/utils/dict.pickle'
with open(save_path, 'wb') as fp:
    pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open(save_path, 'rb') as fp:
    mydict = pickle.load(fp)

preds_arr = mydict['preds']
print(preds_arr)
print(type(preds_arr))