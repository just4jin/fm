'''
Datasets: ml-100k contains the files u.item (list of movie ids and titles) 
and u.data (list of user_id, movie_id, rating, timestamp)
'''

import numpy as np
from sklearn.feature_extraction import DictVectorizer
import pylibfm

# Read in data
def loadData(filename,path="ml-100k/"):
    data = []
    y = []
    users=set()
    items=set()
    with open(path+filename) as f:
        for line in f:
            (user,movieid,rating,ts)=line.split('\t')
            data.append({ "user_id": str(user), "movie_id": str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)y = 

    return (data, np.array(y), users, items)

(train_data, y_train, train_users, train_items) = loadData("ua.base")
(test_data, y_test, test_users, test_items) = loadData("ua.test")
v = DictVectorizer()
X_train = v.fit_transform(train_data)
X_test = v.transform(test_data)

# Build and train a Factorization Machine
fm = pylibfm.FM(num_factors=10, num_iter=10, verbose=True, task="regression", initial_learning_rate=0.01, learning_rate_schedule="constant")

fm.fit(X_train,y_train)
'''
#Creating validation dataset of 0.01 of training for adaptive regularization
#-- Epoch 1
#Training RMSE: 0.49640
#-- Epoch 2
#Training RMSE: 0.44941
#-- Epoch 3
#Training RMSE: 0.44191
#-- Epoch 4
#Training RMSE: 0.44001
#-- Epoch 5
#Training RMSE: 0.44044
#-- Epoch 6
#Training RMSE: 0.44539
#-- Epoch 7
#Training RMSE: 0.45032
#-- Epoch 8
#Training RMSE: 0.43750
#-- Epoch 9
#Training RMSE: 0.43542
#-- Epoch 10
#Training RMSE: 0.43527
'''

# Evaluate test sets
preds = fm.predict(X_test)
from sklearn.metrics import mean_squared_error
print "FM RMSE: %.4f" % mean_squared_error(y_test,preds)
#FM RMSE: 0.9253
