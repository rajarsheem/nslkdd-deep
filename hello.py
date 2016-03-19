from sklearn import linear_model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, precision_score
import numpy as np

# model = LogisticRegression(C=1,class_weight='balanced',n_jobs=-1)
model = linear_model.SGDClassifier(class_weight='balanced', n_jobs=-1)

new_train_data = np.load('out/new_input.npz')['train']
new_test_data = np.load('out/new_input.npz')['test']

model.fit(new_train_data[:, :-1], new_train_data[:, -1])

preds = model.predict(new_test_data[:, :-1])

print('fitted')
print(accuracy_score(new_test_data[:,-1], preds))
print(confusion_matrix(new_test_data[:,-1], preds))
print(precision_score(new_test_data[:,-1], preds, average='macro'))