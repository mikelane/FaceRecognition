
# coding: utf-8

# In[2]:

from datetime import datetime
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import PIL
import numpy as np

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# In[3]:

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print('n_samples: {ns}'.format(ns=n_samples))
print('n_features: {nf}'.format(nf=n_features))
print('n_classes: {}'.format(n_classes))


# In[4]:

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=np.random.RandomState())


# In[5]:

n_components = 150

print("Extracting the top {nc} eigenfaces from {nf} faces".format(nc=n_components, nf=X_train.shape[0]))
start = datetime.now()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print("done in {dur:.3f}s".format(dur=(datetime.now() - start).total_seconds()))

eigenfaces = pca.components_.reshape((n_components, h, w))

print('Projecting the input data on the eigenfaces orthonormal basis')
start = datetime.now()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in {dur:.3f}s".format(dur=(datetime.now() - start).total_seconds()))


# In[6]:

print('Fitting the classifier to the training set')
start = datetime.now()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'solver': ['newton-cg', 'sag', 'lbfgs']}
clf = GridSearchCV(LogisticRegression(), param_grid)
clf = clf.fit(X_train_pca, y_train)
print('done in {dur:.3f}s'.format(dur=(datetime.now() - start).total_seconds()))
print('Best estimator found by grid search:')
print(clf.best_estimator_)


# In[7]:

print("Predicting people's names on the test set")
start = datetime.now()
y_pred = clf.predict(X_test_pca)
print("done in {dur:.3f}s".format(dur=(datetime.now() - start).total_seconds()))

print('\nAccuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# In[8]:

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    plt.style.use('seaborn-dark')
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def title(y_pred, y_test, target_names, i):
    """Helper function to extract the prediction titles"""
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: {p}\ntrue:      {t}'.format(p=pred_name, t=true_name)


# In[9]:

prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface {}".format(i) for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()


# In[ ]:



