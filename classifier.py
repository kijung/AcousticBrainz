import os
import sys
import re
import os.path
import json
from sklearn import svm, linear_model
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#import tensorflow as tf
import numpy as np
import pickle
import random

#logisticRegression
def classifySGD(train_features, train_labels, test_features, test_labels, genre = 'rock'):

    batch_size = 128
   #if train_features[0] is float
    size = len(train_features)
    num_labels = 2
    num_features = 1

    valid_dataset = np.array(train_features[(2*size)//3:]).astype(np.float32)
    valid_labels = np.array(train_labels[(2*size)//3:]).astype(np.float32)

    train_dataset = np.array(train_features[:2*size//3]).astype(np.float32)
    train_labels = np.array(train_labels[:2*size//3]).astype(np.float32)
    
    test_dataset = np.array(test_features).astype(np.float32)
    test_labels = np.array(test_labels).astype(np.float32)

    train_labels = (np.arange(num_labels) == train_labels[:,None]).astype(np.float32)
    valid_labels = (np.arange(num_labels) == valid_labels[:,None]).astype(np.float32)
    test_labels = (np.arange(num_labels) == test_labels[:, None]).astype(np.float32)
    if isinstance(train_dataset[0], np.ndarray):
        num_features = len(train_dataset[0])
    
    graph = tf.Graph()
    with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, num_features))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights = tf.Variable(
        tf.truncated_normal([num_features, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
    
    num_steps = 5501
    test_accur = 0
    valid_accur = 0
    steps = 0
    valid_accur_list = []
    count = 0
    test_predictions = 0
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        #print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run(
              [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                #print("Minibatch loss at step %d: %f" % (step, l))
                #print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                #print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
                
                #print(accuracy(valid_prediction.eval(), valid_labels))
                c = accuracy(valid_prediction.eval(), valid_labels)
                valid_accur_list.append(c)
                if valid_accur < c:
                    valid_accur = c
                    steps = step
                    test_predictions = test_prediction.eval()
                    accur = accuracy(test_prediction.eval(), test_labels)
                
        #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
        #accur = accuracy(test_prediction.eval(), test_labels)
        #for f in test_prediction.eval():
        #    if f[1] == 1:
        #        count+=1
        #print(valid_accur, steps)
        #print(valid_prediction.eval())  
    #print(accur, steps)
    for f in test_predictions:
        if np.argmax(f) == 1:
            count+=1
    print(genre, count)
    return valid_accur_list, accur, test_predictions*accur

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def classifySklearn(train_features, train_labels, test_features, test_labels, genre = 'rock', classifier = RandomForestClassifier(n_estimators = 25)):
    size = len(train_features)
    train_size = (3 * size) //4 
    classifier.fit(train_features[:train_size], train_labels[:train_size])
    

    predictions = classifier.predict_proba(test_features)
    return classifier.score(train_features[train_size:], train_labels[train_size:]), classifier.score(test_features, test_labels), predictions
    #returning validation score, test score, predictions

"""
def classifyRFC(train_features, train_labels, test_features, test_labels, genre = 'rock'):
    rfc = RandomForestClassifier(n_estimators = 25)
    rfc.fit(train_features, train_labels)

    predictions = rfc.predict_proba(test_features, test_labels)
    return rfc.score(test_features, test_labels)

def classifySVM(train_features, train_labels, test_features, test_labels, genre = 'rock'):
    clf = svm.SVC()
    clf.fit(train_features, train_labels)

    predictions = clf.predict(test_features, test_labels)
    return clf.score(test_features, test_labels)

def classifyLR(train_features, train_labels, test_features, test_labels, genre = 'rock'):
    lr = linear_model.LogisticRegression(C=1e5)
    lr.fit(train_features, train_labels)

    predictions = lr.predict_proba(test_features, test_labels)
    return lr.score(test_features, test_labels)

def classifykNR(train_features, train_labels, test_features, test_labels, genre = 'rock'):
    knC = KNeighborsClassifier(13)
    knC.fit(train_features, train_labels)

    predictions = knC.predict_proba(test_features, test_labels)
    return knC.score(test_features, test_labels)

def classifyAD(train_features, train_labels, test_features, test_labels, genre = 'rock'):
    bdt = AdaBoostClassifier()
    bdt.fit(train_features, train_labels)

    predictions = bdt.predict_proba(test_features, test_labels)
    return bdt.score(test_features, test_labels)
"""
