import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      # t = W[j,:].T * X[i] - W[y[i],:].T * X[i] + 1
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X.T[:,i]
        dW[:,y[i]] -= X.T[:,i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # Average gradients
  dW /= num_train

  # add regularization to gradient
  dW += reg * W   ## 3073 x 10


  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]     # num of training examples = 500
  num_classes = W.shape[1]   # num of classes to classify = 10
  delta = 1.0
  scores = X.dot(W)          # X (500, 3073), W (500, 10)
  z = np.arange(num_train)   # create array of 1 to 500
  correct_class_score = scores[z,y].reshape(num_train,1) # select from each row the correct class y
  # print scores.shape       # Scores (500, 10)
  # print len(y)
  L = np.maximum(0, scores - correct_class_score + delta)
  # remove the correct class
  L[z,y] = 0
  # print L
  loss = np.sum(L) / num_train + reg * np.sum(np.square(W))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  ## X = 500 x 3073
  ## L.shape = 500 x 10
  ## W.shape = 3073 x 10
  #L[z,y]

  # If value of L > 0 then 1 else 0
  L[L<0] = 0
  L[L>0] = 1

  # Exclude the value in correct class rows
  L[z,y] = 0
  # Sum the classes contributing to the loss
  L[z,y] = -1 * np.sum(L, axis = 1)

  dW = X.T.dot(L)
  dW /= num_train

  # print np.sum(L)
  #print dW.shape
  dW += W * reg
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
