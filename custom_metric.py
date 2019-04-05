from keras import backend as K
import numpy as np

# Custom all or nothing accuracy metric (rounded prediction labels).
'''
- Rounds the predicted labels to the nearest integer (either 0 or 1)
- Elementwise equivilance check on rounded predicted labels against the actual labels.
- For each set of labels finds the product, resulting in either a 1 or 0 value depending upon whether axis 1 of the is_equal tensor is completely 1's or not respectively.
- Find the mean for the 1 dimentional axis of either correct or incorrect sets of labels in the batch.
'''
def rounded_all_or_nothing_acc(y_true, y_pred):
  y_pred = K.round(y_pred)
  is_equal = K.cast(K.equal(y_pred, y_true), "float32")
  prod = K.prod(is_equal, axis=1)
  return K.mean(prod)

if __name__ == '__main__':
  for a in range(2*3):
    for b in range(2*3):
      true = np.array([[a], [b]], dtype=np.uint8)
      true = np.unpackbits(true, axis=1)[:,-3:]
      pred = np.random.normal(loc=0.5, scale=0.1, size=(2,3))
      print("true: {},\n\npred: {}\n".format(true, pred))
      result = rounded_all_or_nothing_acc(K.variable(true), K.variable(pred))
      result = K.eval(result)
      print("result: {}\n".format(result))