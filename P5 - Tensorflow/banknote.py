"""
Author: https://staging4.aicorespot.io/produce-a-neural-network-for-banknote-authentication/
Author: Mateusz Budzisz

k-fold cross-validation of base model for the banknote dataset
"""
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
df = read_csv('data_banknote_authentication.csv', header=None)

# Split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]

# Ensure all data are floating point values
X = X.astype('float32')

# Encode strings to integer
y = LabelEncoder().fit_transform(y)

# Prepare cross validation
kfold = StratifiedKFold(10)

# Enumerate splits
scores = list()

for train_ix, test_ix in kfold.split(X, y):
    # Split data
    X_train, X_test, y_train, y_test = X[train_ix], X[test_ix], y[train_ix], y[test_ix]

    # Determine the number of input features
    n_features = X.shape[1]

    # Define model
    model = Sequential()
    model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Fit the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # Predict test set
    yhat = (model.predict(X_test) > 0.5).astype("int32")

    # Evaluate predictions
    score = accuracy_score(y_test, yhat)

    print('>%.3f' % score)
    scores.append(score)

# Summarize all scores
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
