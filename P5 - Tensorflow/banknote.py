"""
Author: Akash-Singh539 (https://github.com/Akash-Singh539/Bank-notes-Authentication-using-tensorflow/tree/master)
Author: Mateusz Budzisz

pip install pandas seaborn sklearn tensorflow numpy
"""
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv("data_banknote_authentication.csv", header=None)
# Get the dimensions of the dataset
print("Dimensions: ", data.shape)

# Checking the data and gathering information on it
data.head()
data.info()
data.describe()

# Exploring the dataset
# Checking for the number of real and fake notes in the dataset
data['Class'] = data.index
sns.countplot(x='Class', data=data)

# Looking for relation between the real and fake note to see any visible trends
sns.pairplot(data=data, hue='Class')

# Data Preparation
# Standardising the data by rows without class from the dataset
bank_notes_without_class = data.drop('Class', axis=1)

# Using StandardScaler object from the Scikit-learn library on the independent variables
# Storing the transformed data in a new dataframe
scaler = StandardScaler()
scaler.fit(bank_notes_without_class)
scaled_features = pd.DataFrame(data=scaler.transform(bank_notes_without_class),
                               columns=bank_notes_without_class.columns)

scaled_features.head()

# Rename 'Class' to 'Authentic'
data = data.rename(columns={'Class': 'Authentic'})

# Forged class
data.loc[data['Authentic'] == 0, 'Forged'] = 1
data.loc[data['Authentic'] == 1, 'Forged'] = 0

# X and y
X = scaled_features
y = data[['Authentic', 'Forged']]
# Convert X and y to Numpy arrays
X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train.shape: ", X_train.shape)
print("Y_train.shape: ", y_train.shape)

learning_rate = 0.01
training_epochs = 100
batch_size = 100

n_hidden_1 = 4  # nodes in first hidden layer
n_hidden_2 = 4  # nodes in second hidden layer
n_input = 4  # input shape
n_classes = 2  # total classes (authentic / forged)
n_samples = X_train.shape[0]  # # samples

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

"""
x: Placeholder for data input
weights: Dictionary of weights
biases: Dictionary of biases

"""
layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
layer_1 = tf.nn.relu(layer_1)

layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
layer_2 = tf.nn.relu(layer_2)

predictions = tf.matmul(layer_2, weights['out'] + biases['out'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predictions))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Training the network
# The outer loop runs the epochs
# The inner loop runs the batches for each epoch
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
costs = []
for epoch in range(training_epochs):
    avg_cost = 0.0
    total_batch = int(n_samples / batch_size)
    for batch in range(total_batch):
        batch_x = X_train[batch * batch_size: (1 + batch) * batch_size]
        batch_y = y_train[batch * batch_size: (1 + batch) * batch_size]
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
        avg_cost += c / total_batch

    print("Epoch: {} cost={:.4f}".format(epoch + 1, avg_cost))
    costs.append(avg_cost)

print("Model has completed {} epochs of training.".format(training_epochs))

# Saver object to save our model
saver = tf.train.Saver()
saver.save(sess, "banknote.h5", save_format='h5')

correct_predictions = tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1)), tf.float32)

# Making predictions on the test data
accuracy = tf.reduce_mean(correct_predictions)
print("Accuracy:", accuracy.eval(feed_dict={x: X_test, y: y_test}))
