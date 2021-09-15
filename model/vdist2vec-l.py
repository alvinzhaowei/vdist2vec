import numpy as np
import tensorflow as tf
file_name = "./dg_shortest_distance_matrix.npy"
sdm = np.load(file_name)
maxLengthy = np.max(sdm)
sdm = sdm/maxLengthy
n= sdm.shape[0]

def get_node(index):
    node1_index = index / (n - 1)
    node2_index = index % (n - 1)
    return node1_index, node2_index


def get_batch(index_list):
    l = len(index_list)
    x1_batch = np.zeros((l,))
    x2_batch = np.zeros((l,))
    y_batch = np.zeros((l, 1))
    z = 0
    for i in index_list:
        node1, node2 = get_node(i)
        if node2 >= node1:
            node2 += 1
        x1_batch[z] = node1
        x2_batch[z] = node2
        y_batch[z] = sdm[node1][node2]
        z += 1
    return x1_batch, x2_batch, y_batch


# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = n
display_step = 1
input_l = (n - 1) * n

# Network Parameters
n_hidden_1 = int(n*0.2)
n_hidden_2 = 100
n_hidden_3 = 20
n_input = n
n_output = 1

# tf Graph input
x1 = tf.placeholder("int32", [None, ], name="x1")
x2 = tf.placeholder("int32", [None, ], name="x2")
y = tf.placeholder("float32", [None, n_output], name="y")
delta = tf.placeholder("float32", shape=[], name="delta")
lr = tf.placeholder("float32", shape=[])


def multilayer_perceptron(x1, x2, weights, biases):
    layer_11 = tf.gather(weights['h1'], x1)
    layer_12 = tf.gather(weights['h1'], x2)
    layer_1 = tf.concat([layer_11, layer_12], 1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    out_layer = tf.sigmoid(tf.add(tf.matmul(layer_3, weights['out']), biases['out']))
    return out_layer


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], mean=0.0, stddev=0.01, dtype=tf.float32), name='h1'),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1 * 2, n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32),
                      name='h2'),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32),
                      name='h3'),
    'out': tf.Variable(tf.truncated_normal([n_hidden_3, n_output], mean=0.0, stddev=0.01, dtype=tf.float32),
                       name='wout')
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1], mean=0.0, stddev=0.01, dtype=tf.float32), name='b1'),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32), name='b2'),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32), name='b3'),
    'out': tf.Variable(tf.truncated_normal([n_output], mean=0.0, stddev=0.01, dtype=tf.float32), name='bout')
}

# Construct model

pred = multilayer_perceptron(x1, x2, weights, biases)

# Define loss and optimizer
cost = tf.losses.huber_loss(y, pred, delta=delta)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
tf.add_to_collection("optimizer", optimizer)

# Initializing the variables
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

print ("graph has been built")

# Training cycle
theta = 0.5
for epoch in range(training_epochs):
    avg_cost = 0.
    epoch_cost = np.zeros((input_l, 1))
    total_batch = int(input_l // batch_size) + 1
    # Loop over all batches
    random_index = np.random.permutation(input_l)
    for j in range(total_batch):
        start = j * batch_size
        end = (j + 1) * batch_size
        if end >= input_l:
            end = input_l
        batch_x1, batch_x2, batch_y = get_batch(random_index[start:end])
        _, c, res = sess.run([optimizer, cost, pred], feed_dict={x1: batch_x1,
                                                                 x2: batch_x2,
                                                                 y: batch_y, delta: theta,lr:learning_rate})
        # Compute average loss

        epoch_cost[start:end] = res - batch_y
        avg_cost += c / total_batch
    # Display logs per epoch step
    epoch_cost = np.reshape(epoch_cost, (input_l,))
    epoch_cost = np.abs(epoch_cost)
    epoch_cost = np.sort(epoch_cost)

    percent = 0.99

    accumulate = 0
    total = np.sum(epoch_cost)
    theta = 0
    index_got = 0
    for i, row in enumerate(epoch_cost):
        accumulate = accumulate + row
        if accumulate >= total * percent:
            theta = row
            index_got = i
            break
    if if (epoch+1) % 5 == 0::
        learning_rate = learning_rate * 0.5
    save_path = saver.save(sess, "./model.ckpt")
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", \
              "{:.9f}".format(avg_cost))
print("Optimization Finished!")

# evaluation

def get_eval_batch(p1, p2):
    x1_batch = np.zeros(((p2 - p1), ))
    x2_batch = np.zeros(((p2 - p1), ))
    z = 0
    for j in range(p1, p2):
        node1, node2 = get_node(j)
        if node2 >= node1:
            node2 += 1
        x1_batch[z] = node1
        x2_batch[z] = node2
        z += 1
    return x1_batch, x2_batch


batch_size = 10000
total_batch = int(input_l//batch_size) + 1
result = []
real_dis = []
for i in range(total_batch):
    start = i * batch_size
    end = (i+1)*batch_size
    if end >= input_l:
        end = input_l
    batch_x1, batch_x2, batch_y = get_eval_batch(start, end)
    result_temp = sess.run(pred, feed_dict={x1: batch_x1, x2:batch_x2})
    result = np.append(result, result_temp)
	real_dis = np.append(real_dis, batch_y)

real_dis = real_dis * maxLengthy
result = result * maxLengthy

abe = np.fabs(real_dis - result)
re = abe/real_dis

mse = (abe ** 2).mean()
maxe = np.max(abe ** 2)
mine = np.min(abe ** 2)
mabe = abe.mean()
maxae = np.max(abe)
minae = np.min(abe)
mre = re.mean()
maxre = np.max(re)
minre = np.min(re)
print ("mean square error:", mse)
print ("max square error:", maxe)
print ("min square error:", mine)
print ("mean absolute error:", mabe)
print ("max absolute error:", maxae)
print ("min absolute error:", minae)
print ("mean relative error:", mre)
print ("max relative error:", maxre)
print ("min relative error:", minre)
