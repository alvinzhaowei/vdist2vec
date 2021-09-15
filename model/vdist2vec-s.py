import numpy as np
import tensorflow as tf
file_name = "./dg_shortest_distance_matrix.npy"
sdm = np.load(file_name)
maxLengthy = np.max(sdm)

n = sdm.shape[0]


def get_node(index):
    node1_index = index // (n - 1)
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
lr = tf.placeholder("float32")


def multilayer_perceptron(x1, x2, weights, biases):
    # embedding layer
    layer_11 = tf.gather(weights['h1'], x1)
    layer_12 = tf.gather(weights['h1'], x2)
    layer_1 = tf.concat([layer_11, layer_12], 1)

    # first MLP (0,100)
    layer_21 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h21']), biases['b21']))
    layer_31 = tf.nn.relu(tf.add(tf.matmul(layer_21, weights['h31']), biases['b31']))
    out_layer1 = tf.sigmoid(tf.add(tf.matmul(layer_31, weights['out1']), biases['out1']))

    # second MLP (0,900)
    layer_22 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h22']), biases['b22']))
    layer_32 = tf.nn.relu(tf.add(tf.matmul(layer_21, weights['h32']), biases['b32']))
    out_layer2 = tf.sigmoid(tf.add(tf.matmul(layer_31, weights['out2']), biases['out2']))

    # third MLP (0,9000)
    layer_23 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h23']), biases['b23']))
    layer_33 = tf.nn.relu(tf.add(tf.matmul(layer_21, weights['h33']), biases['b33']))
    out_layer3 = tf.sigmoid(tf.add(tf.matmul(layer_31, weights['out3']), biases['out3']))

    # forth MLP (10000, )
    layer_24 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h24']), biases['b24']))
    layer_34 = tf.nn.relu(tf.add(tf.matmul(layer_21, weights['h34']), biases['b34']))
    out_layer4 = tf.sigmoid(tf.add(tf.matmul(layer_31, weights['out4']), biases['out4']))

    out_layer = 100 * out_layer1 + 900 * out_layer2 + 9000 * out_layer3 + (maxLengthy - 10000) * out_layer4

    return out_layer


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], mean=0.0, stddev=0.01, dtype=tf.float32), name='h1'),

    'h21': tf.Variable(tf.truncated_normal([n_hidden_1 * 2, n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32),
                       name='h21'),
    'h31': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32),
                       name='h31'),
    'out1': tf.Variable(tf.truncated_normal([n_hidden_3, n_output], mean=0.0, stddev=0.01, dtype=tf.float32),
                        name='wout1'),

    'h22': tf.Variable(tf.truncated_normal([n_hidden_1 * 2, n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32),
                       name='h22'),
    'h32': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32),
                       name='h32'),
    'out2': tf.Variable(tf.truncated_normal([n_hidden_3, n_output], mean=0.0, stddev=0.01, dtype=tf.float32),
                        name='wout2'),

    'h23': tf.Variable(tf.truncated_normal([n_hidden_1 * 2, n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32),
                       name='h23'),
    'h33': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32),
                       name='h33'),
    'out3': tf.Variable(tf.truncated_normal([n_hidden_3, n_output], mean=0.0, stddev=0.01, dtype=tf.float32),
                        name='wout3'),

    'h24': tf.Variable(tf.truncated_normal([n_hidden_1 * 2, n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32),
                       name='h24'),
    'h34': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32),
                       name='h34'),
    'out4': tf.Variable(tf.truncated_normal([n_hidden_3, n_output], mean=0.0, stddev=0.01, dtype=tf.float32),
                        name='wout4'),
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1], mean=0.0, stddev=0.01, dtype=tf.float32), name='b1'),

    'b21': tf.Variable(tf.truncated_normal([n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32), name='b21'),
    'b31': tf.Variable(tf.truncated_normal([n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32), name='b31'),
    'out1': tf.Variable(tf.truncated_normal([n_output], mean=0.0, stddev=0.01, dtype=tf.float32), name='bout1'),

    'b22': tf.Variable(tf.truncated_normal([n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32), name='b22'),
    'b32': tf.Variable(tf.truncated_normal([n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32), name='b32'),
    'out2': tf.Variable(tf.truncated_normal([n_output], mean=0.0, stddev=0.01, dtype=tf.float32), name='bout2'),

    'b23': tf.Variable(tf.truncated_normal([n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32), name='b23'),
    'b33': tf.Variable(tf.truncated_normal([n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32), name='b33'),
    'out3': tf.Variable(tf.truncated_normal([n_output], mean=0.0, stddev=0.01, dtype=tf.float32), name='bout3'),

    'b24': tf.Variable(tf.truncated_normal([n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32), name='b24'),
    'b34': tf.Variable(tf.truncated_normal([n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32), name='b34'),
    'out4': tf.Variable(tf.truncated_normal([n_output], mean=0.0, stddev=0.01, dtype=tf.float32), name='bout4'),
}

# Construct model

pred = multilayer_perceptron(x1, x2, weights, biases)

# Define loss and optimizer
cost = tf.losses.mean_squared_error(y, pred)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
tf.add_to_collection("optimizer", optimizer)

# Initializing the variables
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

print ("graph has been built")

# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(input_l // batch_size) + 1
    # Loop over all batches
    random_index = np.random.permutation(input_l)
    for j in range(total_batch):
        start = j * batch_size
        end = (j + 1) * batch_size
        if end >= input_l:
            end = input_l
        batch_x1, batch_x2, batch_y = get_batch(random_index[start:end])
        _, c = sess.run([optimizer, cost], feed_dict={x1: batch_x1,
                                                      x2: batch_x2,
                                                      y: batch_y,
                                                      lr: learning_rate})
        # Compute average loss
        avg_cost += c / total_batch
    # Display logs per epoch step
    if (epoch+1) % 5 == 0::
        learning_rate = learning_rate * 0.5
    save_path = saver.save(sess, "./ensembling_models/model.ckpt")
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
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

