import tensorflow as tf
import numpy as np

class siemese():

    def __init__(self, weights = None):
        self.weights = np.load(weights)
        self.x1 = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.x2 = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.y_ = tf.placeholder(tf.float32, [None])
        with tf.variable_scope("siemese") as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)

        # Create loss
        self.loss = self.compute_contrastive_loss()

    def network(self, x):
        parameter = []
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            x = x - mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            #print(weights['conv1_1_W'])
            # weights['conv1_1_W']
            kernel = tf.get_variable(initializer=self.weights['conv1_1_W'], name='w1_1')
            conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=self.weights['conv1_1_b'],
                                 trainable=True, name='b1_1')
            out = tf.nn.bias_add(conv, biases)
            output_conv1_1 = tf.nn.relu(out, name=scope)
            # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.get_variable(initializer=self.weights['conv1_2_W'], name='w1_2')
            conv = tf.nn.conv2d(output_conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=self.weights['conv1_2_b'],
                                 trainable=True, name='b1_2')
            out = tf.nn.bias_add(conv, biases)
            output_conv1_2 = tf.nn.relu(out, name=scope)

        # pool1
        pool1 = tf.nn.max_pool(output_conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.get_variable(initializer=self.weights['conv2_1_W'], name='w2_1')
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=self.weights['conv2_1_b'],
                                 trainable=True, name='b2_1')
            out = tf.nn.bias_add(conv, biases)
            output_conv2_1 = tf.nn.relu(out, name=scope)

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.get_variable(initializer=self.weights['conv2_2_W'], name='w2_2')
            conv = tf.nn.conv2d(output_conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=self.weights['conv2_2_b'],
                                 trainable=True, name='b2_2')
            out = tf.nn.bias_add(conv, biases)
            output_conv2_2 = tf.nn.relu(out, name=scope)


        # pool2
        pool2 = tf.nn.max_pool(output_conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.get_variable(initializer=self.weights['conv3_1_W'], name='w3_1')
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=self.weights['conv3_1_b'],
                                 trainable=True, name='b3_1')
            out = tf.nn.bias_add(conv, biases)
            output_conv3_1 = tf.nn.relu(out, name=scope)

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.get_variable(initializer=self.weights['conv3_2_W'], name='w3_2')
            conv = tf.nn.conv2d(output_conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=self.weights['conv3_2_b'],
                                 trainable=True, name='b3_2')
            out = tf.nn.bias_add(conv, biases)
            output_conv3_2 = tf.nn.relu(out, name=scope)


        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.get_variable(initializer=self.weights['conv3_3_W'], name='w3_3')
            conv = tf.nn.conv2d(output_conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=self.weights['conv3_3_b'],
                                 trainable=True, name='b3_3')
            out = tf.nn.bias_add(conv, biases)
            output_conv3_3 = tf.nn.relu(out, name=scope)


        # pool3
        pool3 = tf.nn.max_pool(output_conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.get_variable(initializer=self.weights['conv4_1_W'], name='w4_1')
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=self.weights['conv4_1_b'],
                                 trainable=True, name='b4_1')
            out = tf.nn.bias_add(conv, biases)
            output_conv4_1 = tf.nn.relu(out, name=scope)


        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.get_variable(initializer=self.weights['conv4_2_W'], name='w4_2')
            conv = tf.nn.conv2d(output_conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=self.weights['conv4_2_b'],
                                 trainable=True, name='b4_2')
            out = tf.nn.bias_add(conv, biases)
            output_conv4_2 = tf.nn.relu(out, name=scope)


        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.get_variable(initializer=self.weights['conv4_3_W'], name='w4_3')
            conv = tf.nn.conv2d(output_conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=self.weights['conv4_3_b'],
                                 trainable=True, name='b4_3')
            out = tf.nn.bias_add(conv, biases)
            output_conv4_3 = tf.nn.relu(out, name=scope)


        # pool4
        pool4 = tf.nn.max_pool(output_conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.get_variable(initializer=self.weights['conv5_1_W'], name='w5_1')
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=self.weights['conv5_1_b'],
                                 trainable=True, name='b5_1')
            out = tf.nn.bias_add(conv, biases)
            output_conv5_1 = tf.nn.relu(out, name=scope)


        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.get_variable(initializer=self.weights['conv5_2_W'], name='w5_2')
            conv = tf.nn.conv2d(output_conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=self.weights['conv5_2_b'],
                                 trainable=True, name='b5_2')
            out = tf.nn.bias_add(conv, biases)
            output_conv5_2 = tf.nn.relu(out, name=scope)


        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.get_variable(initializer=self.weights['conv5_3_W'], name='w5_3')
            conv = tf.nn.conv2d(output_conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=self.weights['conv5_3_b'],
                                 trainable=True, name='b5_3')
            out = tf.nn.bias_add(conv, biases)
            output_conv5_3 = tf.nn.relu(out, name=scope)


        # pool5
        pool5 = tf.nn.max_pool(output_conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(pool5.get_shape()[1:]))
            fc1w = tf.get_variable(initializer=self.weights['fc6_W'], name='w6')
            fc1b = tf.get_variable(initializer=self.weights['fc6_b'],
                                 trainable=True, name='b6')
            pool5_flat = tf.reshape(pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            fc1 = tf.nn.relu(fc1l)

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.get_variable(initializer=self.weights['fc7_W'], name='w7')
            fc2b = tf.get_variable(initializer=self.weights['fc7_b'],
                                 trainable=True, name='b7')
            fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
            logits= tf.nn.relu(fc2l)

        return logits
    def compute_contrastive_loss(self):
        margin = 1.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.mul(labels_f, tf.sub(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.mul(labels_f, tf.maximum(0.0, tf.sub(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def loss_with_step(self):
        margin = 1.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
        neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss
