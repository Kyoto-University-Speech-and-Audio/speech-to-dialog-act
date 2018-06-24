import tensorflow as tf
from csp.models.cells.CLSTMCell import CLSTMCell

def test_clstm():
    print(CLSTMCell)
    with tf.Session() as sess:
        context = tf.constant(0.5, shape=[10, 5])
        inputs = tf.constant(0.5, shape=[10, 20, 3])
        cell = CLSTMCell(10, context)
        output, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        sess.run([tf.global_variables_initializer()])
        for i in range(10):
            o = sess.run([output])
        print(o)

test_clstm()