import tensorflow as tf
sess = tf.InteractiveSession()

raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13.]
spike = tf.Variable(False)
spike.initializer.run()

for i in range(1, len(raw_data)):
    tf.assign(
        spike,
        raw_data[i] - raw_data[i - 1] > 5
    ).eval()
    print('Spike', spike.eval())

sess.close()
