
# coding: utf-8

# In[1]:


from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
np.set_printoptions(threshold='nan')


# In[2]:


def conv2d(x, output_num):
    return tf.layers.conv2d(inputs = x, filters = output_num, kernel_size=(3,3), padding='VALID', activation = tf.nn.relu)

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def batch_norm(input_data, is_Training):
    return tf.contrib.layers.batch_norm(input_data, is_training= is_Training,updates_collections = None, decay=0.99)

def batch_norm_wrapper(inputs, is_Training, decay = 0.9):
    
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    epsilon = 1e-3
        
    def if_true():
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon), epsilon
        
    def if_false():
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon), pop_mean
    
    result = tf.cond(is_Training, if_true, if_false)
    return result




# In[3]:


def load_file(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_image(filename):
    #filenames = [filename]
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image = tf.image.decode_png(value, channels=3)
    return image

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# In[4]:


CONTEXT_LENGTH = 100
#DATA Process
tokenizer = tf.contrib.keras.preprocessing.text.Tokenizer(filters='', split=" ", lower=False)

X = []
#X.append(load_file("/Users/luhaoyang/Documents/file/3da13a50-8b49-11e8-b025-ed54bda3c788/page_copy.html"))
X.append(load_file("page.html"))

for i in range(2,7):
    X.append(load_file("page" + str(i) + ".html"))
#X.append(load_file("86.html"))

tokenizer.fit_on_texts(X)

VOCAB_SIZE = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(X)
max_length = max(len(s) for s in sequences)
print (max_length)

feature_maps = []
feature_map1 = np.load("image_feature_zhihu_vbig.npy")
feature_maps.append(feature_map1)

for i in range(2,7):
    feature_maps.append(np.load("zhihu_image_feauture" + str(i) + ".npy"))

X, y,image_data = list(), list(),list()
Test_token = list()
Test_label = list()
Test_image = list()

for image_no, seq in enumerate(sequences):
    if image_no != 5:
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]     
            # If the sentence is shorter than max_length, fill it up with empty words
            in_seq = tf.contrib.keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=max_length)[0]
            # Map the output to one-hot encoding
            out_seq = tf.contrib.keras.utils.to_categorical([out_seq], num_classes=VOCAB_SIZE)[0]
            # Cut the input sentence to 100 tokens, and add it to the input data
            X.append(in_seq[-CONTEXT_LENGTH:])
            y.append(out_seq)
            image_data.append(np.squeeze(feature_maps[image_no]))
    else:
        for i in range(1,len(seq)):
            in_seq, out_seq = seq[:i], seq[i]     
            # If the sentence is shorter than max_length, fill it up with empty words
            in_seq = tf.contrib.keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=max_length)[0]
            # Map the output to one-hot encoding
            out_seq = tf.contrib.keras.utils.to_categorical([out_seq], num_classes=VOCAB_SIZE)[0]
            # Cut the input sentence to 100 tokens, and add it to the input data
            Test_token.append(in_seq[-CONTEXT_LENGTH:])
            Test_label.append(out_seq)
            Test_image.append(np.squeeze(feature_maps[image_no]))
            

X,y,image_data  = np.array(X), np.array(y), np.array(image_data)
Test_token, Test_label,Test_image = np.array(Test_token), np.array(Test_label), np.array(Test_image)


X = np.expand_dims(X, axis=2)
Test_token = np.expand_dims(Test_token,axis=2)

# In[5]:



EPOCH = 400

global_step = tf.Variable(0, trainable=False)

decay_learning_rate = tf.train.exponential_decay(0.00005,
                                           global_step=global_step,
                                           decay_steps= 1110,decay_rate=0.98, staircase=True)


batch_size = tf.placeholder(tf.int64)
token, y_label, x_image = tf.placeholder(tf.float32, shape=[None, CONTEXT_LENGTH, 1]), tf.placeholder(tf.float32, shape=[None,VOCAB_SIZE]),tf.placeholder(tf.float32,[None,14,14,1536])
dataset = tf.data.Dataset.from_tensor_slices((token, y_label, x_image)).apply(tf.contrib.data.batch_and_drop_remainder(batch_size)).repeat()
just_for_placehold = tf.placeholder(tf.float32, shape = [None,CONTEXT_LENGTH,1])
Training_Phase = tf.placeholder(tf.bool, name="is_training")
#image_Data = tf.placeholder(tf.float32, shape = [1,])

itera = dataset.make_initializable_iterator()
databatch = itera.get_next()


def build_LSTM_language(input_data):
    hidden_size = 256
    num_layers = 2
    keep_prob = 0.5
    input_data = tf.cast(input_data,tf.float32)
    
    with tf.variable_scope("LSTM_language"):
        
        layers = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=None)
             for _ in range(num_layers)], state_is_tuple=True)
        init_state = layers.zero_state(tf.shape(input_data)[0], dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(layers, inputs=input_data, initial_state=init_state, time_major=False)
    
    return outputs, state


def build_decoded(input_data):
    hidden_size = 512
    num_layers = 2
    keep_prob = 0.5
    input_data = tf.cast(input_data,tf.float32)
    
    with tf.variable_scope("LSTM_decoder"):
    
        layers = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=None)
             for _ in range(num_layers)], state_is_tuple=True)
        init_state = layers.zero_state(tf.shape(input_data)[0], dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(layers, inputs=input_data, initial_state=init_state, time_major=False)
        
    return outputs[:,-1,:]

print (databatch[0].shape)
print (databatch[1].shape)


image_info = databatch[2]
image_info = tf.reshape(image_info, [tf.shape(just_for_placehold)[0], 14 * 14 * 1536])

fc1 = tf.layers.dense(image_info, 1024, activation = tf.nn.relu)

final = tf.tile(fc1,[1, CONTEXT_LENGTH])
output_image = tf.reshape(final, [tf.shape(just_for_placehold)[0],CONTEXT_LENGTH,1024])
output_image = tf.cast(output_image,tf.float32)

output_HTML, _ = build_LSTM_language(databatch[0])
output_HTML = tf.cast(output_HTML,tf.float32)
print ("LSTM:", output_HTML.shape)
print ("above is the progress of 20180719")

feed_decoder_data = tf.concat([output_image,output_HTML],-1)
#print feed_decoder_data.shape

decode_output = build_decoded(feed_decoder_data)
print (decode_output.shape)

fc3 = tf.layers.dense(decode_output, 1024, activation = tf.nn.relu)
#fc3, _ = batch_norm_wrapper(fc3, Training_Phase)
fc3 = tf.contrib.layers.layer_norm(fc3)


fc4 = tf.layers.dense(fc3, 1024, activation = tf.nn.relu)
fc4 = tf.contrib.layers.layer_norm(fc4)

final_output = tf.layers.dense(fc4, VOCAB_SIZE)
#print (final_output.shape)
print ("final_output", final_output.shape)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=databatch[1]))

train_op =  tf.train.AdamOptimizer(learning_rate = decay_learning_rate, epsilon = 0.0001).minimize(cost,global_step = global_step)


predict = tf.nn.softmax(final_output,1)
correct_predict = tf.equal(tf.argmax(predict,1), tf.argmax(databatch[1],1))
acc = tf.reduce_mean(tf.cast(correct_predict, "float"))



init = tf.global_variables_initializer()
with tf.Session() as sess:
    n_batches = 370
    sess.run(init)
    sess.run(itera.initializer, feed_dict={token: X, y_label: y, batch_size: 64, x_image:image_data})
    for i in range(EPOCH): 
        tot_loss = 0
        print("lr:", sess.run(decay_learning_rate))
        for j in range(n_batches):
            _, loss_value = sess.run([train_op, cost], feed_dict={just_for_placehold: np.zeros([64,CONTEXT_LENGTH,1]), Training_Phase: True})
            #print ("Loss_val:", loss_value)
            tot_loss += loss_value
            #hm = sess.run(image_data, feed_dict = {just_for_placehold: np.zeros([1,CONTEXT_LENGTH,1]), Training_Phase: True})
            #print("hm",hm)        
            if (j % 10 == 0):
                print ("epoch:%i batch:%i cost:%f" % (i,  j, loss_value))
        print("Iter: {}, Loss: {:.4f}".format(i, tot_loss / n_batches))
        print("")
    
    
    print ("above is the progress of 20180720")
         
    
    sess.run(itera.initializer, feed_dict={token: X, y_label: y, batch_size: 64, x_image:image_data})
    Training_accu = 0
    
    for _ in range(n_batches):
        train_acc = acc.eval(feed_dict = {just_for_placehold: np.zeros([64,CONTEXT_LENGTH,1]), Training_Phase: False})
        #popu = sess.run(pop_Mean, feed_dict = {just_for_placehold: np.zeros([64,CONTEXT_LENGTH,1]), Training_Phase: False})
        #print('popu mean:', popu)
        Training_accu += train_acc
        
    print("Train accu", Training_accu/n_batches)
    
    
    test_batches = 72
    sess.run(itera.initializer, feed_dict = {token: Test_token, y_label: Test_label, batch_size:64, x_image:Test_image} )
    Test_accu = 0
    for _ in range(test_batches):
        test_acc = acc.eval(feed_dict = {just_for_placehold: np.zeros([64,CONTEXT_LENGTH,1]), Training_Phase: False})
        #popu = sess.run(pop_Mean, feed_dict = {just_for_placehold: np.zeros([64,CONTEXT_LENGTH,1]), Training_Phase: False})
        #print('popu mean:', popu)
        Test_accu += test_acc
        
    print("Test accu", Test_accu/test_batches)
        
        
        
    in_text = 'START'
    # iterate over the whole length of the sequence
    for i in range(5000):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0][-CONTEXT_LENGTH:]
        # pad input
        sequence = tf.contrib.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=CONTEXT_LENGTH)
        # predict next word
        sequence = tf.tile(sequence,[1,1])
        #print ("sequence2", sequence.shape)
        sequence = tf.reshape(sequence,[1,CONTEXT_LENGTH,1])
        sequence = sess.run(sequence)
        #print ("sequence", list(sequence))
        sess.run(itera.initializer, feed_dict={token: sequence, y_label: 0 * np.random.rand(1,VOCAB_SIZE), x_image:np.expand_dims(Test_image[0],axis=0), batch_size: 1})
        yhat = sess.run(final_output, feed_dict={just_for_placehold: np.zeros([1,CONTEXT_LENGTH,1]), Training_Phase: False})

        # convert probability to integer
        yhat = np.argmax(yhat)
        #print ("yhat", yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        #print ("predicted_word:",word)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # Print the prediction
        print(' ' + word, end='')
        # stop if we predict the end of the sequence
        if word == 'END':
            break
    
    print ("Progress of 20180723")
    sess.close()
    