from . import graph
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import sklearn
import scipy.sparse
import numpy as np
import os, time, collections, shutil


# Common methods for all models
class base_model(object):
    
    def __init__(self):
        self.regularizers = []
    
    # High-level interface which runs the constructed computational graph.
        
    def evaluate(self, data, labels, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """
        t_process, t_wall = time.process_time(), time.time()
        predictions, loss = self.predict(data, labels, sess)
        ncorrects = sum(predictions == labels)
        accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')
        string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}, loss: {:.2e}'.format(
                accuracy, ncorrects, len(labels), f1, loss)
        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall)
        return string, accuracy, f1, loss
    
    def evaluate2(self, data, labels, model, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the sensitivity, specificity, positive predictive value, and negative predictive value
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """
        sess = self._get_session(sess)
        t_process, t_wall = time.process_time(), time.time()
        predictions, loss = self.predict(data, labels, sess)
        ncorrects = sum(predictions == labels)
        accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        
        print(predictions)
        print(labels)
        true_positive=0
        false_positive=0
        true_negative=0
        false_negative=0
        for i in range(len(labels)):

            if (predictions[i] == 1):
                if (predictions[i] == labels[i]):
                    true_positive += 1
                else:
                    false_positive += 1
    
            if (predictions[i] == 0):
                if (predictions[i] == labels[i]):
                    true_negative += 1
                else:
                    false_negative += 1
        print(true_positive,false_positive, true_negative, false_negative)
        
        # Workaround for
        # UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
        if (true_positive == 0 and false_positive == 0):
            ppv=-1.0
        else:
            ppv = true_positive / (true_positive + false_positive) * 100

        if (true_negative == 0 and false_negative == 0):
            npv=-1.0
        else:
            npv = true_negative / (true_negative + false_negative) * 100

        sensitivity = true_positive / (true_positive + false_negative) * 100
        specificity = true_negative / (true_negative + false_positive) * 100

        string = 'test accuracy: {:.2f} ({:d} / {:d}), sensitivity: {:.2f}, specificity: {:.2f}, ppv: {:.2f}, npv: {:.2f}'.format(
                accuracy, ncorrects, len(labels), sensitivity, specificity, ppv, npv)
        


        # Embed
        path=os.path.join(self._get_path('checkpoints'), 'test')
        self.op_saver.save(sess, path)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = self.embedding.name
        embedding.metadata_path = os.path.join(os.getcwd(), 'test_' + model + '_labels.tsv')
        writer = tf.summary.FileWriter(path, self.graph)
        projector.visualize_embeddings(writer, config)

        return string
    
    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    # Methods to construct the computational graph.
    
    def chebyshev5(self, x, L, Fout, K):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        print('Chebyshev; N:',N,'M:',M,'Fin:',Fin,'Fout:',Fout,'K:',K)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=False)

        #
        #   Save coefficients as an image
        #
        x_image = tf.expand_dims(W,0)
        x_image = tf.expand_dims(x_image,-1)
        tf.summary.image('coefficients',x_image)
        print(x.shape)
        print(W.shape)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def b2relu(self, x):
        """Bias and ReLU. One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def mpool1(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            #tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def apool1(self, x, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.avg_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def inference(self, data, data2, dropout):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples) ---> 
            M: number of vertices (features) ---> 86
        training: we may want to discriminate the two, e.g. for dropout.
            True: the model is built for training.
            False: the model is built for evaluation.
        """
        # TODO: optimizations for sparse data
        logits = self._inference(data, data2, dropout)  #data2 for FC layer
        return logits
    
    def probabilities(self, logits):
        """Return the probability of a sample to belong to each class."""
        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):
        """Return the predicted classes."""
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, axis=1)
            return prediction

    def loss(self, logits, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                labels = tf.to_int64(labels)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                cross_entropy = tf.reduce_mean(cross_entropy)
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = cross_entropy + regularization
            
            # Summaries for TensorBoard.
            tf.summary.scalar('loss/cross_entropy', cross_entropy)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([cross_entropy, regularization, loss])
                tf.summary.scalar('loss/avg/cross_entropy', averages.average(cross_entropy))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average
    
    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                        learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if momentum == 0:
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    # Helper methods.

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        with tf.name_scope('weights.'):
            tf.summary.histogram(var.op.name, var)
            # print('_weights:', var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Model Type 1 & Type 2
class cgcnn(base_model):
    """
    Graph CNN which uses the Chebyshev approximation.

    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.

    L: List of Graph Laplacians. Size M x M. One per coarsening level.

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.
    
    The following are choices of implementation for various blocks.
        filter: filtering operation, e.g. chebyshev5, lanczos2 etc.
        brelu: bias and relu, e.g. b1relu or b2relu.
        pool: pooling, e.g. mpool1.
    
    Training parameters:
        num_epochs:    Number of training epochs.
        learning_rate: Initial learning rate.
        decay_rate:    Base of exponential decay. No decay with 1.
        decay_steps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """
    def __init__(self, L, F, K, p, M, covar, filter='chebyshev5', brelu='b1relu', pool='mpool1',
                num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                regularization=0, dropout=0, batch_size=100, eval_frequency=200,
                dir_name=''):
        super().__init__()   #ensures inheretence
        
        # Verify the consistency w.r.t. the number of layers.
        assert len(L) >= len(F) == len(K) == len(p)
        assert np.all(np.array(p) >= 1)
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.
        assert len(L) >= 1 + np.sum(p_log2)  # Enough coarsening levels for pool sizes.
        
        # Keep the useful Laplacians only. May be zero.
        M_0 = L[0].shape[0]

        j = 0
        self.L = []
        for pp in p:
            self.L.append(L[j])
            j += int(np.log2(pp)) if pp > 1 else 0
        L = self.L
        
        # Print information about NN architecture.
        Ngconv = len(p)
        Nfc = len(M)
        print('NN architecture')
        print('  input: M_0 = {}'.format(M_0))
        for i in range(Ngconv):
            print('dreage')
            print('  layer {0}: cgconv{0}'.format(i+1))
            print('    representation: M_{0} * F_{1} / p_{1} = {2} * {3} / {4} = {5}'.format(
                    i, i+1, L[i].shape[0], F[i], p[i], L[i].shape[0]*F[i]//p[i]))
            F_last = F[i-1] if i > 0 else 1
            print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                    i, i+1, F_last, F[i], K[i], F_last*F[i]*K[i]))
            if brelu == 'b1relu':
                print('    biases: F_{} = {}'.format(i+1, F[i]))
            elif brelu == 'b2relu':
                print('    biases: M_{0} * F_{0} = {1} * {2} = {3}'.format(
                        i+1, L[i].shape[0], F[i], L[i].shape[0]*F[i]))
        for i in range(Nfc):
            name = 'logits (softmax)' if i == Nfc-1 else 'fc{}'.format(i+1)
            print('  layer {}: {}'.format(Ngconv+i+1, name))
            print('    representation: M_{} = {}'.format(Ngconv+i+1, M[i]))
            M_last = M[i-1] if i > 0 else M_0 if Ngconv == 0 else L[-1].shape[0] * F[-1] // p[-1]
            print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                    Ngconv+i, Ngconv+i+1, M_last, M[i], M_last*M[i]))
            print('    biases: M_{} = {}'.format(Ngconv+i+1, M[i]))
        
        # Store attributes and bind operations.
        self.L, self.F, self.K, self.p, self.M = L, F, K, p, M
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)
        self.covar = covar #use covariates or not

        # Build the computational graph.
        self.build_graph(M_0)

    def build_graph(self, M_0):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_atrophy = tf.placeholder(tf.float32, (self.batch_size, M_0), 'atrophy')
                self.ph_data2 = tf.placeholder(tf.float32, (self.batch_size,8), 'data2')
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')

            # Model.
            op_logits = self.inference(self.ph_atrophy, self.ph_data2, self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                    self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(op_logits)

            # Embedding.
            with tf.name_scope('fc'):
                self.ph_embed = tf.placeholder(tf.float32, name='embed')
                self.embedding = tf.get_variable('embedding', initializer=np.zeros((1,1),dtype = np.float32), validate_shape=False)
                self.op_assignment = tf.assign(self.embedding, self.ph_embed, name='assignment', validate_shape=False)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()
            
            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)
        
        self.graph.finalize()

    def _inference(self, x, x2, dropout): #[1x6]
        # Graph convolutional layers.
        x = tf.expand_dims(x, 2)  # N x M x F=1
        for i in range(len(self.p)):
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('filter'):
                    x = self.filter(x, self.L[i], self.F[i], self.K[i])
                with tf.name_scope('bias_relu'):
                    x = self.brelu(x)
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.p[i])
        
        # Fully connected hidden layers.
        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N), int(M*F)])  # N x M
        for i,M in enumerate(self.M[:-1]): 
            with tf.variable_scope('fc{}'.format(i+1)):
                x = self.fc(x, M)
                x = tf.nn.dropout(x, dropout)

        # Embedding
        with tf.variable_scope('dense_layer'):
            # Add covariates
            if self.covar == 1:
                print('Appending covariates')
                print('Before:{}'.format(x.shape))
                x = tf.concat([x2,x],1)
                print('After:{}'.format(x.shape))
            self.embedding_size = x.shape[1]

        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            x = self.fc(x, self.M[-1], relu=False)
        return 

    def predict(self, data, labels=None, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = np.empty(size)
        embed = np.empty([size, self.embedding_size]) #embedding shape
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            
            batch_data = np.zeros((self.batch_size, 96))
            batch_data2 = np.zeros((self.batch_size, 8))
            tmp_data = data[begin:end,:96]
            tmp_data2 = data[begin:end,96:]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end-begin] = tmp_data
            batch_data2[:end-begin] = tmp_data2
            feed_dict = {self.ph_atrophy: batch_data, self.ph_data2: batch_data2, self.ph_dropout: 1}
            
            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros(self.batch_size)
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
                # Batch Embedding
                if self.covar == 1:
                    batch_embed = sess.run('dense_layer/concat:0', feed_dict) #with covariates
                else:
                    # batch_embed = sess.run('fc1/dropout/mul:0', feed_dict) #atrophy only
                    batch_embed=np.zeros((self.batch_size,self.embedding_size))
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)
            
            predictions[begin:end] = batch_pred[:end-begin]
            embed[begin:end] = batch_embed[:end-begin] #embedding

        # Embedding
        sess.run(self.op_assignment,{self.ph_embed:embed})

        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions

    def fit(self, train_data, train_labels, val_data, val_labels):
        t_process, t_wall = time.process_time(), time.time()
        sess = tf.Session(graph=self.graph)
        # shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        # writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)
        writer = tf.summary.FileWriter(os.path.join(self._get_path('checkpoints'),'summary'), self.graph)

        # Training.
        accuracies = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        for step in range(1, num_steps+1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]
            batch_data, batch_data2, batch_labels = train_data[idx,:96], train_data[idx,96:], train_labels[idx]  #Added data for FC layer
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices

            feed_dict = {self.ph_atrophy: batch_data, self.ph_data2: batch_data2, self.ph_labels: batch_labels, self.ph_dropout: self.dropout} #MODIFY
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)

            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                string, accuracy, f1, loss = self.evaluate(val_data, val_labels, sess)
                accuracies.append(accuracy)
                losses.append(loss)

                print('  validation {}'.format(string))
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall))
           
                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validation/accuracy', simple_value=accuracy)     #Scalars
                summary.value.add(tag='validation/f1', simple_value=f1)                 #Scalars
                summary.value.add(tag='validation/loss', simple_value=loss)             #Scalars
                writer.add_summary(summary, step)
                
                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)
                if (len(losses) > 15):
                    if ( round(losses[-1],4) > round(np.mean(losses[-10:]),4) ):
                        print('Current loss:', round(losses[-1],4), 'is greater than', round(losses[-2],4) )
                        break
                    elif ( round(losses[-1],2) == round(np.mean(losses[-5:]),2) ):
                        break


        print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))
        writer.close()
        sess.close()
        
        t_step = (time.time() - t_wall) / num_steps
        return accuracies, losses, t_step

    def fit_more(self, train_data, train_labels):
        t_process, t_wall = time.process_time(), time.time()
        sess = tf.Session(graph=self.graph)
        # shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        # writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)
        writer = tf.summary.FileWriter(os.path.join(self._get_path('checkpoints'),'summary'), self.graph)

        # Training.
        accuracies = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        for step in range(1, num_steps+1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]
            batch_data, batch_data2, batch_labels = train_data[idx,:96], train_data[idx,96:], train_labels[idx]  #Added data for FC layer
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices

            feed_dict = {self.ph_atrophy: batch_data, self.ph_data2: batch_data2, self.ph_labels: batch_labels, self.ph_dropout: self.dropout} #MODIFY
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)

            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                losses.append(loss_average)
                
                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                writer.add_summary(summary, step)
                
                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)
                if (len(losses) > 15):
                    if ( round(losses[-1],4) > round(np.mean(losses[-10:]),4) ):
                        print('Current loss:', round(losses[-1],4), 'is greater than', round(losses[-2],4) )
                        break
                    elif ( round(losses[-1],4) == round(np.mean(losses[-5:]),4) ):
                        print('Current loss:', round(losses[-1],4), 'equals', round(np.mean(losses[-5:]),4) )
                        break

        writer.close()
        sess.close()
        
        t_step = (time.time() - t_wall) / num_steps
        return accuracies, losses, t_step

# Model Type 3
class cgcnn3(base_model):
    """Graph CNN which uses the Chebyshev approximation.
    """
    def __init__(self, L, F, K, p, M, filter='chebyshev5', brelu='b2relu', pool='mpool1',
                num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                regularization=0, dropout=0, batch_size=100, eval_frequency=200,
                dir_name=''):
        super().__init__()   #ensures inheretence
        
        # Verify the consistency w.r.t. the number of layers.
        assert len(L) >= len(F) == len(K) == len(p)
        assert np.all(np.array(p) >= 1)
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.
        assert len(L) >= 1 + np.sum(p_log2)  # Enough coarsening levels for pool sizes.
        
        # Keep the useful Laplacians only. May be zero.
        M_0 = L[0].shape[0]

        j = 0
        self.L = []
        for pp in p:
            self.L.append(L[j])
            j += int(np.log2(pp)) if pp > 1 else 0
        L = self.L
        
        # Print information about NN architecture.
        Ngconv = len(p)
        Nfc = len(M)
        print('NN architecture')
        print('  input: M_0 = {}'.format(M_0))
        for i in range(Ngconv):
            print('dreage')
            print('  layer {0}: cgconv{0}'.format(i+1))
            print('    representation: M_{0} * F_{1} / p_{1} = {2} * {3} / {4} = {5}'.format(
                    i, i+1, L[i].shape[0], F[i], p[i], L[i].shape[0]*F[i]//p[i]))
            F_last = F[i-1] if i > 0 else 1
            print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                    i, i+1, F_last, F[i], K[i], F_last*F[i]*K[i]))
            if brelu == 'b1relu':
                print('    biases: F_{} = {}'.format(i+1, F[i]))
            elif brelu == 'b2relu':
                print('    biases: M_{0} * F_{0} = {1} * {2} = {3}'.format(
                        i+1, L[i].shape[0], F[i], L[i].shape[0]*F[i]))
        for i in range(Nfc):
            name = 'logits (softmax)' if i == Nfc-1 else 'fc{}'.format(i+1)
            print('  layer {}: {}'.format(Ngconv+i+1, name))
            print('    representation: M_{} = {}'.format(Ngconv+i+1, M[i]))
            M_last = M[i-1] if i > 0 else M_0 if Ngconv == 0 else L[-1].shape[0] * F[-1] // p[-1]
            print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                    Ngconv+i, Ngconv+i+1, M_last, M[i], M_last*M[i]))
            print('    biases: M_{} = {}'.format(Ngconv+i+1, M[i]))
        
        # Store attributes and bind operations.
        self.L, self.F, self.K, self.p, self.M = L, F, K, p, M
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)

        # Build the computational graph.
        self.build_graph(M_0)

    def build_graph(self, M_0):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder(tf.float32, (self.batch_size,M_0,10), 'data')
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')

            # Model.
            op_logits = self.inference(self.ph_data, self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                    self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(op_logits)

            # Embedding.
            with tf.name_scope('fc'):
                self.ph_embed = tf.placeholder(tf.float32, name='embed')
                self.embedding = tf.get_variable('embedding', initializer=np.zeros((1,1),dtype = np.float32), validate_shape=False)
                self.op_assignment = tf.assign(self.embedding, self.ph_embed, name='assignment', validate_shape=False)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()
            
            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)
        
        self.graph.finalize()

    def _inference(self, x, dropout): #[1x6]
        # Graph convolutional layers.
        # x = tf.expand_dims(x, 2)  # N x M x F=1
        for i in range(len(self.p)):
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('filter'):
                    x = self.filter(x, self.L[i], self.F[i], self.K[i])
                with tf.name_scope('bias_relu'):
                    x = self.brelu(x)
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.p[i])
        
        # Fully connected hidden layers.
        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N), int(M*F)])  # N x M
        for i,M in enumerate(self.M[:-1]): 
            with tf.variable_scope('fc{}'.format(i+1)):
                x = self.fc(x, M)
                x = tf.nn.dropout(x, dropout)

        # Embedding
        with tf.variable_scope('dense_layer'):
            self.embedding_size = x.shape[1]

        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            x = self.fc(x, self.M[-1], relu=False)
        return x

    def predict(self, data, labels=None, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = np.empty(size)
        embed = np.empty([size, self.embedding_size]) #embedding shape
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            
            batch_data = np.zeros((self.batch_size, 96, 10))
            tmp_data = data[begin:end,:,:]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end-begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1}
            
            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros(self.batch_size)
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
                # Batch Embedding
                # batch_embed = sess.run('fc1/dropout/mul:0', feed_dict)

            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)
            
            predictions[begin:end] = batch_pred[:end-begin]
            # embed[begin:end] = batch_embed[:end-begin] #embedding

        # Embedding
        # sess.run(self.op_assignment,{self.ph_embed:embed})

        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions

    def fit(self, train_data, train_labels, val_data, val_labels):
        t_process, t_wall = time.process_time(), time.time()
        sess = tf.Session(graph=self.graph)
        # shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        # writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)
        writer = tf.summary.FileWriter(os.path.join(self._get_path('checkpoints'),'summary'), self.graph)
        # tf.train.Saver().save(sess, os.path.join(self._get_path('checkpoints'), 'model-init'))

        # Training.
        accuracies = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        for step in range(1, num_steps+1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]
            batch_data, batch_labels = train_data[idx,:,:], train_labels[idx]
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices

            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_dropout: self.dropout} #MODIFY
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)

            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                string, accuracy, f1, loss = self.evaluate(val_data, val_labels, sess)
                accuracies.append(accuracy)
                losses.append(loss)

                print('  validation {}'.format(string))
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall))
           
                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validation/accuracy', simple_value=accuracy)     #Scalars
                summary.value.add(tag='validation/f1', simple_value=f1)                 #Scalars
                summary.value.add(tag='validation/loss', simple_value=loss)             #Scalars
                writer.add_summary(summary, step)
                
                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)
                if (len(losses) > 15):
                    if ( round(losses[-1],4) > round(np.mean(losses[-10:])*1.1, 4) ):
                        print('Current loss:', round(losses[-1],4), 'is greater than', round(np.mean(losses[-10:]),4) )
                        break
                    elif ( round(losses[-1],4) == round(np.mean(losses[-10:]),4) ):
                        print('Current loss:', round(losses[-1],2), 'equals', round(np.mean(losses[-5:]),2) )
                        break

        print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))
        writer.close()
        sess.close()
        
        t_step = (time.time() - t_wall) / num_steps
        return accuracies, losses, t_step

    def fit_more(self, train_data, train_labels, sess_path=None):
        t_process, t_wall = time.process_time(), time.time()
        sess = tf.Session(graph=self.graph)
        if sess_path:
            self.op_saver.restore(sess, sess_path)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)
        writer = tf.summary.FileWriter(os.path.join(self._get_path('checkpoints'),'summary'), self.graph)

        # Training.
        accuracies = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        for step in range(1, num_steps+1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]
            batch_data, batch_labels = train_data[idx,:,:], train_labels[idx]
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices

            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_dropout: self.dropout} #MODIFY
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)

            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                losses.append(loss_average)
                
                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                writer.add_summary(summary, step)
                
                # if loss_average <0.1:
                #     break
                if (len(losses) > 15):
                    if ( round(losses[-1],4) > round(np.mean(losses[-10:]),4) ):
                        print('Current loss:', round(losses[-1],4), 'is greater than', round(losses[-2],4) )
                        break
                    elif ( round(losses[-1],4) == round(np.mean(losses[-5:]),4) ):
                        print('Current loss:', round(losses[-1],4), 'equals', round(np.mean(losses[-5:]),4) )
                        break

                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

        writer.close()
        sess.close()
        
        t_step = (time.time() - t_wall) / num_steps
        return accuracies, losses, t_step