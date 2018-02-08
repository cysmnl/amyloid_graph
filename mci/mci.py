import numpy as np
import scipy.sparse, sys, os
sys.path.insert(0,'../..')
from lib import model, graph, coarsening, utils, parser
from lib.parser import parser

# Suppress Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

args = parser.parse_args()

d = 86  # Number of nodes/regions
f = 10  # Number of features

# load files
if args.eval:
    trainx = args.trainx
    trainy = args.trainy
    testx  = args.testx
    testy  = args.testy
else:
    trainx = 'train_mci_x.csv'
    trainy = 'train_mci_y.csv'
    testx  = 'test_mci_x.csv'
    testy  = 'test_mci_y.csv'

x_train = np.loadtxt(trainx,delimiter=',').astype(np.float32)
offset  = x_train.shape[1] - d  #['CDR', 'AGE', 'APOE', 'GEN', 'EDU', 'ADNI_EF', 'ADNI_MEM', 'ADAS', 'MMSCORE']
n_train = len(x_train)
y_train = np.loadtxt(trainy, delimiter=',').astype(np.uint8)
x_val   = np.loadtxt('val_mci_x.csv', delimiter=',').astype(np.float32)
y_val   = np.loadtxt('val_mci_y.csv', delimiter=',').astype(np.uint8)
x_test  = np.loadtxt(testx,delimiter=',').astype(np.float32)
y_test  = np.loadtxt(testy, delimiter=',').astype(np.uint8)

# process inputs
if args.model3:
	X_train = utils.expand(x_train, offset)
	X_val   = utils.expand(x_val, offset)
	X_test  = utils.expand(x_test, offset)
else:
	X_train  = x_train[:,offset:]
	X_train2 = x_train[:,:offset]
	X_val    = x_val[:,offset:]
	X_val2   = x_val[:,:offset]
	X_test   = x_val[:,offset:]
	X_test2  = x_val[:,:offset]

# Number of classes.
C = y_train.max() + 1
assert C == np.unique(y_train).size

# WEIGHTED ADJACENCY MATRIX
A = np.loadtxt('./graph/mci_connectome.csv', delimiter=',').astype(np.float32)
A = scipy.sparse.csr_matrix(A)
assert type(A) is scipy.sparse.csr.csr_matrix
assert A.shape == (d, d)

# COARSEN
# Layer 0: M_0 = |V| = 96 nodes (10 added),|E| = 1137 edges
# Layer 1: M_1 = |V| = 48 nodes (3 added),|E| = 466 edges
# Layer 2: M_2 = |V| = 24 nodes (0 added),|E| = 177 edges
# Layer 3: M_3 = |V| = 12 nodes (0 added),|E| = 55 edges
# Layer 4: M_4 = |V| = 6 nodes (0 added),|E| = 15 edges

# make and save graph coarsening
# graphs, perm = coarsening.coarsen(A, levels=4, self_connections=False)
# for i in range(len(graphs)):
# 	filename='mci_graph_csr_M_'+str(i)
# 	scipy.sparse.save_npz('./graph/'+filename, graphs[i])
# np.save('./graph/mci_graph_perm', perm)

# keep the graph consistent
graphs = [1,2,3,4]
for i in range(4): 
	filename = './graph/mci_graph_csr_M_'+ str(i) + '.npz'
	graphs[i] = scipy.sparse.load_npz(filename)
perm = np.load('./graph/mci_graph_perm.npy')

# coarsen and permute graph inputs
if args.model3:
	X_train = coarsening.perm_3data(X_train, perm)
	X_val   = coarsening.perm_3data(X_val, perm)
	X_test  = coarsening.perm_3data(X_test, perm)
else:
	X_train = coarsening.perm_data(X_train, perm)
	X_val   = coarsening.perm_data(X_val, perm)
	X_test  = coarsening.perm_data(X_test, perm)
	# append the covariates
	X_train = np.append(X_train, X_train2, axis=1)
	X_val   = np.append(X_val, X_val2, axis=1)
	X_test  = np.append(X_test, X_test2, axis=1)
	print('Training: {}, Validation: {}, Test: {}'.format(X_train.shape, X_val.shape, X_test.shape))

# LAPLACIAN
L = [graph.laplacian(A, normalized=True) for A in graphs]

# HYPER PARAMETERS
if args.eval:
	params = dict()
	params['dir_name']       = str(args.dir_name)
	params['num_epochs']     = int(args.epochs)
	params['batch_size']     = int(args.batch)
	params['eval_frequency'] = len(x_train)/2
	params['F']              = list(map(int, args.filters.strip('[').strip(']').split(','))) # Number of graph convolutional filters.
	params['K']              = list(map(int, args.poly_order.strip('[').strip(']').split(','))) # Polynomial orders.
	params['p']              = [4, 2]    # Pooling sizes.
	params['M']              = list(map(int, args.fc.strip('[').strip(']').split(','))) # Output dimensionality of fully connected layers.
	params['regularization'] = float(args.reg)
	params['dropout']        = int(args.dropout)
	params['learning_rate']  = float(args.learn_rate)
	params['decay_rate']     = float(args.decay_rate)
	params['momentum']       = float(args.momentum)
	params['decay_steps']    = n_train / params['batch_size']
else:	
	params = dict()
	params['dir_name']       = 'mci_atrophy_test0'
	params['num_epochs']     = 5000
	params['batch_size']     = 5
	params['eval_frequency'] = 200

	# Building blocks.
	params['filter']         = 'chebyshev5'
	params['brelu']          = 'b1relu'
	params['pool']           = 'apool1'

	# Architecture.
	params['F']              = [5, 5]  	# Number of graph convolutional filters.
	params['K']              = [4, 18]  # Polynomial orders.
	params['p']              = [4, 2]   # Pooling sizes.
	params['M']              = [112, C] # Output dimensionality of fully connected layers.

	# Optimization.
	params['regularization'] = 1e-1
	params['dropout']        = 1
	params['learning_rate']  = 1e-3
	params['decay_rate']     = 1
	params['momentum']       = 0
	params['decay_steps']    = n_train / params['batch_size']


# define the right model
if args.model3:
	model = model.cgcnn3(L, **params)
else:
	params['covar']		 = offset 	# number of covariates, use 0 if no covariates
	model = model.cgcnn(L, **params)

# TRAIN
if args.eval:
    model.fit_more(X_train, y_train)
else:
    model.fit(X_train, y_train, X_val, y_val)

# TEST
# res=model.evaluate2(X_test,y_test,'mci')
# print(res)