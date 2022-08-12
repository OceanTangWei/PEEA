# %%

import warnings
import scipy.sparse as sp
warnings.filterwarnings('ignore')

import os
import keras
import numpy as np
import numba as nb
from utils import *
from tqdm import *
from evaluate import evaluate
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from layer import RAAttention, POSAttention
import random
import numpy as np
from scipy import optimize
from sklearn.metrics.pairwise import cosine_distances,manhattan_distances
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  
session = tf.Session(config=config)

seed = 12306
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# %%

def get_noisy_version(x,p=0.1):
    N = len(x)
    save_index = list(np.random.permutation(N))[:int(N*(1-p))]
    return x[save_index] 

train_pair, dev_pair, adj_matrix, r_index, r_val, adj_features, rel_features,rwr_features,rel_adj_matrix, tools,log_sparse_rel_matrix = load_data("data/D_W_15K_V2/",train_ratio=0.01)
print(len(train_pair))
adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)
rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data
rel_adj_matrix, rel_adj_val = np.stack(rel_adj_matrix.nonzero(), axis=1), sp.csr_matrix(rel_adj_matrix).data

item_size = len(rel_val)
# %%
node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
triple_size = len(adj_matrix)
node_hidden = 128
rel_hidden = 128
batch_size = 1024 #1024
dropout_rate = 0.3 # 0.3
lr = 0.005 #0.005
gamma = 15
depth = 2

anchor_num = train_pair.shape[0]
print(anchor_num)


# %%

def get_embedding(index_a, index_b, vec=None):
    if vec is None:
        inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix,rwr_features,rel_adj_matrix, rel_adj_val]
        inputs = [np.expand_dims(item, axis=0) for item in inputs]
        vec = get_emb.predict_on_batch(inputs)
    Lvec = np.array([vec[e] for e in index_a])
    Rvec = np.array([vec[e] for e in index_b])
    Lvec = Lvec / (np.linalg.norm(Lvec, axis=-1, keepdims=True) + 1e-5)
    Rvec = Rvec / (np.linalg.norm(Rvec, axis=-1, keepdims=True) + 1e-5)
    return Lvec, Rvec


def get_all_embedding():
 
    inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix,rwr_features,rel_adj_matrix, rel_adj_val]
    inputs = [np.expand_dims(item, axis=0) for item in inputs]
    vec = get_emb.predict_on_batch(inputs)
    vec = vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-5)
    return vec


class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return self.embeddings

    
    
def get_trgat(node_hidden, rel_hidden, anchor_num, item_size, triple_size=triple_size, node_size=node_size, rel_size=rel_size, dropout_rate=0,
              gamma=3, lr=0.005, depth=2):
    adj_input = Input(shape=(None, 2))
    index_input = Input(shape=(None, 2), dtype='int64')
    val_input = Input(shape=(None,))
    rel_adj = Input(shape=(None, 2))
    ent_adj = Input(shape=(None, 2))
    
    rwr_input = Input(shape=(None, node_hidden))
    
    rel_adj_matrix_input=Input(shape=(None, 2), dtype='int64')
    rel_adj_val = Input(shape=(None,))
    
    alignment_input = Input(shape=(None, 2))
    


    ent_emb = TokenEmbedding(node_size, node_hidden, trainable=True)(val_input)
    
    rel_emb = TokenEmbedding(rel_size, node_hidden, trainable=True)(val_input)
    

    

    def avg(tensor, size,soft=True):
        adj = K.cast(K.squeeze(tensor[0], axis=0), dtype="int64")
        adj = tf.SparseTensor(indices=adj, values=tf.ones_like(adj[:, 0], dtype='float32'),
                              dense_shape=(node_size, size))
        if soft:
          adj = tf.sparse_softmax(adj)
        return tf.sparse_tensor_dense_matmul(adj, tensor[1])
        
    def rwr_avg(tensor, size, soft=True):
        adj = K.cast(K.squeeze(tensor[0], axis=0), dtype="int64")
        vals = K.cast(K.squeeze(tensor[1], axis=0), dtype="float32")
        rwr_feature = K.cast(K.squeeze(tensor[2], axis=0), dtype="float32")
        adj = tf.SparseTensor(indices=adj, values=vals,
                              dense_shape=(node_size, size))

        adj = tf.sparse_softmax(adj)
        return tf.sparse_tensor_dense_matmul(adj, rwr_feature)

#      
    
    opt = [rel_emb, adj_input, index_input, val_input]
    
    
    
    ent_feature = Lambda(avg, arguments={'size': node_size})([ent_adj, ent_emb])
    rel_feature = Lambda(avg, arguments={'size': rel_size})([rel_adj, rel_emb])
 

   


    rwr_feature = Lambda(rwr_avg, arguments={'size': node_size})([rel_adj_matrix_input,rel_adj_val, rwr_input])

    rwr_feature = Lambda(lambda x:tf.nn.l2_normalize(x, 1))(rwr_feature)
    



    acti = "tanh"
    e_encoder = RAAttention(node_size, activation=acti,
                                  rel_size=rel_size,
                                  use_bias=True,
                                  depth=depth,
                                  triple_size=triple_size)
                                  

    r_encoder = RAAttention(node_size, activation=acti,
                                  rel_size=rel_size,
                                  use_bias=True,
                                  depth=depth,
                                  triple_size=triple_size)
                                  
    rwr_encoder = POSAttention(node_size, activation=acti,
                                  rel_size=rel_size,
                                  use_bias=True,
                                  depth=depth,
                                  triple_size=triple_size)                            

  
    

    R_feat =    rwr_encoder([rwr_feature]+opt)
    ent_feat = e_encoder([ent_feature] + opt)
    rel_feat = r_encoder([rel_feature] + opt)
    
    out_feature = Concatenate(-1)([ent_feat,rel_feat, R_feat] )
    

    out_feature = Lambda(lambda x:tf.nn.l2_normalize(x, 1))(out_feature) 
    out_feature = Dropout(dropout_rate)(out_feature)

    
    

    def align_loss(tensor):
        def squared_dist(x):
            A, B = x
            row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
            row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.
            row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
            row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.
            return row_norms_A + row_norms_B - 2 * tf.matmul(A, B, transpose_b=True)
             
         


        emb = tensor[1]
        l, r = K.cast(tensor[0][0, :, 0], 'int32'), K.cast(tensor[0][0, :, 1], 'int32')
        l_emb, r_emb = K.gather(reference=emb, indices=l), K.gather(reference=emb, indices=r)

        pos_dis = K.sum(K.square(l_emb - r_emb), axis=-1, keepdims=True)
        
        

        rwr_feature = tensor[-1]

        l_rwr, r_rwr = K.gather(reference=rwr_feature, indices=l), K.gather(reference=rwr_feature, indices=r)
        

        r_neg_dis_pos = tf.exp(-0.1*(squared_dist([r_rwr, rwr_feature])))
        l_neg_dis_pos = tf.exp(-0.1*(squared_dist([l_rwr, rwr_feature])))


        r_neg_dis = squared_dist([r_emb, emb])
        l_neg_dis = squared_dist([l_emb, emb])
        

   

        

        l_loss = pos_dis - l_neg_dis * tf.sqrt(l_neg_dis_pos)+ gamma 
        
        l_loss = l_loss * (
                    1 - K.one_hot(indices=l, num_classes=node_size) - K.one_hot(indices=r, num_classes=node_size))

        r_loss = pos_dis - r_neg_dis * tf.sqrt(r_neg_dis_pos)  + gamma
        r_loss = r_loss * (
                    1 - K.one_hot(indices=l, num_classes=node_size) - K.one_hot(indices=r, num_classes=node_size))

        r_loss = (r_loss - K.stop_gradient(K.mean(r_loss, axis=-1, keepdims=True))) / K.stop_gradient(
            K.std(r_loss, axis=-1, keepdims=True))
        l_loss = (l_loss - K.stop_gradient(K.mean(l_loss, axis=-1, keepdims=True))) / K.stop_gradient(
            K.std(l_loss, axis=-1, keepdims=True))

        lamb, tau = 30, 10

        l_loss1 = K.logsumexp(lamb * l_loss + tau, axis=-1) 
        r_loss1 = K.logsumexp(lamb * r_loss + tau, axis=-1)
        
        
       
        
        

  
        return K.mean(r_loss1+l_loss1) 

    loss = Lambda(align_loss)([alignment_input, out_feature, R_feat])
    



    inputs = [adj_input, index_input, val_input, rel_adj, ent_adj,rwr_input,rel_adj_matrix_input,rel_adj_val]
    
    train_model = keras.Model(inputs=inputs + [alignment_input], outputs=loss)
    train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=keras.optimizers.rmsprop(lr))

    feature_model = keras.Model(inputs=inputs, outputs=out_feature)

    return train_model, feature_model


# %%

model, get_emb = get_trgat(dropout_rate=dropout_rate,
                           node_size=node_size,
                           anchor_num=anchor_num,
                           item_size = item_size,
                           rel_size=rel_size,
                           depth=depth,
                           gamma=gamma,
                           node_hidden=node_hidden,
                           rel_hidden=rel_hidden,
                           lr=lr,
                           triple_size=triple_size,
                           )
                          
evaluater = evaluate(dev_pair)



# %%

rest_set_1 = [e1 for e1, e2 in dev_pair]
rest_set_2 = [e2 for e1, e2 in dev_pair]
np.random.shuffle(rest_set_1)
np.random.shuffle(rest_set_2)


G1,G2,node_dict1,node_dict2,ent_size = tools
epoch = 60

print(len(train_pair), len(dev_pair))
for turn in range(1):
    for i in trange(epoch):
 
    
 
        np.random.shuffle(train_pair)
        for pairs in [train_pair[i * batch_size:(i + 1) * batch_size] for i in
                      range(len(train_pair) // batch_size + 1)]:
            if len(pairs) == 0:
                continue
            
 
            inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, rwr_features, rel_adj_matrix, rel_adj_val, pairs]

            inputs = [np.expand_dims(item, axis=0) for item in inputs]
            model.train_on_batch(inputs, np.zeros((1, 1)))
        if i == epoch - 1:
            feature = get_all_embedding()
            feature = tf.cast(feature, tf.float32)            
            sims = cal_sims(dev_pair,feature) 
          
                       
            log_sparse_rel_matrix = tf.Session().run(log_sparse_rel_matrix)
 
            values = np.array([1.0 for i in range(len(log_sparse_rel_matrix[0]))])
            log_sparse_rel_matrix = tf.SparseTensor(indices=log_sparse_rel_matrix[0],values=values,dense_shape=log_sparse_rel_matrix[-1])

            log_sparse_rel_matrix = tf.cast(log_sparse_rel_matrix, tf.float32)

            
            sims = tf.Session().run(sims)

            
            sims = np.exp(sims*50)
            for k in range(10):
                sims = sims / np.sum(sims,axis=1,keepdims=True)
                sims = sims / np.sum(sims,axis=0,keepdims=True)
            test(sims,"sinkhorn")
        
            Lvec, Rvec = get_embedding(dev_pair[:, 0], dev_pair[:, 1])




