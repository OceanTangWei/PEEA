import numpy as np
import scipy.sparse as sp
import scipy
import tensorflow as tf
import os
import multiprocessing
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances,euclidean_distances
from scipy.spatial import distance

import keras.backend as K

import numba as nb
import numpy as np
import os
import time
import multiprocessing
import tensorflow as tf
import math
import random


def cal_sims(test_pair,feature):
    feature_a = tf.gather(indices=test_pair[:,0],params=feature)
    feature_b = tf.gather(indices=test_pair[:,1],params=feature)
    return tf.matmul(feature_a,tf.transpose(feature_b,[1,0]))
    
def load_triples(file_path,reverse = True):
    @nb.njit
    def reverse_triples(triples):
        reversed_triples = np.zeros_like(triples)
        for i in range(len(triples)):
            reversed_triples[i,0] = triples[i,2]
            reversed_triples[i,2] = triples[i,0]
            if reverse:
                reversed_triples[i,1] = triples[i,1] + rel_size
            else:
                reversed_triples[i,1] = triples[i,1]
        return reversed_triples
    
    with open(file_path + "triples_1") as f:
        triples1 = f.readlines()
        
    with open(file_path + "triples_2") as f:
        triples2 = f.readlines()
        
    triples = np.array([line.replace("\n","").split("\t") for line in triples1 + triples2]).astype(np.int64)
    node_size = max([np.max(triples[:,0]),np.max(triples[:,2])]) + 1
    rel_size = np.max(triples[:,1]) + 1
    
    all_triples = np.concatenate([triples,reverse_triples(triples)],axis=0)
    all_triples = np.unique(all_triples,axis=0)
    
    return all_triples, node_size, rel_size*2 if reverse else rel_size

def load_aligned_pair(file_path,ratio = 0.3):
    if "sup_ent_ids" not in os.listdir(file_path):
        with open(file_path + "ref_ent_ids") as f:
            aligned = f.readlines()
    else:
        with open(file_path + "ref_ent_ids") as f:
            ref = f.readlines()
        with open(file_path + "sup_ent_ids") as f:
            sup = f.readlines()
        aligned = ref + sup
        
    aligned = np.array([line.replace("\n","").split("\t") for line in aligned]).astype(np.int64)
    np.random.shuffle(aligned)
    return aligned[:int(len(aligned) * ratio)], aligned[int(len(aligned) * ratio):]

def test(sims,mode = "sinkhorn",batch_size = 1024):
    
    if mode == "sinkhorn":
        results = []
        for epoch in range(len(sims) // batch_size + 1):
            sim = sims[epoch*batch_size:(epoch+1)*batch_size]
            rank = np.argsort(-sim,axis=-1)
            ans_rank = np.array([i for i in range(epoch * batch_size,min((epoch+1) * batch_size,len(sims)))])
            tmp=np.equal(rank.astype(ans_rank.dtype),  np.tile(np.expand_dims(ans_rank,axis=1),[1,len(sims)]))
            tmp = tf.cast(tmp,"int32")
            tmp = tf.where(tmp)
            tmp = tf.Session().run(tmp)
            results.append(tmp)
  
        results = np.concatenate(results,axis=0)
        
        @nb.jit(nopython = True)
        def cal(results):
            hits1,hits5,hits10,mrr = 0,0,0,0
            for x in results[:,1]:
                if x < 1:
                    hits1 += 1
                if x<5:
                    hits5 += 1
                if x < 10:
                    hits10 += 1
                mrr += 1/(x + 1)
            return hits1,hits5,hits10,mrr
        hits1,hits5,hits10,mrr = cal(results)
        print("hits@1 : %.2f%% hits@5 : %.2f%% hits@10 : %.2f%% MRR : %.2f%%" % (hits1/len(sims)*100,hits5/len(sims)*100, hits10/len(sims)*100,mrr/len(sims)*100))
    else:
        c = 0
        for i,j in enumerate(sims[1]):
            if i == j:
                c += 1
        print("hits@1 : %.2f%%"%(100 * c/len(sims[0])))

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

def load_triples(file_name):
    triples = []
    entity = set()
    rel = set([0])
    for line in open(file_name,'r'):
        head,r,tail = [int(item) for item in line.split()]
        entity.add(head); entity.add(tail); rel.add(r+1)
        triples.append((head,r+1,tail))
    return entity,rel,triples

def load_alignment_pair(file_name):
    alignment_pair = []
    c = 0
    for line in open(file_name,'r'):
        e1,e2 = line.split()
        alignment_pair.append((int(e1),int(e2)))
    return alignment_pair

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T


def rwr_scores(adj_matrix, anchors):
    iterations = 100
    p = 0.85
    # G = nx.from_scipy_sparse_matrix(adj_matrix)
    node_num = adj_matrix.shape[0]
    anchor_num = len(anchors)
    a = get_a_emb(adj_matrix, anchors)
    norm_adj = normalize_adj(adj_matrix)
    prev_R = np.ones((node_num, anchor_num)) / anchor_num

    for i in range(iterations):
        R = (1 - p) * (norm_adj @ prev_R) + p * a
      
        diff = np.linalg.norm(R - prev_R, ord=2)
        print(i, diff)
        if diff < 1e-6:
            break
        prev_R = R

    return R


def get_a_emb(adj_matrix, anchors):
    anchor_num = len(anchors)
    node_num = adj_matrix.shape[0]
    a = sp.lil_matrix((node_num, anchor_num))
    for i, (node1, node2) in enumerate(anchors):
        a[node1, i] = 1.0
        a[node2, i] = 1.0
    return a




def get_matrix(triples, entity, rel):
    ent_size = max(entity) + 1
    rel_size = (max(rel) + 1)
    print(ent_size, rel_size)
    adj_matrix = sp.lil_matrix((ent_size, ent_size))
    adj_features = sp.lil_matrix((ent_size, ent_size))
    radj = []
    rel_in = np.zeros((ent_size, rel_size))
    rel_out = np.zeros((ent_size, rel_size))


    for i in range(max(entity) + 1):
        adj_features[i, i] = 1

    for h, r, t in triples:
       
        adj_matrix[h, t] = 1;
        adj_matrix[t, h] = 1;
        adj_features[h, t] = 1;
        adj_features[t, h] = 1;
        radj.append([h, t, r]);
        radj.append([t, h, r + rel_size]);
        rel_out[h][r] += 1;
        rel_in[t][r] += 1


    count = -1
    s = set()
    d = {}
    r_index, r_val = [], []
    for h, t, r in sorted(radj, key=lambda x: x[0] * 10e10 + x[1] * 10e5):
        if ' '.join([str(h), str(t)]) in s:
            r_index.append([count, r])
            r_val.append(1)
            d[count] += 1
        else:
            count += 1
            d[count] = 1
            s.add(' '.join([str(h), str(t)]))
            r_index.append([count, r])
            r_val.append(1)
    for i in range(len(r_index)):
        r_val[i] /= d[r_index[i][0]]

    rel_features = np.concatenate([rel_in, rel_out], axis=1)
    adj_features = normalize_adj(adj_features)
    rel_features = normalize_adj(sp.lil_matrix(rel_features))
    rel_adj_matrix = get_rel_matrix(triples, entity,rel)
    return adj_matrix, rel_adj_matrix, r_index, r_val, adj_features, rel_features   
    
def laplacian_positional_encoding(G, pos_enc_dim, backend="numpy"):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    # Laplacian
    # G = nx.from_scipy_sparse_matrix(adj)
    L = nx.normalized_laplacian_matrix(G)
    if backend == "numpy":
        # Eigenvectors with numpy
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort()  # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
        pos_emb = EigVec[:, 1:pos_enc_dim + 1]
    else:

        # Eigenvectors with scipy
        EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
        EigVec = EigVec[:, EigVal.argsort()] # increasing order
        pos_emb = np.abs(EigVec[:,1:pos_enc_dim+1])

    return pos_emb


def svd_deepwalk_matrix(X, dim):
    u, s, v = sp.linalg.svds(X, dim)
    # return U \Sigma^{1/2}


    res  = sp.diags(np.sqrt(s)).dot(u.T).T

    return res

def rwr_positional_encoding(G1,G2, anchors,pos_enc_dim, fast_dp=False):


    '''
    Compute initial node embedding vectors by random walk with restart
    :param edge_list1: network G1
    :param edge_list2: network G2
    :param anchors: anchor nodes, e.g., [1,1; 2,2]
    :return: rwr vectors of two networks
    '''
    n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
    score1, score2 = [], []

    for i, (node1, node2) in enumerate(anchors):
        s1 = nx.pagerank_scipy(G1, personalization={node1: 1})
        s2 = nx.pagerank_scipy(G2, personalization={node2: 1})
        s1_list = [0] * n1
        s2_list = [0] * n2
        for k, v in s1.items():
            s1_list[k] = v
        for k, v in s2.items():
            s2_list[k] = v
        score1.append(s1_list)
        score2.append(s2_list)
        print(i)

    rwr_score1 = np.array(score1).T
    rwr_score2 = np.array(score2).T

    rwr_score = np.concatenate((rwr_score1,rwr_score2),axis=0)




      
    if fast_dp:
    
      sample_num = pos_enc_dim
      sample_nodes = random.sample(range(n1+n2), sample_num)
      sample_emb = rwr_score[sample_nodes]
      sim = cosine_similarity(rwr_score,sample_emb)
      sim  = np.exp(sim/0.5)
      W_pinv = np.linalg.pinv(sim[sample_nodes])
      U, X, V = np.linalg.svd(W_pinv)
      Wfac = np.dot(U, np.diag(np.sqrt(X)))
      final_rwr_score = np.dot(sim, Wfac)
  
      
    else:
      sim = cosine_similarity(rwr_score,rwr_score)
      sim  = np.exp(sim/0.5)
      final_rwr_score = svd_deepwalk_matrix(sim,pos_enc_dim)


    rwr_score1 = final_rwr_score[:n1,:]
    rwr_score2 = final_rwr_score[n1:,:]

    return rwr_score1, rwr_score2



def get_rel_matrix(triples, entity, rel):
    ent_size = max(entity) + 1
    rel_size = (max(rel) + 1)
    rel_dict = {}
    rel_adj_matrix = sp.lil_matrix((ent_size, ent_size))

    for h, r, t in triples:
        rel_dict[r] = rel_dict[r] + 1 if r in rel_dict.keys() else 1
        rel_dict[r + rel_size] = rel_dict[r + rel_size] + 1 if r + rel_size in rel_dict.keys() else 1
    rel_cnt = sum([rel_dict[i] for i in rel_dict.keys()])
    
    tmp = []
    for h, r, t in triples:
        tmp.append(rel_cnt/rel_dict[r])
        tmp.append(rel_cnt/rel_dict[r + rel_size])
    cnt=max(tmp)
    for h, r, t in triples:
        rel_adj_matrix[h, t] = rel_cnt/rel_dict[r]/cnt
        rel_adj_matrix[t, h] = rel_cnt/rel_dict[r + rel_size]/cnt 
        
    
    return rel_adj_matrix

def get_single_adj_matrix(triples,rel_size):
    G = nx.Graph()
    nodes = set()
    rel_dict = {}
    for h,r,t in triples:
        nodes.add(h)
        nodes.add(t)
        rel_dict[r] = rel_dict[r] + 1 if r in rel_dict.keys() else 1
        rel_dict[r + rel_size] = rel_dict[r + rel_size] + 1 if r + rel_size in rel_dict.keys() else 1
    rel_cnt = sum([rel_dict[i] for i in rel_dict.keys()])
    for h, r, t in triples:
        rel_dict[r] = rel_dict[r] / rel_cnt
        rel_dict[r+rel_size] = rel_dict[r + rel_size] / rel_cnt
    nodes = sorted(list(nodes))
  
    G.add_nodes_from([i for i in range(len(nodes))])

    
    node_dict = {j:i for i,j in enumerate(nodes)}
    for h, r, t in triples:

        G.add_edge(node_dict[h], node_dict[t], weight= 1)
        G.add_edge(node_dict[t], node_dict[h], weight=1)
    return G, node_dict



def node_to_degree(G_degree, SET):
    SET = list(SET)
    SET = sorted([G_degree[x] for x in SET])
    return SET

def cal_degree_dict(G_list, G, layer):
    G_degree = G.degree()
    degree_dict = {}
    degree_dict[0] = {}
    for node in G_list:
        degree_dict[0][node] = {node}
    for i in range(1, layer + 1):
        degree_dict[i] = {}
        for node in G_list:
            neighbor_set = []
            for neighbor in degree_dict[i - 1][node]:
                neighbor_set += nx.neighbors(G, neighbor)
            neighbor_set = set(neighbor_set)
            for j in range(i - 1, -1, -1):
                neighbor_set -= degree_dict[j][node]
            degree_dict[i][node] = neighbor_set
    for i in range(layer + 1):
        for node in G_list:
            if len(degree_dict[i][node]) == 0:
                degree_dict[i][node] = [0]
            else:
                degree_dict[i][node] = node_to_degree(G_degree, degree_dict[i][node])
    return degree_dict



def CenaExtractNodeFeature(g, layers):
    g_degree_dict = cal_degree_dict(list(g.nodes()), g, layers)
    g_nodes = [i for i in range(len(g))]
    N1 = len(g_nodes)
    feature_mat = []
    for layer in range(layers + 1):
        L_max = [np.log(np.max(g_degree_dict[layer][x]) + 1) for x in g_nodes]
        L_med = [np.log(np.median(g_degree_dict[layer][x]) + 1) for x in g_nodes]
        L_min = [np.log(np.min(g_degree_dict[layer][x]) + 1) for x in g_nodes]
        L_75 = [np.log(np.percentile(g_degree_dict[layer][x], 75) + 1) for x in g_nodes]
        L_25 = [np.log(np.percentile(g_degree_dict[layer][x], 25) + 1) for x in g_nodes]
        feature_mat.append(L_max)
        feature_mat.append(L_min)
        feature_mat.append(L_med)
        feature_mat.append(L_75)
        feature_mat.append(L_25)
    feature_mat = np.array(feature_mat).reshape((-1, N1))
    return feature_mat.transpose()



def compute_structural_feature_sim(g1,g2 ,layer):
    features1 = CenaExtractNodeFeature(g1, layer)
    features2 = CenaExtractNodeFeature(g2, layer)
    sim = manhattan_distances(features1,features2)
    sim = np.exp(-sim*5)
    return sim




def compute_rel_feature_sim(triples,rel_size,num_node1,num_node2,new_to_old_dict1, new_to_old_dict2):
    entity_rel_dict = {}
    for h,r,t in triples:
        if h in entity_rel_dict.keys():
            entity_rel_dict[h].add(r)
        else:
            entity_rel_dict[h] = {r}

        if t in entity_rel_dict.keys():
            entity_rel_dict[t].add(r + rel_size)
        else:
            entity_rel_dict[t] = {r + rel_size}
    rel_jaccard_sim = np.zeros((num_node1,num_node2))

    for node in range(num_node1):
        print(node)
        rel_jaccard_sim[node] = np.array([jaccard(entity_rel_dict[new_to_old_dict1[node]],
                                         entity_rel_dict[new_to_old_dict2[i]]) for i in range(num_node2)])
    return rel_jaccard_sim

def jaccard(set1, set2):
    intersection = len(list(set1.intersection(set2)))
    union = (len(set1) + len(set2)) - intersection
    return float(intersection) / union


def get_pseudo_anchor_links(sim, new_to_old_dict1,new_to_old_dict2,num=500):
    
    N1 = sim.shape[0]
    index = np.argmax(sim,axis=1)
    value =  np.max(sim,axis=1)
    pred_list = [(i, index[i], value[i]) for i in range(N1)]
    pred_list_sorted = sorted(pred_list,key=lambda x:x[2],reverse=True)
    anchor_links = pred_list_sorted[:num]
    anchor_links=[[new_to_old_dict1[item[0]],new_to_old_dict2[item[1]]] for item in anchor_links]

    return anchor_links
def evaluate_pseudo(train_pair, alignment):
    train_pair = {item[0]:item[1] for item in train_pair}
    alignment = {item[0]:item[1] for item in alignment}
    cnt = 0
    for key in train_pair.keys():
        
        if key in alignment.keys() and train_pair[key] == alignment[key]:
            cnt += 1
       
    acc = cnt/len(train_pair.keys()) 
    return acc
        
def get_log_sparse_rel_matrix(all_triples, node_size):
    dr = {}
    for x,r,y in all_triples:
      if r not in dr:
        dr[r] = 0
      dr[r] += 1
    sparse_rel_matrix = []
    for i in range(node_size):
        sparse_rel_matrix.append([i, i, np.log(len(all_triples) / node_size)]);
    for h, r, t in all_triples:
        sparse_rel_matrix.append([h, t, np.log(len(all_triples) / dr[r])])

    sparse_rel_matrix = np.array(sorted(sparse_rel_matrix, key=lambda x: x[0]))
    sparse_rel_matrix = tf.SparseTensor(indices=sparse_rel_matrix[:, :2], values=sparse_rel_matrix[:, 2],
                                        dense_shape=(node_size, node_size))
    sparse_rel_matrix = tf.cast(sparse_rel_matrix, tf.float32)
    return sparse_rel_matrix
    
    
def load_data(lang, train_ratio=0.3):
    entity1, rel1, triples1 = load_triples(lang + 'triples_1')
    entity2, rel2, triples2 = load_triples(lang + 'triples_2')
    
    alignment_pair = load_alignment_pair(lang + 'ref_ent_ids')
    np.random.shuffle(alignment_pair)
    train_pair, dev_pair = alignment_pair[0:int(len(alignment_pair) * train_ratio)], alignment_pair[int(
          len(alignment_pair) * train_ratio):]
  
        

    adj_matrix, rel_adj_matrix, r_index, r_val, adj_features, rel_features = get_matrix(triples1 + triples2,
                                                                                        entity1.union(entity2),
                                                                                        rel1.union(rel2))
    rel_size = max(rel1.union(rel2))+1
    ent_size = max(entity1.union(entity2)) + 1
    
    G1, node_dict1  = get_single_adj_matrix(triples1, rel_size)
    G2, node_dict2 = get_single_adj_matrix(triples2, rel_size)
      
  
    node_dict1_r = {node_dict1[key]:key for key in node_dict1.keys()} # new->old
    node_dict2_r = {node_dict2[key]: key for key in node_dict2.keys()}
    
    

    if os.path.exists(lang+"_rwr_feature_{}.npy".format(train_ratio)):
      rwr_feature = np.load(lang+"rwr_feature_{}.npy".format(train_ratio))   
       
    else:                                                                 

#        
      if os.path.exists(lang+"pos1_{}.npy".format(train_ratio)):
         pos1 = np.load(lang+"pos1_{}.npy".format(train_ratio))   
         pos2 = np.load(lang+"pos2_{}.npy".format(train_ratio))
#        pos1= laplacian_positional_encoding(G1, 128, backend="nump")
#        pos2 = laplacian_positional_encoding(G2, 128, backend="nump")
#         pos1 = netmf(nx.to_scipy_sparse_matrix(G1, nodelist=list(range(len(G1)))),128)
#         pos2 = netmf(nx.to_scipy_sparse_matrix(G2, nodelist=list(range(len(G2)))),128)
      else:

        pos1,pos2=rwr_positional_encoding(G1, G2, [(node_dict1[pair[0]], node_dict2[pair[1]]) for pair in train_pair],128, fast_dp=False)
        np.save(lang+"pos1_{}.npy".format(train_ratio),pos1)
        np.save(lang+"pos2_{}.npy".format(train_ratio),pos2)
#      pos1= laplacian_positional_encoding(G1, 128, backend="nump")
#      pos2 = laplacian_positional_encoding(G2, 128, backend="nump")
#      pos1 = netmf(nx.to_scipy_sparse_matrix(G1, nodelist=list(range(len(G1)))),128)
#      pos2 = netmf(nx.to_scipy_sparse_matrix(G2, nodelist=list(range(len(G2)))),128)
      
      rwr_feature = np.zeros((ent_size,128))
      
      nodes1 = [node_dict1_r[i] for i in range(len(node_dict1.keys()))] 
      nodes2 = [node_dict2_r[i] for i in range(len(node_dict2.keys()))] 
      
      anchor_nodes1 = [node_dict1[pair[0]] for pair in train_pair]
      anchor_nodes2 = [node_dict2[pair[1]] for pair in train_pair]

      pos1[anchor_nodes1] = pos2[anchor_nodes2]
      rwr_feature[nodes1] = pos1
      rwr_feature[nodes2] = pos2
      
      rel_adj_matrix = get_rel_matrix(triples1 + triples2, entity1.union(entity2), rel1.union(rel2))
      np.save(lang+"rwr_feature_{}.npy".format(train_ratio),rwr_feature)
      
      log_sparse_rel_matrix =  get_log_sparse_rel_matrix(triples1 + triples2, ent_size)

    
    
    return np.array(train_pair), np.array(dev_pair), adj_matrix, np.array(r_index), np.array(
        r_val), adj_features, rel_features,rwr_feature, rel_adj_matrix, [G1,G2,node_dict1,node_dict2,ent_size],log_sparse_rel_matrix 
