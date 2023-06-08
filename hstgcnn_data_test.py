import os
import math
import sys
import time

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
import networkx as nx
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
from tqdm import tqdm

def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    else:
        return 1/(NORM)
                
def seq_to_graph(seq_,norm_lap_matr = True):
    #seq_ = seq_.squeeze()
    '''print(seq_[:,:,0])
    print(seq_rel[:,:,0])
    print("1111111111111111111111111")'''
    #seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[0]
    max_nodes = seq_.shape[1]

    #print(seq_.shape)
    
    V = np.zeros((seq_len,max_nodes,2))
    A = np.zeros((seq_len,max_nodes,max_nodes))
    for s in range(seq_len):
        step_ = seq_[s,:,:]
        #step_rel = seq_rel[:,:,s]
        for h in range(len(step_)):
            #print(step_rel[h]) 
            V[s,h,:] = step_[h]
            A[s,h,h] = 1
            for k in range(h+1,len(step_)):
                l2_norm = anorm(step_[h],step_[k])
                A[s,h,k] = l2_norm
                A[s,k,h] = l2_norm
        if norm_lap_matr: 
            G = nx.from_numpy_matrix(A[s,:,:])
            A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()

    #print(V.shape, A.shape)
            
    return V, A


class HstgcnnDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, seq_now_len=8, seq_pred_len=8, norm_lap_matr = True):

        super(HstgcnnDataset, self).__init__()

        self.data_dir = data_dir
        self.seq_now_len = seq_now_len
        self.seq_pred_len = seq_pred_len
        self.seq_len = self.seq_pred_len+self.seq_now_len
        self.norm_lap_matr = norm_lap_matr
        self.V_out = [] 
        self.A_out = []
        self.V_target = [] 
        self.A_target = []
        self.index = []
        self.track = []

        '''all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]'''

        f=open(self.data_dir,"r")
        data=f.readline()
        data=eval(data)
        track_num = list(data.keys())
        #track_num.sort(reverse=True)

        track_list = []
        track_list5 = [] 
        
        max_list = []
        for x in track_num: 
            track_list.append(list(map(int,data[str(x)].keys())))
            max_list.append(list(map(int,data[str(x)].keys()))[-1])
        
        
        max_number = max(max_list)
        #pbar = tqdm(total=max_number-(self.seq_len-2))
        for start in range(1,max_number-(self.seq_len-2),1):
            #pbar.update(1)
            t0 = time.time()
            track_list5 = []
            graph = []
            for x in range(len(track_num)):
                graph.append([])
                mark = True
                for y in range(start,start+self.seq_len,1):
                    #print(y)
                    if(y not in track_list[x]):
                        mark = False
                if(mark):
                    track_list5.append(list(range(start,start+self.seq_len,1)))
                else:
                    track_list5.append([])
            track_id = []
            for x in range(len(track_num),0,-1):
                
                if(track_list5[x-1] != []):
                    #print(track_list5[x-1])
                    
                    for y in track_list5[x-1]:
                        keypoints = data[str(track_num[x-1])][str(y)]
                        graph[x-1].append(keypoints)
                    track_id.append(int(track_num[x-1]))
                else:
                    del graph[x-1]
            if(graph == []):
                continue
            que = np.array(graph)
            que = np.transpose(que,(1,0,2,3))
            #que = np.transpose(que,(3,1,2,0))
            que = que.reshape(que.shape[0],que.shape[1]*que.shape[2],que.shape[3])
            #print(que.shape)
            '''que = que.reshape(que.shape[0],int(que.shape[1]/17),17,que.shape[2])
            print(que.shape)'''
            t1 = time.time()
            V, A = seq_to_graph(que, self.norm_lap_matr)
            #print(V.shape, A.shape)
            t2 = time.time()-t1
            self.V_out.append(V[0:self.seq_now_len])
            self.A_out.append(A[0:self.seq_now_len])
            self.V_target.append(V[self.seq_now_len:self.seq_len])
            self.A_target.append(A[self.seq_now_len:self.seq_len])
            self.index.append(np.asarray([start+4]))
            self.track.append(np.asarray(track_id))
        f.close()
        #pbar.close()


    def __len__(self):
        self.num_seq = len(self.V_out)
        assert len(self.V_out)==len(self.A_out)
        assert len(self.V_target)==len(self.A_target)
        return self.num_seq

    def __getitem__(self, index):
        out = [torch.from_numpy(self.V_out[index]).type(torch.FloatTensor), torch.from_numpy(self.A_out[index]).type(torch.FloatTensor), \
            torch.from_numpy(self.V_target[index]).type(torch.FloatTensor), torch.from_numpy(self.A_target[index]).type(torch.FloatTensor),\
            torch.from_numpy(self.index[index]).type(torch.FloatTensor), torch.from_numpy(self.track[index]).type(torch.IntTensor)]
        return out
