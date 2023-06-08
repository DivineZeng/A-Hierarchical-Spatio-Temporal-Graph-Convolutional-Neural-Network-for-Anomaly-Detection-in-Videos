import os
import time
import math
import torch
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from hstgcnn_data_test import *
from hstgcnn_model import hstgcnn
from scipy import interpolate
from sklearn import metrics
import multiprocessing as mult


def Parallel_process(loader_test_list, loader_index_list, data_dir, lock, index, seq_now_len, seq_pred_len):
	
	hstgcn_test = HstgcnnDataset(data_dir, seq_now_len=seq_now_len, seq_pred_len=seq_pred_len)
	loader_test = DataLoader(hstgcn_test, batch_size=1, shuffle =False, num_workers=12)
	lock.acquire()
	loader_test_list.append(loader_test)
	loader_index_list.append(index)
	lock.release()
	#print(index+1)


def MedianAverage(inputs,input,filter_mark = False):
	inputs_list = []
	for mid in range(len(inputs)):
		input_list = []
		for r in range(int(input/2)-input+1,input-int(input/2),1):
			if((mid+r)>(len(inputs)-1)):
				input_list.append(inputs[len(inputs)-1])
			elif((mid+r)<0):
				input_list.append(inputs[0])
			else:
				input_list.append(inputs[mid+r])
		inputs_list.append(input_list)
	inputs_list = np.asarray(inputs_list,dtype='float32')
	mean = []
	for tmp in inputs_list:
		tmp = np.sort(tmp)
		if(filter_mark):
			mean.append(tmp[1:input-1].mean())
		else:
			mean.append(tmp[int((input-1)/2)].mean())
	return mean

loss_func = torch.nn.MSELoss()
seq_now_len = 4
seq_pred_len = 1
model = hstgcnn(n_stgcnn =1,n_txpcnn=5,input_feat=2,output_feat=2,
                 	seq_len=seq_now_len,pred_seq_len=seq_pred_len,kernel_size=3).cuda()

lock= mult.Lock()
p_all = []
m = mult.Manager()
loader_test_list = m.list([])
loader_index_list = m.list([])
for i in range(21):
	p = mult.Process(target=Parallel_process, args=(loader_test_list, loader_index_list,\
									"./data/avenue_data/data/m"+str(1+i)+".txt",\
									 lock, i, seq_now_len, seq_pred_len))
	p_all.append(p)
	p.start()

for v in p_all:
	v.join()
loader_test_list = list(loader_test_list)
loader_index_list = list(loader_index_list)
filter_para = 41

model.load_state_dict(torch.load('./weight/avenue_weight.pth'))
model.eval()
all_auc=0
all_len=0
all_ab_mark = []
all_loc_ae = []

all_scores = np.array([], dtype=np.float32)
all_labels = np.array([], dtype=np.float32)

for sg, index in enumerate(loader_index_list):
	loader_test = loader_test_list[sg]
	ab_mark = np.load("./data/avenue_data/label/"+str(index+1)+".npy")
	if np.all(ab_mark == ab_mark[-1]):
		ab_mark[-1] = 1 - ab_mark[-1]

	loc_x = []
	loc_y = []
	loc_d = []
	loc_f = []
	for index_test, batch in enumerate(loader_test): 

		batch = [tensor.cuda() for tensor in batch]
		V, A, V_target, A_target, V_index, _ = batch
		V_index = int(V_index.squeeze().cpu().detach().numpy())

		V_tmp =V.permute(0,3,1,2)
		V_pred = model(V_tmp,A[0])#.squeeze())
		V_pred = V_pred.permute(0,2,3,1)
		V_pred = V_pred[0]#.squeeze()
		V_target = V_target[0]#.squeeze()

		V_pred = V_pred.reshape(V_pred.shape[0],int(V_pred.shape[1]/17),17,V_pred.shape[2])
		V_target = V_target.reshape(V_target.shape[0],int(V_target.shape[1]/17),17,V_target.shape[2])
		
		V_pred = V_pred.permute(1,0,2,3)
		V_target = V_target.permute(1,0,2,3)

		loss_data = []
		distance = []
		V_pred_graph = []
		V_target_graph = []
		for x in range(V_pred.shape[0]):
			
			x1_p = torch.min(V_pred[x][:,:,0])
			y1_p = torch.min(V_pred[x][:,:,1])
			x2_p = torch.max(V_pred[x][:,:,0])
			y2_p = torch.max(V_pred[x][:,:,1])

			xc1 = min(torch.min(V_pred[x][:,5:7,0]),torch.min(V_pred[x][:,11:13,0]))
			xc2 = max(torch.max(V_pred[x][:,5:7,0]),torch.max(V_pred[x][:,11:13,0]))

			yc1 = min(torch.min(V_pred[x][:,5:7,1]),torch.min(V_pred[x][:,11:13,1]))
			yc2 = max(torch.max(V_pred[x][:,5:7,1]),torch.max(V_pred[x][:,11:13,1]))

			x_p = float(xc2+xc1)/2
			y_p = float(yc2+yc1)/2
			V_pred_graph.append((x_p,y_p))

			x1_t = torch.min(V_target[x][:,:,0])
			y1_t = torch.min(V_target[x][:,:,1])
			x2_t = torch.max(V_target[x][:,:,0])
			y2_t = torch.max(V_target[x][:,:,1])
			
			xd1 = min(torch.min(V_target[x][:,5:7,0]),torch.min(V_target[x][:,11:13,0]))
			xd2 = max(torch.max(V_target[x][:,5:7,0]),torch.max(V_target[x][:,11:13,0]))

			yd1 = min(torch.min(V_target[x][:,5:7,1]),torch.min(V_target[x][:,11:13,1]))
			yd2 = max(torch.max(V_target[x][:,5:7,1]),torch.max(V_target[x][:,11:13,1]))

			x_t = float(xd2+xd1)/2
			y_t = float(yd2+yd1)/2
			V_target_graph.append((x_t,y_t))

			V_pred[x][:,:,0] = 0.5*(V_pred[x][:,:,0] -x_p)*((y2_p-y1_p)**(0.6))
			V_pred[x][:,:,1] = (V_pred[x][:,:,1] -y_p)*((y2_p-y1_p)**(0.6))

			V_target[x][:,:,0] = 0.5*(V_target[x][:,:,0] -x_t)*((y2_t-y1_t)**(0.6))
			V_target[x][:,:,1] = (V_target[x][:,:,1] -y_t)*((y2_t-y1_t)**(0.6))

			#norm = ((y2_p-y1_p)**(-1))*((y2_t-y1_t)**(-1))
			norm = max((y2_p-y1_p),(y2_t-y1_t))**(0.4)
			distance.append(float(((x_p-x_t)**2+(y_p-y_t)**2)*norm))
			loss_data.append(float(loss_func(V_target[x],V_pred[x])))

		V_pred_graph = np.asarray(V_pred_graph)
		V_target_graph = np.asarray(V_target_graph)

		if(V_pred_graph.shape[0]>1):
			pred_all = []
			target_all = []
			diff_list = []
			assert V_pred_graph.shape[0] == V_target_graph.shape[0]
			for m in range(V_pred_graph.shape[0]-1):
				l_pred = list(range(m+1,V_pred_graph.shape[0],1))
				for z in l_pred:
					pred_all.append(((V_pred_graph[m][0]-V_pred_graph[z][0])**2+\
								(V_pred_graph[m][1]-V_pred_graph[z][1])**2)**(0.8))
			
			for n in range(V_target_graph.shape[0]-1):
				l_target = list(range(n+1,V_target_graph.shape[0],1))
				for z in l_target:
					target_all.append(((V_target_graph[n][0]-V_target_graph[z][0])**2+\
								(V_target_graph[n][1]-V_target_graph[z][1])**2)**(0.8))
			assert len(pred_all) == len(target_all)
			for j,k in zip(target_all,pred_all):
				diff_list.append(abs(j-k))
			diff_all = (float(max(diff_list)))
		else:
			diff_all = 0.0
		loc_f.append(diff_all)
		loc_d.append(float(max(distance)))
		loss_data = np.asarray(loss_data)
		loc_x.append(V_index)
		if(len(loss_data)==0):
			loc_y.append(0)
		else:
			loc_y.append(float(sum(loss_data)/len(loss_data)))
	
	loc_y = MedianAverage(loc_y,filter_para)
	loc_d = MedianAverage(loc_d,filter_para)
	loc_f = MedianAverage(loc_f,filter_para)

	if(max(loc_y) > 0.0):
		#loc_y = loc_y-min(loc_y)
		loc_y = loc_y/max(loc_y)
	if(max(loc_d) > 0.0):
		#loc_d = loc_d-min(loc_d)
		loc_d = loc_d/max(loc_d)
	if(max(loc_f) > 0.0):
		#loc_f = loc_f-min(loc_f)
		loc_f = loc_f/max(loc_f)


	loc_a = []
	for h in range(len(loc_x)):
		#loc_a.append(loc_f[h])
		loc_a.append(loc_y[h]+loc_d[h]+loc_f[h])


	assert len(loc_a)==len(loc_x)

	x_new = np.arange(loc_x[0],loc_x[-1]+1)
	loc_a = np.asarray(loc_a,dtype='float32')
	loc_x = np.asarray(loc_x,dtype='float32')

	func = interpolate.interp1d(loc_x,loc_a,kind='slinear')

	loc_ae_be = func(x_new).tolist()
	x_new = x_new.tolist()

	loc_ae = [0]*x_new[0]
	loc_ae.extend(loc_ae_be)

	u_list = list(range(x_new[-1]+1))
	for u in range(ab_mark.shape[0]):
		if u not in u_list:
			loc_ae.append(0)

	ab_mark = ab_mark.tolist()
	assert len(loc_ae) == len(ab_mark)

	all_gt = np.array(ab_mark[5:],dtype='float32')
	all_dist = np.array(loc_ae[5:],dtype='float32')

	all_dist -= all_dist.min()
	all_dist /= all_dist.max()
	#all_dist = 1 - all_dist

	assert all_dist.shape==all_gt.shape
	all_scores = np.concatenate((all_scores[:], all_dist), axis=0)
	all_labels = np.concatenate((all_labels[:], all_gt), axis=0)
	

	scores = np.array([], dtype=np.float32)
	labels = np.array([], dtype=np.float32)

	scores = np.concatenate((scores, all_dist), axis=0)
	labels = np.concatenate((labels, all_gt), axis=0)

	fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
	auc = metrics.auc(fpr, tpr)

	all_auc += (all_gt.shape[0]-5)*auc
	all_len += (all_gt.shape[0]-5) 
	#print(str(auc))

all_fpr, all_tpr, all_thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
total_auc = metrics.auc(all_fpr, all_tpr)
print(total_auc, all_auc/all_len)