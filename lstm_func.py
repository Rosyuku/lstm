#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 16:00:05 2017

@author: kazuyuki
"""

import pandas as pd
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import report, training, Chain, datasets, iterators, optimizers, serializers, cuda
from chainer.training import extensions
from chainer.datasets import tuple_dataset

class Model(Chain):
    
    def __init__(self, n_input, n_output, n_units):
        super(Model, self).__init__(
                l1 = L.Linear(n_input, n_units),
                l2 = L.LSTM(n_units, n_units),
                l3 = L.Linear(n_units, n_output),
                )
        
    def reset_state(self):
        self.l2.reset_state()
        
    def __call__(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        o = self.l3(h2)
        return o
    
class LossFuncL(Chain):
    
    def __init__(self, predictor):
        super(LossFuncL, self).__init__(predictor=predictor)
        
    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        report({'loss':loss}, self)
        return loss
    
class LSTM_Iterator(chainer.dataset.Iterator):
    
    def __init__(self, dataset, batch_size=10, seq_len=10, support_len=10, repeat=True, pred=1):
        self.seq_length = seq_len
        self.support_len = support_len
        self.dataset = dataset
        self.nsamples = dataset.shape[0]
        self.columns = dataset.shape[1]
        self.pred = pred
        self.batch_size = batch_size
        self.repeat = repeat
        
        self.epoch = 0
        self.iteration = 0
        self.loop = 0
        self.is_new_epoch = False
        
    def __next__(self):
        if self.loop == 0:
            self.iteration += 1
            if self.repeat == True:
                self.offsets = np.random.randint(0, self.nsamples-self.seq_length-self.pred-1, size=self.batch_size)
            else:
                self.offsets = np.arange(0, self.nsamples-self.seq_length-self.pred-1)
            
        x, t = self.get_data(self.loop)
#        print(self.iteration)
        self.epoch = int((self.iteration * self.batch_size) // self.nsamples)
        return x, t
    
    def get_data(self, i):
        x = self.dataset[self.offsets+i, :]
        t = self.dataset[self.offsets+i+self.pred, :]
        return x, t
    
    def serialze(self, serialzier):
        self.iteration = serialzier('iteration', self.iteration)
        self.epoch = serialzier('epoch', self.epoch)
        
    @property
    def epoch_detail(self):
        return self.epoch
    
class LSTM_updater(training.StandardUpdater):
    
    def __init__(self, train_iter, optimizer, device):
        super(LSTM_updater, self).__init__(train_iter, optimizer, device=device)
        self.seq_length = train_iter.seq_length
        self.support_len = train_iter.support_len
        
    def update_core(self):
        loss = 0
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        optimizer.target.predictor.reset_state()
        
        for i in range(self.seq_length):
            train_iter.loop = i
            x, t  = train_iter.__next__()
            
            if i == self.support_len:
                y = optimizer.target.predictor(x)
                
            if i <= self.support_len:
                loss += optimizer.target(x, t)
            else:
                loss += optimizer.target(y, t)
                y = optimizer.target.predictor(y)
                
        optimizer.target.zerograds()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
            
def getTrainer(train, n_units, gpu=-1, batch_size=10, seq_len=10, support_len=10, pred=1, out='result', snap=100, epoch=1000):
    
    if gpu != -1:
        train = cuda.to_gpu(train, device=gpu)
        
    train_iter = LSTM_Iterator(train, batch_size, seq_len, support_len, True, pred)
        
    model = LossFuncL(Model(train.shape[1], train.shape[1], n_units))
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    
    updater = LSTM_updater(train_iter, optimizer, gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)
        
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(['iteration', 'epoch', 'main/loss']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.snapshot(filename="trainer_{.updater.epoch:05d}"), trigger=(snap, 'epoch'))
    
    return trainer

def getSineData():
    
    N_data = 200
    N_Loop = 4
    t = np.linspace(0, 2*np.pi*N_Loop, num=N_data)
    
    X = 0.8*np.sin(2.0*t)
    Y = 0.8*np.cos(1.0*t)
    
    N_train = int(N_data*0.75)
#    N_test = int(N_data*0.25)
    
    DataSet = np.c_[X, Y].astype(np.float32)
    
    train, test = np.array(DataSet[:N_train]), np.array(DataSet[N_train:])
    
    return train, test

def valid(model, test_iter, total='Total' ,s=0):
    
    model.reset_state()
    res1 = pd.DataFrame(index=range(test_iter.seq_length), columns=range(test_iter.columns), data=pd.np.NaN)
    res2 = pd.DataFrame(index=range(test_iter.seq_length), columns=[total], data=pd.np.NaN)

    for i in range(test_iter.seq_length):
        
        test_iter.loop = i
        x, t = test_iter.next()
        
        if i <= s:
            y = model(x)
        else:
            y = model(y)
        
        res1.iloc[i, :] = ((y - t)**2).data.mean(axis=0)**0.5
        res2.iloc[i, 0] = ((y - t)**2).data.mean()**0.5     
        
    res = pd.concat([res1, res2], axis=1)
    
    res.index += 1
    
    return res

def pred(model, data, seq, s=0, diff=1):
    
    model.reset_state()
    res1 = pd.DataFrame(index=range(seq), columns=range(data.shape[1]), data=pd.np.NaN)
    res2 = pd.DataFrame(index=range(seq), columns=range(data.shape[1]), data=pd.np.NaN)

    for i in range(seq):
        
        if i <= s or i==0:
            x = data[[i]]
            
        x = model(x)

        res1.iloc[i] = x.data
        
        if data.shape[0] > i + diff:
            res2.iloc[i] = data[[i+diff]]
    
    return res1, res2
            
if __name__ == "__main__":
    
    sine = True
    gpu = -1
    
    if sine == True:
        
        n_units = 100
        train, test = getSineData()
        
    trainer = getTrainer(train, n_units, gpu=-1, batch_size=10, seq_len=10, support_len=10, pred=1, out='result_1', snap=10, epoch=100)
    
    chainer.config.train = True
    
    chainer.config.train = False
    
#        if gpu != -1:
#            train = cuda.to_gpu(train, device=gpu)
#            
#        train_iter = LSTM_test_Iterator(train, 10, 10, 10)
#        test_iter = LSTM_test_Iterator(test, 10, 10, 10, False)
#        
#    model = LossFuncL(Model(train.shape[1], train.shape[1], n_units))
#    optimizer = optimizers.Adam()
#    optimizer.setup(model)
#    
#    updater = LSTM_updater(train_iter, optimizer, gpu)
#    trainer = training.Trainer(updater, (1000, 'epoch'), out='result')
#    
##    eval_model = model.copy()
##    eval_rnn = eval_model.predictor
##    eval_rnn.train = False
##    trainer.extend(extensions.Evaluator(test_iter, eval_model, device=gpu, eval_hook=lambda _: eval_rnn.reset_state()))
#    
#    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
#    trainer.extend(extensions.PrintReport(['iteration', 'epoch', 'main/loss', 'validation/main/loss']))
#    trainer.extend(extensions.ProgressBar())
#    trainer.extend(extensions.snapshot(filename="trainer_{.updater.epoch:05d}"), trigger=(100, 'epoch'))
#
##    trainer.run()        
    
    
        
