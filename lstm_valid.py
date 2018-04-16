#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 00:07:15 2017

@author: kazuyuki
"""

import lstm_func as lf
import glob
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    lf.chainer.config.train = False
    sine = True
    gpu = -1
    out = 'result_1'
    
    if sine == True:
        
        n_units = 100
        train, test = lf.getSineData()
        
    trainer = lf.getTrainer(train, n_units, gpu=-1, batch_size=10, seq_len=10, support_len=10, pred=1, 
                            out=out, snap=10, epoch=100)
    
    test_iter = lf.LSTM_Iterator(test, repeat=False, seq_len=20)
    
    trainers = glob.glob(out+"/trainer*")
    trainers.sort()

    lf.serializers.load_npz(trainers[-1], trainer)
    trainer.updater.get_iterator('main').iteration = trainer.updater.iteration
    
    model = trainer.updater.get_optimizer('main').target.predictor
    
    test1res = lf.valid(model, test_iter)
    plt.figure()
    test1res.plot()

    plt.figure()
    for t in trainers:
        lf.serializers.load_npz(t, trainer)
        trainer.updater.get_iterator('main').iteration = trainer.updater.iteration
        
        model = trainer.updater.get_optimizer('main').target.predictor
        
        test1res = lf.valid(model, test_iter, t[-5:])
        test1res.iloc[:, -1].plot()   
        plt.legend()