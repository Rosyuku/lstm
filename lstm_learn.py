#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 23:27:41 2017

@author: kazuyuki
"""

import lstm_func as lf
import glob

if __name__ == "__main__":
    
    sine = True
    gpu = -1
    out = 'result_1'
    
    if sine == True:
        
        n_units = 100
        train, test = lf.getSineData()
        
    trainer = lf.getTrainer(train, n_units, gpu=-1, batch_size=10, seq_len=10, support_len=10, pred=1, 
                            out=out, snap=10, epoch=100)
    
    lf.chainer.config.train = True
    
    models = glob.glob(out+"/trainer*")
    models.sort()

    if len(models) == 0:
        trainer.run()
    else:
        lf.serializers.load_npz(models[-1], trainer)
        trainer.updater.get_iterator('main').iteration = trainer.updater.iteration
        trainer.run()
    
    