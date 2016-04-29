"""
Run bAbI experiments and evaluate performance.

Yujia Li, 04/2016
"""

import argparse
import subprocess
import re
import numpy as np

N_FOLDS = 10

def eval_task(eval_cmd):
    """
    eval_cmd must contain two slots for fold ID to be filled in.
    """
    p_acc = re.compile('=([0-9\.]+)')
    acc = np.zeros(N_FOLDS, dtype=np.float)

    print ''
    print 'Starting evaluation.'
    print ''

    for fold in xrange(N_FOLDS):
        print 'Fold %d:' % (fold+1),
        output = subprocess.check_output((eval_cmd % (fold+1, fold+1)).split(' ')).split('\n')
        acc[fold] = 1 - float(p_acc.search(output[-2]).group(1))
        print 'Accuracy: %.4f' % acc[fold]

    print '=================================='
    print 'Overall Accuracy: %.4f (%.4f)' % (acc.mean(), acc.std())

def eval_task_multi_ntrain(eval_cmd, n_train_to_eval):
    """
    eval_cmd must contain fold_id, n_train, fold_id three slots to be filled in.
    n_train_to_eval: a list of n_train's to be evaluated
    """
    p_acc = re.compile('=([0-9\.]+)')

    for n_train in n_train_to_eval:
        acc = np.zeros(N_FOLDS, dtype=np.float)

        print ''
        print 'Starting evaluation for n_train=%d.' % n_train
        print ''

        for fold in xrange(N_FOLDS):
            print 'Fold %d:' % (fold+1),
            output = subprocess.check_output((eval_cmd % (fold+1, n_train, fold+1)).split(' ')).split('\n')
            acc[fold] = 1 - float(p_acc.search(output[-2]).group(1))
            print 'Accuracy: %.4f' % acc[fold]

        print '=================================='
        print 'Overall Accuracy: %.4f (%.4f)' % (acc.mean(), acc.std())

def run_q4():
    for fold in xrange(N_FOLDS):
        subprocess.call(('th babi_train.lua -learnrate 0.01 -maxiters 100 -saveafter 100 -ntrain 50 -nval 50 -outputdir exp_%d/q4 -mode selectnode -datafile data/processed_%d/train/4_graphs.txt' % (fold+1, fold+1)).split(' '))

def eval_q4():
    eval_task('th babi_eval.lua -modeldir exp_%d/q4 -datafile data/processed_%d/test/4_graphs.txt')

def run_q15():
    for fold in xrange(N_FOLDS):
        subprocess.call(('th babi_train.lua -learnrate 0.005 -momentum 0.9 -maxiters 300 -saveafter 100 -ntrain 50 -nval 50 -statedim 5 -annotationdim 1 -outputdir exp_%d/q15 -mode selectnode -datafile data/processed_%d/train/15_graphs.txt' % (fold+1, fold+1)).split(' '))

def eval_q15():
    eval_task('th babi_eval.lua -modeldir exp_%d/q15 -datafile data/processed_%d/test/15_graphs.txt')

def run_q16():
    for fold in xrange(N_FOLDS):
        subprocess.call(('th babi_train.lua -learnrate 0.01 -momentum 0.9 -maxiters 600 -saveafter 100 -ntrain 50 -nval 50 -statedim 6 -annotationdim 1 -outputdir exp_%d/q16 -mode selectnode -datafile data/processed_%d/train/16_graphs.txt' % (fold+1, fold+1)).split(' '))

def eval_q16():
    eval_task('th babi_eval.lua -modeldir exp_%d/q16 -datafile data/processed_%d/test/16_graphs.txt')

def run_q18():
    for fold in xrange(N_FOLDS):
        subprocess.call(('th babi_train.lua -learnrate 0.01 -maxiters 400 -saveafter 100 -ntrain 50 -nval 50 -statedim 3 -annotationdim 2 -outputdir exp_%d/q18 -mode classifygraph -datafile data/processed_%d/train/18_graphs.txt' % (fold+1, fold+1)).split(' '))

def eval_q18():
    eval_task('th babi_eval.lua -modeldir exp_%d/q18 -datafile data/processed_%d/test/18_graphs.txt')

def run_q19():
    n_train_to_try = [50, 100, 250]
    for fold in xrange(N_FOLDS):
        for n_train in n_train_to_try:
            subprocess.call(('th babi_train.lua -learnrate 0.005 -momentum 0.9 -maxiters 1000 -saveafter 1000 -ntrain %d -nval 50 -statedim 6 -annotationdim 3 -outputdir exp_%d/q19/%d -mode shareprop_seqclass -datafile data/processed_%d/train/19_graphs.txt' % (n_train, fold+1, n_train, fold+1)).split(' '))

def eval_q19():
    eval_task_multi_ntrain('th babi_eval.lua -modeldir exp_%d/q19/%d -datafile data/processed_%d/test/19_graphs.txt', [50, 100, 250])

def run_seq4():
    for fold in xrange(N_FOLDS):
        subprocess.call(('th seq_train.lua -learnrate 0.002 -momentum 0.9 -mb 10 -maxiters 700 -statedim 20 -ntrain 50 -nval 50 -annotationdim 10 -outputdir exp_%d/seq4 -mode shareprop_seqnode -datafile data/extra_seq_tasks/fold_%d/noisy_parsed/train/4_graphs.txt' % (fold+1, fold+1)).split(' '))

def eval_seq4():
    eval_task('th seq_eval.lua -modeldir exp_%d/seq4 -datafile data/extra_seq_tasks/fold_%d/noisy_parsed/test/4_graphs.txt')

def run_seq5():
    for fold in xrange(N_FOLDS):
        subprocess.call(('th seq_train.lua -learnrate 0.001 -momentum 0.9 -mb 10 -maxiters 300 -statedim 20 -ntrain 50 -nval 50 -annotationdim 10 -outputdir exp_%d/seq5 -mode shareprop_seqnode -datafile data/extra_seq_tasks/fold_%d/noisy_parsed/train/5_graphs.txt' % (fold+1, fold+1)).split(' '))

def eval_seq5():
    eval_task('th seq_eval.lua -modeldir exp_%d/seq5 -datafile data/extra_seq_tasks/fold_%d/noisy_parsed/test/5_graphs.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Which task to run.', choices=['babi4','babi15','babi16','babi18','babi19','seq4','seq5'])
    args = parser.parse_args()

    if args.task == 'babi4':
        run_q4()
        eval_q4()
    elif args.task == 'babi15':
        run_q15()
        eval_q15()
    elif args.task == 'babi16':
        run_q16()
        eval_q16()
    elif args.task == 'babi18':
        run_q18()
        eval_q18()
    elif args.task == 'babi19':
        run_q19()
        eval_q19()
    elif args.task == 'seq4':
        run_seq4()
        eval_seq4()
    elif args.task == 'seq5':
        run_seq5()
        eval_seq5()

