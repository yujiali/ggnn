"""
Run RNN and LSTM baselines for bAbI and the two extra sequence tasks.

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



def run_q4(model):
    """
    model is one of {rnn, lstm}
    """
    n_train_to_try = [50, 100, 250, 500, 950]
    for fold in xrange(N_FOLDS):
        for n_train in n_train_to_try:
            subprocess.call(('th babi_rnn_train.lua -learnrate 0.001 -momentum 0.9 -mb 20 -maxiters 1000 -printafter 100 -saveafter 2000 -ntargets 1 -ntrain %d -nval 50 -outputdir exp_%d/%s/q4/%d -datafile data/processed_%d/rnn/train/4_rnn.txt -model %s' % (n_train, fold+1, model, n_train, fold+1, model)).split(' '))

def eval_q4(model):
    n_train_to_try = [50, 100, 250, 500, 950]
    eval_task_multi_ntrain('th babi_rnn_eval.lua -modeldir exp_%d/' + model + '/q4/%d -datafile data/processed_%d/rnn/test/4_rnn.txt', n_train_to_try)

def run_q15(model):
    """
    model is one of {rnn, lstm}
    """
    maxiters = '20000' if model == 'rnn' else '5000'
    for fold in xrange(N_FOLDS):
        subprocess.call(('th babi_rnn_train.lua -learnrate 0.001 -momentum 0.9 -mb 100 -maxiters ' + maxiters + ' -printafter 100 -saveafter 2000 -ntargets 1 -outputdir exp_%d/%s/q15 -datafile data/processed_%d/rnn/train/15_rnn.txt -model %s' % (fold+1, model, fold+1, model)).split(' '))

def eval_q15(model):
    eval_task('th babi_rnn_eval.lua -modeldir exp_%d/' + model + '/q15 -datafile data/processed_%d/rnn/test/15_rnn.txt')

def run_q16(model):
    """
    model is one of {rnn, lstm}
    """
    maxiters = '20000' if model == 'rnn' else '5000'
    for fold in xrange(N_FOLDS):
        subprocess.call(('th babi_rnn_train.lua -learnrate 0.0005 -momentum 0.9 -mb 100 -maxiters ' + maxiters + ' -printafter 100 -saveafter 2000 -ntargets 1 -outputdir exp_%d/%s/q16 -datafile data/processed_%d/rnn/train/16_rnn.txt -model %s' % (fold+1, model, fold+1, model)).split(' '))

def eval_q16(model):
    eval_task('th babi_rnn_eval.lua -modeldir exp_%d/' + model + '/q16 -datafile data/processed_%d/rnn/test/16_rnn.txt')

def run_q18(model):
    """
    model is one of {rnn, lstm}
    """
    maxiters = '500' if model == 'rnn' else '500'
    for fold in xrange(N_FOLDS):
        subprocess.call(('th babi_rnn_train.lua -learnrate 0.0005 -momentum 0.9 -mb 100 -maxiters ' + maxiters + ' -printafter 10 -saveafter 1000 -ntargets 1 -outputdir exp_%d/%s/q18 -datafile data/processed_%d/rnn/train/18_rnn.txt -model %s' % (fold+1, model, fold+1, model)).split(' '))

def eval_q18(model):
    eval_task('th babi_rnn_eval.lua -modeldir exp_%d/' + model + '/q18 -datafile data/processed_%d/rnn/test/18_rnn.txt')

def run_q19(model):
    """
    model is one of {rnn, lstm}
    """
    maxiters = '10000' if model == 'rnn' else '5000'
    for fold in xrange(N_FOLDS):
        subprocess.call(('th babi_rnn_train.lua -learnrate 0.001 -momentum 0.9 -mb 20 -maxiters ' + maxiters + ' -printafter 100 -saveafter 2000 -ntargets 2 -outputdir exp_%d/%s/q19 -datafile data/processed_%d/rnn/train/19_rnn.txt -model %s' % (fold+1, model, fold+1, model)).split(' '))

def eval_q19(model):
    eval_task('th babi_rnn_eval.lua -modeldir exp_%d/' + model + '/q19 -datafile data/processed_%d/rnn/test/19_rnn.txt')

def run_seq4(model):
    maxiters = '10000' if model == 'rnn' else '5000'
    for fold in xrange(N_FOLDS):
        subprocess.call(('th seq_rnn_train.lua -learnrate 0.001 -momentum 0.9 -mb 10 -maxiters ' + maxiters + ' -printafter 100 -saveafter 5000 -nval 50 -outputdir exp_%d/%s/seq4 -datafile data/extra_seq_tasks/fold_%d/noisy_rnn/train/4_rnn.txt -model %s' % (fold+1, model, fold+1, model)).split(' '))

def eval_seq4(model):
    eval_task('th seq_rnn_eval.lua -modeldir exp_%d/' + model + '/seq4 -datafile data/extra_seq_tasks/fold_%d/noisy_rnn/test/4_rnn.txt')

def run_seq5(model):
    maxiters = '10000' if model == 'rnn' else '5000'
    for fold in xrange(N_FOLDS):
        subprocess.call(('th seq_rnn_train.lua -learnrate 0.001 -momentum 0.9 -mb 10 -maxiters ' + maxiters + ' -printafter 100 -saveafter 5000 -nval 50 -outputdir exp_%d/%s/seq5 -datafile data/extra_seq_tasks/fold_%d/noisy_rnn/train/5_rnn.txt -model %s' % (fold+1, model, fold+1, model)).split(' '))

def eval_seq5(model):
    eval_task('th seq_rnn_eval.lua -modeldir exp_%d/' + model + '/seq5 -datafile data/extra_seq_tasks/fold_%d/noisy_rnn/test/5_rnn.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Which task to run.', choices=['babi4','babi15','babi16','babi18','babi19','seq4','seq5'])
    parser.add_argument('model', help="RNN/LSTM", choices=['rnn', 'lstm'])
    args = parser.parse_args()

    if args.task == 'babi4':
        run_q4(args.model)
        eval_q4(args.model)
    elif args.task == 'babi15':
        run_q15(args.model)
        eval_q15(args.model)
    elif args.task == 'babi16':
        run_q16(args.model)
        eval_q16(args.model)
    elif args.task == 'babi18':
        run_q18(args.model)
        eval_q18(args.model)
    elif args.task == 'babi19':
        run_q19(args.model)
        eval_q19(args.model)
    elif args.task == 'seq4':
        run_seq4(args.model)
        eval_seq4(args.model)
    elif args.task == 'seq5':
        run_seq5(args.model)
        eval_seq5(args.model)

