# Gated Graph Sequence Neural Networks
This is the code for our ICLR'16 paper:
* Yujia Li, Danial Tarlow, Marc Brockschmidt, Richard Zemel. [*Gated Graph Sequence Neural Networks*](http://arxiv.org/abs/1511.05493).
International Conference on Learning Representations, 2016.

Please cite the above paper if you use our code.

The code is released under the [MIT license](LICENSE).

### Testing

Run `th test.lua` to test all the modules in the ggnn and rnn libraries.

### Reproducing the experiment results

To run the bAbI experiments, and experiments on the two extra sequence tasks:

1. Go into `babi/data`, run `bash get_10_fold_data.sh` to get 10 folds of bAbI
   data for 5 tasks (4, 15, 16, 18, 19) and do some preprocessing.
2. Go into `babi/data/extra_seq_tasks`, run `bash generate_10_fold_data.sh` to
   get 10 folds of data for the two extra sequence tasks.
3. Go back to `babi/` and use `run_experiments.py` to run the GGNN/GGS-NN
   experiments, e.g. `python run_experiments.py babi18` runs GGNN on bAbI task
   18 for all 10 folds of data.
4. Use `run_rnn_baselines.py` to run RNN/LSTM baseline experiments, e.g.
   `python run_rnn_baselines.py babi18 lstm` runs LSTM on bAbI task 18 for all
   10 folds of data.

### Notes
* Make sure `"./?.lua"` and `"./?/init.lua"` are on your lua path. For example by 
  `export LUA_PATH="./?.lua;./?/init.lua;$LUA_PATH"`.
* The experiment results may differ slightly from what we reported in the paper, as the
  datasets are randomly generated and will be different from run to run.

