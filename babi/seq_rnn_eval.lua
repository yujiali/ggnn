-- Evaluate performance of thr RNN models.
--
-- Yujia Li, 10/2015
--

require 'lfs'
rnn = require '../rnn'
eval_util = require 'eval_util'
babi_data = require 'babi_data'
seq_data = require 'seq_data'

cmd = torch.CmdLine()
cmd:option('-modeldir', '', 'Path to the root model output directory.')
cmd:option('-mb', 1000, 'Size of the minibatch, this should not affect the final performance.')
cmd:option('-nthreads', 1, 'Number of threads to use.')
cmd:option('-datafile', '', 'Path to the data file to evaluate on')
opt = cmd:parse(arg)

print('')
print(opt)
print('')

torch.setnumthreads(opt.nthreads)

x_test, t_test = seq_data.load_rnn_data_from_file(opt.datafile)

if lfs.attributes(opt.modeldir .. '/model_best') ~= nil then
    net = rnn.load_rnn_model(opt.modeldir .. '/model_best')
elseif lfs.attributes(opt.modeldir .. '/model_end') ~= nil then
    net = rnn.load_rnn_model(opt.modeldir .. '/model_end')
else
    error('No model_best nor model_end available in ' .. opt.modeldir)
end
net:print_model()

w, _ = net:getParameters()
print('')
print('Number of parameters: ' .. w:nElement())
print('')

if type(x_test) == 'userdata' then
    err_rate, err, total = eval_util.eval_seq_rnn_classification(net, x_test, t_test, opt.mb)
else
    err_rate, err, total = eval_util.eval_seq_rnn_classification_per_example(net, x_test, t_test)
end

print(string.format('Total error rate: %d/%d=%.4f, acc %.4f', err, total, err_rate, 1 - err_rate))

