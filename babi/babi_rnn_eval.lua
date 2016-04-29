-- Evaluate performance of thr RNN models.
--
-- Yujia Li, 10/2015
--

require 'lfs'
rnn = require '../rnn'

eval_util = require 'eval_util'
babi_data = require 'babi_data'

cmd = torch.CmdLine()
cmd:option('-modeldir', '', 'Directory for the model file to load, if this is provided, ntargets does not need to be provided.')
cmd:option('-mb', 1000, 'Size of the minibatch, this should not affect the final performance.')
cmd:option('-nthreads', 1, 'Number of threads to use.')
cmd:option('-datafile', '', 'Path to the data file to evaluate on')
cmd:option('-ntargets', 1, 'Number of outputs per example, output will be treated as a sequence when > 1')
opt = cmd:parse(arg)

if lfs.attributes(opt.modeldir .. '/' .. 'params') ~= nil then
    d = torch.load(opt.modeldir .. '/params')
    opt.ntargets = d.ntargets
end

print('')
print(opt)
print('')

torch.setnumthreads(opt.nthreads)

x_test, t_test = babi_data.load_rnn_data_from_file(opt.datafile, opt.ntargets)

if opt.ntargets > 1 then
    opt.ntargets = opt.ntargets + 1
end

if lfs.attributes(opt.modeldir .. '/model_best') ~= nil then
    net = rnn.load_rnn_model(opt.modeldir .. '/model_best')
elseif lfs.attributes(opt.modeldir .. '/model_end') ~= nil then
    net = rnn.load_rnn_model(opt.modeldir .. '/model_end')
else
    error('No model_best nor model_end found in ' .. opt.modeldir)
end

net:print_model()

print('')

if type(x_test) == 'userdata' then
    -- err_rate, err, total = eval_util.eval_standard_classification(net, x_test, t_test, opt.mb)
    err_rate, err, total = eval_util.eval_seq_rnn_classification(net, x_test, t_test, opt.mb, opt.ntargets)
else
    -- err_rate, err, total = eval_util.eval_standard_classification_per_example(net, x_test, t_test)
    err_rate, err, total = eval_util.eval_seq_rnn_classification_per_example(net, x_test, t_test, opt.ntargets)
end

print(string.format('Total error rate: %d/%d=%.4f, acc %.4f', err, total, err_rate, 1 - err_rate))

