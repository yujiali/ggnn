-- Evaluate performance of trained model.
--
-- Yujia Li, 10/2015
--

require 'lfs'

babi_data = require 'babi_data'
eval_util = require 'eval_util'

ggnn = require '../ggnn'

cmd = torch.CmdLine()
cmd:option('-modeldir', '', 'Path to the root model output directory, e.g. exp/q4. If this directory contains the "params" file generated during training, nsteps, annotationdim and mode do not need to be set.')
cmd:option('-nsteps', 5, 'Number of propagation steps to run.')
cmd:option('-mb', 10, 'Size of the minibatch, this should not affect the final performance.')
cmd:option('-nthreads', 1, 'Number of threads to use.')
cmd:option('-annotationdim', 1, 'dimensionality of node annotations')
cmd:option('-datafile', '', 'Path to the data file to evaluate on, e.g. data/processed/test/4_graphs.txt')
cmd:option('-mode', 'selectnode', 'one of {selectnode, classifygraph, seqclass, shareprop_seqclass}')
opt = cmd:parse(arg)

if lfs.attributes(opt.modeldir .. '/' .. 'params') ~= nil then
    d = torch.load(opt.modeldir .. '/params')
    opt.nsteps = d.nsteps
    opt.annotationdim = d.annotationdim
    opt.mode = d.mode
end

print('')
print(opt)
print('')

torch.setnumthreads(opt.nthreads)

if opt.mode == 'seqclass' or opt.mode == 'shareprop_seqclass' then
    test_data = babi_data.load_graphs_from_file(opt.datafile)
    test_data = babi_data.data_list_to_standard_data_seq(test_data, opt.annotationdim)
else
    test_data = babi_data.prepare_standard_data(opt.datafile, opt.annotationdim)
end
n_tasks = #test_data

total_err = 0
total_count = 0

for task_id=1,n_tasks do
    local timer = torch.Timer()
    model_file = opt.modeldir .. '/' .. task_id .. '/model_end'
    if opt.mode == 'selectnode' then
        net = ggnn.load_node_selection_ggnn_from_file(model_file)
        w, _ = net:getParameters()
        print('#parameters: ' .. w:nElement())
        err_rate, err, total = eval_util.eval_node_selection(net, test_data[task_id], opt.mb, opt.nsteps, false)
    elseif opt.mode == 'classifygraph' then
        net = ggnn.load_graph_level_ggnn_from_file(model_file)
        w, _ = net:getParameters()
        print('#parameters: ' .. w:nElement())
        err_rate, err, total = eval_util.eval_graph_classification(net, test_data[task_id], opt.mb, opt.nsteps, false)
    elseif opt.mode == 'seqclass' or opt.mode == 'shareprop_seqclass' then
        if opt.mode == 'seqclass' then
            net = ggnn.load_graph_level_seq_ggnn_from_file(model_file)
        else
            net = ggnn.load_graph_level_seq_share_prop_ggnn_from_file(model_file)
        end
        w, _ = net:getParameters()
        print('#parameters: ' .. w:nElement())
        err_rate, err, total = eval_util.eval_seq_classification(net, test_data[task_id], opt.mb, opt.nsteps, false)
    end
    print(string.format('Task %d error rate: %d/%d=%.4f [%.2fs]', task_id, err, total, err_rate, timer:time().real))
    total_err = total_err + err
    total_count = total_count + total
end
print('======================')
print(string.format('Total error rate: %d/%d=%.4f', total_err, total_count, total_err / total_count))

