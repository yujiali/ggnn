-- Evaluate performance of trained model.
--
-- Yujia Li, 10/2015
--

require 'lfs'

babi_data = require 'babi_data'
seq_data = require 'seq_data'
eval_util = require 'eval_util'

ggnn = require '../ggnn'

cmd = torch.CmdLine()
cmd:option('-modeldir', '', 'Path to the root model output directory.')
cmd:option('-mb', 10, 'Size of the minibatch, this should not affect the final performance.')
cmd:option('-nthreads', 1, 'Number of threads to use.')
cmd:option('-datafile', '', 'Path to the data file to evaluate on')
opt = cmd:parse(arg)

d = torch.load(opt.modeldir .. '/params')
opt.nsteps = d.nsteps
opt.annotationdim = d.annotationdim
opt.mode = d.mode

print('')
print(opt)
print('')

torch.setnumthreads(opt.nthreads)

test_data = babi_data.load_graphs_from_file(opt.datafile)
uniform_length = babi_data.targets_are_uniform_length(test_data)
if uniform_length then
    print('Sequences are of the same length')
else
    print('Sequence lengths are not uniform')
end

test_data = babi_data.data_list_to_standard_data_seq(test_data, opt.annotationdim)
seq_data.add_end_node(test_data, false)

n_tasks = #test_data

total_err = 0
total_count = 0

for task_id=1,n_tasks do
    local timer = torch.Timer()
    if lfs.attributes(opt.modeldir .. '/' .. task_id .. '/model_best') ~= nil then
        model_file = opt.modeldir .. '/' .. task_id .. '/model_best'
    elseif lfs.attributes(opt.modeldir .. '/' .. task_id .. '/model_end') ~= nil then
        model_file = opt.modeldir .. '/' .. task_id .. '/model_end'
    else
        error('No model_best nor model_end available for task ' .. task_id)
    end
    print('Loading model from ' .. model_file)

    if opt.mode == 'seqnode' then
        net = ggnn.load_node_selection_seq_ggnn_from_file(model_file)
    elseif opt.mode == 'shareprop_seqnode' then
        net = ggnn.load_node_selection_seq_share_prop_ggnn_from_file(model_file)
    else
        error('Unknown mode ' .. opt.mode .. '.')
    end
    w, _ = net:getParameters()
    print('#parameters: ' .. w:nElement())
    if uniform_length then
        err_rate, err, total = eval_util.eval_seq_classification(net, test_data[task_id], opt.mb, opt.nsteps)
    else
        err_rate, err, total = eval_util.eval_seq_classification_per_example(net, test_data[task_id], opt.nsteps)
    end

    print(string.format('Task %d error rate: %d/%d=%.4f [%.2fs]', task_id, err, total, err_rate, timer:time().real))
    total_err = total_err + err
    total_count = total_count + total
end
print('======================')
print(string.format('Total error rate: %d/%d=%.4f', total_err, total_count, total_err / total_count))

