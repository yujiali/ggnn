--
-- Train sequence GNN for the two graph sequence problems.
--
-- Yujia Li, 11/2015

require 'torch'
require 'optim'
require 'gnuplot'
color = require 'trepl.colorize'

babi_data = require 'babi_data'
seq_data = require 'seq_data'
eval_util = require 'eval_util'
ggnn = require '../ggnn'

cmd = torch.CmdLine()
cmd:option('-nsteps', 5, 'number of propagation iterations')
cmd:option('-learnrate', 1e-3, 'learning rate')
cmd:option('-momentum', 0, 'momentum')
cmd:option('-mb', 10, 'minibatch size')
cmd:option('-maxiters', 1000, 'maximum number of weight updates')
cmd:option('-printafter', 10, 'print training information after this amount of weight updates')
cmd:option('-saveafter', 100, 'save checkpoint after this amount of weight updates')
cmd:option('-optim', 'adam', 'type of optimization algorithm to use')
cmd:option('-statedim', 4, 'dimensionality of the node representations')
cmd:option('-nthreads', 1, 'set the number of threads to use with this process')
cmd:option('-ntrain', 100, 'number of training instances')
cmd:option('-nval', 100, 'number of validation instances')
cmd:option('-annotationdim', 1, 'dimensionality of the node annotations')
cmd:option('-outputdir', '.', 'output directory')
cmd:option('-mode', 'seqnode', 'should be {seqnode or shareprop_seqnode}')
cmd:option('-datafile', '', 'should contain lists of edges and questions in standard format')
cmd:option('-seed', 8, 'random seed')
opt = cmd:parse(arg)

print('')
print(opt)
print('')


torch.setnumthreads(opt.nthreads)

------------------------ parameters ----------------------------

state_dim = opt.statedim

prop_net_h_sizes = {}
output_net_h_sizes = {state_dim}
n_steps = opt.nsteps
eval_n_steps = n_steps
minibatch_size = opt.mb
max_iters = opt.maxiters
max_grad_scale = 5
print_after = opt.printafter
save_after = opt.saveafter
plot_after = print_after

optfunc = optim[opt.optim]
if optfunc == nil then
    error('Unknown optimization method: ' .. opt.optim)
end

optim_config = {
    learningRate=opt.learnrate, 
    weightDecay=0, 
    momentum=opt.momentum, 
    alpha=0.95,
    maxIter=1,
    maxEval=2,
    dampening=0
}

os.execute('mkdir -p ' .. opt.outputdir)

print('')
print('checkpoints will be saved to [ ' .. opt.outputdir .. ' ]')
print('')
print(optim_config)
print('')

torch.save(opt.outputdir .. '/params', opt)

------------------------ prepare data ---------------------------

math.randomseed(opt.seed)
torch.manualSeed(opt.seed)

all_data = babi_data.load_graphs_from_file(opt.datafile)

uniform_length = babi_data.targets_are_uniform_length(all_data)

n_edge_types = babi_data.find_max_edge_id(all_data)
n_tasks = babi_data.find_max_task_id(all_data)
if opt.nval > 0 then
    all_task_train_data, all_task_val_data = babi_data.split_set(all_data, {opt.ntrain, opt.nval}, true)
else
    all_task_train_data = babi_data.split_set(all_data, {opt.ntrain}, true)
    all_task_val_data = all_task_train_data
end

if opt.mode == 'seqnode' or opt.mode == 'shareprop_seqnode' then
    all_task_train_data = babi_data.data_list_to_standard_data_seq(all_task_train_data, opt.annotationdim)
    all_task_val_data = babi_data.data_list_to_standard_data_seq(all_task_val_data, opt.annotationdim)
else
    error('Unknown mode ' .. opt.mode)
end

seq_data.add_end_node(all_task_train_data, false)
seq_data.add_end_node(all_task_val_data, false)

print(tostring(n_tasks) .. ' tasks in total')
print('')

annotation_dim = opt.annotationdim
print(string.format('%d types of edges in total', n_edge_types))
print(string.format('%d-dimensional annotations for each node', annotation_dim))
print('')

----------------------------- loop over all tasks --------------------------------

-- outer for loop until then end of this file
for task_id=1,n_tasks do

print('')
print('')
print('')
print('=========================== Task ' .. task_id .. ' =================================')
print('')

train_data = all_task_train_data[task_id]
val_data = all_task_val_data[task_id]

print(tostring(#train_data) .. ' training examples')
print(tostring(#val_data) .. ' validation examples')
print('')

task_output_dir = opt.outputdir .. '/' .. task_id
os.execute('mkdir -p ' .. task_output_dir)


train_data_loader = babi_data.DataLoader(train_data, true)
val_data_loader = babi_data.DataLoader(val_data, false)

------------------------ set up network and training --------------------------

n_nodes = babi_data.find_max_target(train_data)
print('Max number of nodes: ' .. n_nodes)

if opt.mode == 'seqnode' then
    nsnet = ggnn.NodeSelectionGGNN(state_dim, annotation_dim, prop_net_h_sizes, output_net_h_sizes, n_edge_types)
    anet = ggnn.PerNodeGGNN(state_dim, annotation_dim, prop_net_h_sizes, output_net_h_sizes, n_edge_types)
    model = ggnn.NodeSelectionSequenceGGNN(nsnet, anet)
elseif opt.mode == 'shareprop_seqnode' then
    pnet = ggnn.BaseGGNN(state_dim, annotation_dim, prop_net_h_sizes, n_edge_types)
    nsnet = ggnn.NodeSelectionOutputNet(state_dim, annotation_dim, output_net_h_sizes)
    anet = ggnn.PerNodeOutputNet(state_dim, annotation_dim, output_net_h_sizes)
    model = ggnn.NodeSelectionSequenceSharePropagationGGNN(pnet, nsnet, anet)
else
    error('Unknown mode: ' .. opt.mode)
end

params, grad_params = model:getParameters()

criterion = nn.CrossEntropyCriterion()
if ggnn.use_gpu then
    criterion:cuda()
end

model:print_model()
print('number of parameters in the model: ' .. params:nElement())
print('')

optim_state = {}

train_records = {}
train_error_records = {}
val_records = {}

function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    local loss = 0

    local edges_list = {}
    local annotations_list = {}
    local target_list = {}

    for i=1,minibatch_size do
        local edges, annotation, target = train_data_loader:next()
        table.insert(edges_list, edges)
        table.insert(annotations_list, annotation)
        table.insert(target_list, target)
    end

    local loss = 0
    if uniform_length then
        -- this assumes all the targets are of the same size
        local targets = torch.Tensor(target_list)

        -- forward pass
        local output = model:forward(edges_list, targets:size(2), n_steps, annotations_list)
        
        local output_grad
        loss, output_grad = ggnn.compute_node_selection_seq_ggnn_loss_and_grad(criterion, output, targets, model.n_nodes_list, true)
        
        -- backward pass
        model:backward(output_grad)
    else
        for i=1,minibatch_size do
            edges = edges_list[i]
            annotation = annotations_list[i]
            targets = target_list[i]

            targets = torch.Tensor(targets)
            targets:resize(1,targets:nElement())

            local output = model:forward({edges}, targets:size(2), n_steps, {annotation})
            local i_loss, output_grad = ggnn.compute_node_selection_seq_ggnn_loss_and_grad(criterion, output, targets, model.n_nodes_list, true)

            loss = loss + i_loss
            model:backward(output_grad)
        end
    end
    loss = loss / minibatch_size
    grad_params:mul(1 / minibatch_size)

    grad_params:clamp(-max_grad_scale, max_grad_scale)
    return loss, grad_params
end

function train()
    local loss = 0
    local batch_loss = 0
    local iter = 0

    local best_val_err = math.huge
    local best_params = params:clone()

    while iter < max_iters do
        local timer = torch.Timer()
        batch_loss = 0
        for iter_before_print=1,print_after do
            _, loss = optfunc(feval, params, optim_config, optim_state)
            loss = loss[1]
            batch_loss = batch_loss + loss
        end
        iter = iter + print_after
        batch_loss = batch_loss / print_after
        if uniform_length then
            val_err = eval_util.eval_seq_classification(model, val_data, minibatch_size, n_steps)
        else
            val_err = eval_util.eval_seq_classification_per_example(model, val_data, n_steps)
        end
        io.write(string.format('iter %d, grad_scale=%.8f, train_loss=%.6f,%s val_error_rate=%.6f, time=%.2f',
                iter, torch.abs(grad_params):max(), batch_loss, 
                '', val_err, timer:time().real))

        table.insert(train_records, {iter, batch_loss})
        table.insert(val_records, {iter, val_err})

        if val_err < best_val_err then
            best_val_err = val_err
            best_params:copy(params)
            ggnn.save_model_to_file(task_output_dir .. '/model_best', model, best_params)
            print(color.green(' *'))
        else
            print('')
        end

        if iter % save_after == 0 then
            ggnn.save_model_to_file(task_output_dir .. '/model_' .. iter, model, params)
        end

        if iter % plot_after == 0 then
            generate_plots()
            collectgarbage()
        end
    end
    ggnn.save_model_to_file(task_output_dir .. '/model_end', model, params)
end

function plot_learning_curve(records, fname, ylabel, xlabel)
    xlabel = xlabel or '#iterations'
    local rec = torch.Tensor(records)
    gnuplot.pngfigure(task_output_dir .. '/' .. fname .. '.png')
    gnuplot.plot(rec:select(2,1), rec:select(2,2))
    gnuplot.xlabel(xlabel)
    gnuplot.ylabel(ylabel)
    gnuplot.plotflush()
    collectgarbage()
end

function generate_plots()
    plot_learning_curve(train_records, 'train', 'training loss')
    plot_learning_curve(val_records, 'val', 'validation error rate')
    collectgarbage()
end

train()

end -- end the loop over tasks


