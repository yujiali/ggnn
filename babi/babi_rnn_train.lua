-- RNN/LSTM training for the bAbI dataset.
--
-- Yujia Li, 10/2015
--

require 'lfs'
require 'nn'
require 'optim'
require 'gnuplot'
color = require 'trepl.colorize'
require '../rnn'

babi_data = require 'babi_data'

cmd = torch.CmdLine()
cmd:option('-learnrate', 1e-3, 'learning rate')
cmd:option('-momentum', 0, 'momentum')
cmd:option('-mb', 10, 'minibatch size')
cmd:option('-nepochs', 0, 'number of epochs to train, if this is set, maxiters will not be used')
cmd:option('-maxiters', 1000, 'maximum number of weight updates')
cmd:option('-printafter', 10, 'print training information after this amount of weight updates')
cmd:option('-saveafter', 100, 'save checkpoint after this amount of weight updates')
cmd:option('-plotafter', 10, 'upldate training curves after this amount of weight updates')
cmd:option('-optim', 'adam', 'type of optimization algorithm to use')
cmd:option('-embedsize', 50, 'dimensionality of the embeddings')
cmd:option('-hidsize', 50, 'dimensionality of the hidden layers')
cmd:option('-ntargets', 1, 'number of targets for each example, if > 1 the targets will be treated as a sequence')
cmd:option('-nthreads', 1, 'set the number of threads to use with this process')
cmd:option('-ntrain', 0, 'number of training instances, 0 to use all available')
cmd:option('-nval', 50, 'number of validation instances, this will not be used if datafile.val exists')
cmd:option('-outputdir', '.', 'output directory')
cmd:option('-datafile', '', 'should contain lists of edges and questions in standard format')
cmd:option('-model', 'rnn', 'rnn or lstm')
cmd:option('-seed', 8, 'random seed')
-- cmd:option('-gpuid', -1, 'ID of the GPU to use, not using any GPUs if <= 0')
opt = cmd:parse(arg)

print('')
print(opt)
print('')


torch.setnumthreads(opt.nthreads)

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
optim_state = {}

os.execute('mkdir -p ' .. opt.outputdir)

print('')
print('checkpoints will be saved to [ ' .. opt.outputdir .. ' ]')
print('')
print(optim_config)
print('')

torch.save(opt.outputdir .. '/params', opt)

----------------------- prepare data ---------------------------

math.randomseed(opt.seed)
torch.manualSeed(opt.seed)

x_train, t_train = babi_data.load_rnn_data_from_file(opt.datafile, opt.ntargets)


-- uniform length if x_train is a tensor, otherwise x_train is a list
uniform_length = (type(x_train) == 'userdata')

if uniform_length then
    vocab_size = x_train:max()
    embed_size = opt.embedsize
    hid_size = opt.hidsize
    output_size = t_train:max()
else    -- sequences of different lengths
    vocab_size = babi_data.find_max_in_list_of_tensors(x_train)
    embed_size = opt.embedsize
    hid_size = opt.hidsize
    output_size = babi_data.find_max_in_list_of_tensors(t_train)
end

-- split training data into train & val
if lfs.attributes(opt.datafile .. '.val') == nil then
    print('Splitting part of training data for validation.')
    if uniform_length then
        x_train, t_train, x_val, t_val = babi_data.split_set_tensor(x_train, t_train, opt.ntrain, opt.nval, true)
    else
        x_train, t_train, x_val, t_val = babi_data.split_set_input_output(x_train, t_train, opt.ntrain, opt.nval, true)
    end
else
    if opt.ntrain ~= 0 then -- if ntrain is 0, automatically use all the training data available
        if uniform_length then
            x_train, t_train = babi_data.split_set_tensor(x_train, t_train, opt.ntrain, 0, true)
        else
            x_train, t_train = babi_data.split_set_input_output(x_train, t_train, opt.ntrain, 0, true)
        end
    end
    print('Loading validation data from ' .. opt.datafile .. '.val')
    x_val, t_val = babi_data.load_rnn_data_from_file(opt.datafile .. '.val', opt.ntargets)
end

if opt.ntargets > 1 then
    opt.ntargets = opt.ntargets + 1     -- add 1 to include the end symbol
end

print('')

if uniform_length then
    train_data_loader = babi_data.MiniBatchLoader(x_train, t_train, opt.mb, true)
    print('Training set  : ' .. x_train:size(1) .. 'x' .. x_train:size(2) .. ' sequences')
    print('Validation set: ' .. x_val:size(1) .. 'x' .. x_val:size(2) .. ' sequences')
    n_train = x_train:size(1)
else
    train_data_loader = babi_data.PairedDataLoader(x_train, t_train, true)
    print('Training set  : ' .. #x_train .. ' sequences')
    print('Validation set: ' .. #x_val .. ' sequences')
    n_train = #x_train
end

print('Number of output classes: ' .. output_size)
print('')

if opt.nepochs > 0 then
    opt.maxiters = opt.nepochs * math.ceil(n_train / opt.mb)
end
print('Total number of weight updates: ' .. opt.maxiters)
print('')

if opt.model == 'rnn' then
    net = rnn.RNN(vocab_size, embed_size, hid_size, output_size)
elseif opt.model == 'lstm' then
    net = rnn.LSTM(vocab_size, embed_size, hid_size, output_size)
else
    error('Unknown model type: ' .. opt.model)
end
c = nn.CrossEntropyCriterion()

net:print_model()
print(c)
print('')

params, grad_params = net:getParameters()

print('Total number of parameters: ' .. params:nElement())
print('')

function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    local x_batch, t_batch, y, loss
    if uniform_length then
        x_batch, t_batch = train_data_loader:next()
        if opt.ntargets > 1 then
            t_batch = torch.reshape(t_batch, t_batch:size(1) * opt.ntargets, t_batch:size(2) / opt.ntargets)
        end
        y = net:forward(x_batch, opt.ntargets)
        loss = c:forward(y, t_batch)
        net:backward(c:backward(y, t_batch))
    else
        loss = 0
        for i=1,opt.mb do
            x_batch, t_batch = train_data_loader:next()
            if opt.ntargets > 1 then
                t_batch = torch.reshape(t_batch, t_batch:size(1) * opt.ntargets, t_batch:size(2) / opt.ntargets)
            end
            y = net:forward(x_batch, opt.ntargets)
            loss = loss + c:forward(y, t_batch)
            net:backward(c:backward(y, t_batch))
        end
        loss = loss / opt.mb
        grad_params:mul(1 / opt.mb)
    end
    grad_params:clamp(-5, 5)

    return loss, grad_params
end

function eval_loss(model, x, t)
    if uniform_length then
        if opt.ntargets > 1 then
            t = torch.reshape(t, t:size(1) * opt.ntargets, t:size(2) / opt.ntargets)
        end
        return c:forward(model:forward(x, opt.ntargets), t)
    else
        local loss = 0
        for i=1,#x do
            local tt = t[i]
            if opt.ntargets > 1 then
                tt = torch.reshape(tt, tt:size(1) * opt.ntargets, tt:size(2) / opt.ntargets)
            end
            loss = loss + c:forward(model:forward(x[i], tt), tt)
        end
        return loss / #x
    end
end

function eval_err(model, x, t)
    if uniform_length then
        local pred = model:predict(x, opt.ntargets)
        pred = pred:typeAs(t)
        if opt.ntargets > 1 then
            return pred:ne(t):type('torch.DoubleTensor'):sum(2):gt(0):type('torch.DoubleTensor'):mean()
        else
            return pred:ne(t):type('torch.DoubleTensor'):mean()
        end
    else
        local err = 0
        for i=1,#x do
            local pred = model:predict(x[i], opt.ntargets)
            if opt.ntargets > 1 then
                err = err + pred:typeAs(t[i]):ne(t[i]):type('torch.DoubleTensor'):sum():gt(0):sum()
            else
                err = err + pred:typeAs(t[i]):ne(t[i]):sum()
            end
        end
        return err / #x
    end
end

train_records = {}
train_error_records = {}
val_records = {}

function train()
    local loss = 0
    local batch_loss = 0
    local iter = 0

    local best_loss = math.huge
    local best_params = params:clone()

    local plot_iter = 0

    while iter < opt.maxiters do
        local timer = torch.Timer()
        batch_loss = 0
        for iter_before_print=1,opt.printafter do
            _, loss = optfunc(feval, params, optim_config, optim_state)
            loss = loss[1]
            batch_loss = batch_loss + loss
        end
        iter = iter + opt.printafter 
        batch_loss = batch_loss / opt.printafter

        plot_iter = plot_iter + 1
        
        val_err = eval_err(net, x_val, t_val)
        io.write(string.format('iter %d, grad_scale=%.8f, train_loss=%.6f, val_error_rate=%.6f, time=%.2f',
                iter, torch.abs(grad_params):max(), batch_loss, val_err, timer:time().real))

        table.insert(train_records, {iter, batch_loss})
        table.insert(val_records, {iter, val_err})

        if val_err < best_loss then
            best_loss = val_err
            best_params:copy(params)
            rnn.save_rnn_model(opt.outputdir .. '/model_best', best_params, opt.model, vocab_size, embed_size, hid_size, output_size)
            print(color.green(' *'))
        else
            print('')
        end

        if iter % opt.saveafter == 0 then
            rnn.save_rnn_model(opt.outputdir .. '/model_' .. iter, params, opt.model, vocab_size, embed_size, hid_size, output_size)
        end

        if plot_iter % opt.plotafter == 0 then
            generate_plots()
            collectgarbage()
        end
    end

    generate_plots()

    rnn.save_rnn_model(opt.outputdir .. '/model_end', params, opt.model, vocab_size, embed_size, hid_size, output_size)
end

function plot_learning_curve(records, fname, ylabel, xlabel)
    xlabel = xlabel or '#iterations'
    local rec = torch.Tensor(records)
    gnuplot.pngfigure(opt.outputdir .. '/' .. fname .. '.png')
    gnuplot.plot(rec:select(2,1), rec:select(2,2))
    gnuplot.xlabel(xlabel)
    gnuplot.ylabel(ylabel)
    gnuplot.plotflush()
    collectgarbage()
end

function generate_plots()
    if not pcall(function () plot_learning_curve(train_records, 'train', 'training loss') end) then
        print('[Warning] Failed to update training curve plot. Error ignored.')
    end
    if not pcall(function () plot_learning_curve(val_records, 'val', 'validation error rate') end) then
        print('[Warning] Failed to update validation curve plot. Error ignored.')
    end
    -- plot_learning_curve(train_records, 'train', 'training loss')
    -- plot_learning_curve(val_records, 'val', 'validation error rate')
    if eval_train_err then
        if not pcall(function () plot_learning_curve(train_error_records, 'train-err', 'training error rate') end) then
            print('[Warning] Failed to update training error curve plot. Error ignored.')
        end
        -- plot_learning_curve(train_error_records, 'train-err', 'training error rate')
    end
    collectgarbage()
end

train()

