-- Evaluation utils for bAbI.
--
-- Yujia Li, 10/2015
--

local eval_util = {}

function eval_util.eval_node_selection(model, data_set, mb_size, n_steps, verbose)
    if verbose == nil then verbose = false end
    local err = 0
    local edges_list = {}
    local annotations_list = {}
    local target_list = {}

    for i,g in ipairs(data_set) do
        local edges, annotation, target = unpack(g)

        table.insert(edges_list, edges)
        table.insert(annotations_list, annotation)
        table.insert(target_list, target)

        if i % mb_size == 0 or i == #data_set then
            local pred = model:predict(edges_list, n_steps, annotations_list)
            local targets = torch.Tensor(target_list):type(pred:type())

            err = err + torch.ne(pred, targets):sum()

            edges_list = {}
            annotations_list = {}
            target_list = {}
        end
    end
    if verbose then
        print('')
        print(string.format('Error Rate: %.4f', err / #data_set))
        print('')
    end
    return err / #data_set, err, #data_set
end

function eval_util.eval_graph_classification(model, data_set, mb_size, n_steps, verbose)
    local edges_list = {}
    local annotations_list = {}
    local target_list = {}

    local err = 0
    local total = 0

    for i, g in ipairs(data_set) do
        local edges, annotation, target = unpack(g)

        table.insert(edges_list, edges)
        table.insert(annotations_list, annotation)
        table.insert(target_list, target)

        if i % mb_size == 0 or i == #data_set then
            local pred = model:predict(edges_list, n_steps, annotations_list)
            local targets = torch.Tensor(target_list):typeAs(pred)

            err = err + pred:ne(targets):sum()
            total = total + pred:nElement()

            edges_list = {}
            annotations_list = {}
            target_list = {}
        end
    end

    return err / total, err, total
end

function eval_util.eval_seq_classification(model, data_set, mb_size, n_steps, verbose)
    local edges_list = {}
    local annotations_list = {}
    local target_list = {}

    local err = 0
    local total = 0

    for i, g in ipairs(data_set) do
        local edges, annotation, target = unpack(g)

        table.insert(edges_list, edges)
        table.insert(annotations_list, annotation)
        table.insert(target_list, target)

        if i % mb_size == 0 or i == #data_set then
            local targets = torch.Tensor(target_list)
            local pred = model:predict(edges_list, targets:size(2), n_steps, annotations_list):typeAs(targets)

            -- compute the sequence level accuracy, have to get all outputs correct to get credit
            err = err + pred:ne(targets):type('torch.DoubleTensor'):sum(2):gt(0):sum()
            total = total + pred:size(1)

            edges_list = {}
            annotations_list = {}
            target_list = {}
        end
    end

    return err / total, err, total
end

function eval_util.eval_seq_classification_per_example(model, data_set, n_steps, verbose)
    local err = 0
    local total = #data_set
    for i, g in ipairs(data_set) do
        local edges, annotation, targets = unpack(g)
        targets = torch.Tensor(targets)
        targets:resize(1, targets:nElement())

        local pred = model:predict({edges}, targets:size(2), n_steps, {annotation}):typeAs(targets)

        if pred:ne(targets):any() then
            err = err + 1
        end
    end
    return err / total, err, total
end

-- model is a neural network like classifier
function eval_util.eval_standard_classification(model, x, t, mb_size)
    local total = x:size(1)
    local i_start = 0

    local err = 0
    while i_start < total do
        local curr_mb_size = mb_size
        if i_start + curr_mb_size > total then
            curr_mb_size = total - i_start
        end

        local x_batch = x:narrow(1,i_start+1, curr_mb_size)
        local t_batch = t:narrow(1,i_start+1, curr_mb_size)

        local pred = model:predict(x_batch):typeAs(t_batch)

        err = err + pred:ne(t_batch):sum()

        i_start = i_start + curr_mb_size
    end

    return err / total, err, total
end

function eval_util.eval_standard_classification_per_example(model, x, t)
    local err = 0
    local total = #x
    for i=1,total do
        local pred = model:predict(x[i]):typeAs(t[i])
        err = err + pred:ne(t[i]):sum()
    end
    return err / total, err, total
end

-- n_outputs is not used
function eval_util.eval_seq_rnn_classification(model, x, t, mb_size, n_outputs)
    -- n_outputs = n_outputs or 1
    local total = x:size(1)
    local i_start = 0

    local err = 0
    while i_start < total do
        local curr_mb_size = mb_size
        if i_start + curr_mb_size > total then
            curr_mb_size = total - i_start
        end

        local x_batch = x:narrow(1,i_start+1, curr_mb_size)
        local t_batch = t:narrow(1,i_start+1, curr_mb_size)

        local n_outputs = t_batch:size(2)
        local pred = model:predict(x_batch, n_outputs):typeAs(t_batch)

        if n_outputs > 1 then
            err = err + pred:ne(t_batch):sum(2):gt(0):sum()
        else
            err = err + pred:ne(t_batch):sum()
        end

        i_start = i_start + curr_mb_size
    end

    return err / total, err, total
end

-- n_outputs is not used
function eval_util.eval_seq_rnn_classification_per_example(model, x, t, n_outputs)
    -- n_outputs = n_outputs or 1
    local err = 0
    local total = #x
    for i=1,total do
        local n_outputs = t[i]:nElement()
        local pred = model:predict(x[i], n_outputs):typeAs(t[i])
        if n_outputs > 1 then
            err = err + pred:ne(t[i]):sum(2):gt(0):sum()
        else
            err = err + pred:ne(t[i]):sum()
        end
    end
    return err / total, err, total
end


return eval_util

