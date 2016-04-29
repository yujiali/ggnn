-- Data Loading for bAbI tasks.
--
-- Yujia Li, 10/2015
-- 

require 'nn'

babi_data = {}

function babi_data.load_graphs_from_file(filename)
    local data_list = {}
    local edge_list = {}
    local target_list = {}
    for line in io.lines(filename) do
        if string.len(line) == 0 then
            table.insert(data_list, {edge_list, target_list})
            edge_list = {}
            target_list = {}
        else
            local digits = {}
            if line:sub(1,1) == '?' then
                for d in line:gmatch('%d+') do
                    table.insert(digits, tonumber(d))
                end
                table.insert(target_list, digits)
            else
                for d in line:gmatch('%d+') do
                    table.insert(digits, tonumber(d))
                end
                table.insert(edge_list, digits)
            end
        end
    end
    return data_list
end

function babi_data.find_max_edge_id(data_list)
    local max_edge_id = 0
    for _, v in pairs(data_list) do
        local edges = v[1]
        for _, e in pairs(edges) do
            if e[2] > max_edge_id then
                max_edge_id = e[2]
            end
        end
    end
    return max_edge_id
end

function babi_data.find_max_node_id(data_list)
    local max_node_id = 0
    for _, v in pairs(data_list) do
        local edges = v[1]
        for _, e in pairs(edges) do
            if e[1] > max_node_id then
                max_node_id = e[1]
            end
            if e[3] > max_node_id then
                max_node_id = e[3]
            end
        end
    end
    return max_node_id
end

function babi_data.find_max_task_id(data_list)
    local max_task_id = 0
    for _, v in pairs(data_list) do
        local tasks = v[2]
        for _, t in pairs(tasks) do
            if t[1] > max_task_id then
                max_task_id = t[1]
            end
        end
    end
    return max_task_id
end

function babi_data.find_max_label(data_list, task_type)
    local max_label = 0
    for _, v in pairs(data_list) do
        local tasks = v[2]
        for _, t in pairs(tasks) do
            if (task_type == nil or t[1] == task_type) and t[#t] > max_label then
                max_label = t[#t]
            end
        end
    end
    return max_label
end

-- Find the maximum label for a task that contains sequences of labels, not a
-- single label.
function babi_data.find_max_label_seq(data_list, task_type)
    local max_label = 0
    for _, v in pairs(data_list) do
        local tasks = v[2]
        for _, t in pairs(tasks) do
            if task_type == nil or t[1] == task_type then
                for i=4,#t do   -- t[2] t[3] are src, tgt
                    if t[i] > max_label then
                        max_label = t[i]
                    end
                end
            end
        end
    end
    return max_label
end

-- data_list is a list of {edges, targets} tuples.
--
-- Return standard data, lists of {edges, annotations, target}, one list per 
-- task.
function babi_data.data_list_to_standard_data(data_list, n_annotation_dim)
    local n_nodes = babi_data.find_max_node_id(data_list)
    local n_tasks = babi_data.find_max_task_id(data_list)

    local task_data_list = {}
    for i=1,n_tasks do
        task_data_list[i] = {}
    end

    for _, v in pairs(data_list) do
        local edge_list = v[1]
        local target_list = v[2]
        for _, target in pairs(target_list) do
            local task_type = target[1]
            local task_output = target[#target]
            local annotation = torch.zeros(n_nodes, n_annotation_dim)
            for i=2,#target-1 do
                annotation[target[i]][i-1] = 1
            end
            table.insert(task_data_list[task_type], {edge_list, annotation:totable(), task_output})
        end
    end
    return task_data_list
end

-- data_list is a list of {edges, targets} tuples.
--
-- Return standard data, lists of {edges, annotations, targets}, one list per
-- task.  targets is a sequence of targets.
function babi_data.data_list_to_standard_data_seq(data_list, n_annotation_dim)
    local n_nodes = babi_data.find_max_node_id(data_list)
    local n_tasks = babi_data.find_max_task_id(data_list)
    local max_label = {}
    for t=1,n_tasks do
        max_label[t] = babi_data.find_max_label_seq(data_list, t)
    end

    local task_data_list = {}
    for i=1,n_tasks do
        task_data_list[i] = {}
    end

    for _, v in pairs(data_list) do
        local edge_list = v[1]
        local target_list = v[2]
        for _, target in pairs(target_list) do
            local task_type = target[1]
            local annotation = torch.zeros(n_nodes, n_annotation_dim)
            for i=2,3 do
                annotation[target[i]][i-1] = 1
            end
            local task_output = {}
            for i=4,#target do
                table.insert(task_output, target[i])
            end
            table.insert(task_output, max_label[task_type] + 1)    -- add the end-of-sequence label
            table.insert(task_data_list[task_type], {edge_list, annotation:totable(), task_output})
        end
    end
    return task_data_list
end

-- Load and prepare data in standard GNN data format from file.
--
-- n_train when provided and nonzero will be used to split the set of examples
-- into training part and validation part.
function babi_data.prepare_standard_data(filename, n_annotation_dim, n_train)
    if n_train == nil then
        n_train = 0
    end

    local data_list = babi_data.load_graphs_from_file(filename)

    if n_train == 0 then
        return babi_data.data_list_to_standard_data(data_list, n_annotation_dim)
    end

    -- otherwise split the set into two

    local n_examples = #data_list
    local idx = torch.randperm(n_examples)
    local train_set = {}
    local val_set = {}
    for i=1,n_train do
        train_set[i] = data_list[idx[i]]
    end
    for i=1,n_examples-n_train do
        val_set[i] = data_list[idx[n_train+i]]
    end

    return babi_data.data_list_to_standard_data(train_set, n_annotation_dim),
            babi_data.data_list_to_standard_data(val_set, n_annotation_dim)
end

-- data_list should be in standard format, i.e. lists of {edges, annotations, target}
function babi_data.find_max_target(data_list)
    local max_target = 0
    for _, v in pairs(data_list) do
        if type(v[3]) == 'table' then
            for _, vv in pairs(v[3]) do
                if vv > max_target then
                    max_target = vv
                end
            end
        else
            if v[3] > max_target then
                max_target = v[3]
            end
        end
    end
    return max_target
end

---------------------- some utils -------------------------

function babi_data.choose_subset(dataset, subset_size, random_shuffle)
    local n_examples = #dataset
    if subset_size > n_examples then
        subset_size = n_examples
    end

    local idx
    if random_shuffle then
        idx = torch.randperm(n_examples)
    else
        idx = torch.range(1,n_examples)
    end

    local subset = {}
    for i=1,subset_size do
        subset[i] = dataset[idx[i]]
    end
    return subset
end

function babi_data.split_set(dataset, set_size, random_shuffle, split_seed)
    split_seed = split_seed or 1023
    local rand_state = torch.getRNGState()
    -- this makes sure each time the same split is used
    torch.manualSeed(split_seed)

    local n_examples = #dataset
    local idx
    if random_shuffle then
        idx = torch.randperm(n_examples)
    else
        idx = torch.range(1,n_examples)
    end
    torch.setRNGState(rand_state)

    local subsets = {}
    local start_idx = 0
    for i=1,#set_size do
        local subset_size = set_size[i]
        local subset = {}
        for j=1,subset_size do
            subset[j] = dataset[idx[start_idx+j]]
        end
        start_idx = start_idx + subset_size
        subsets[i] = subset
    end
    
    if #subsets == 1 then
        return subsets[1]
    elseif #subsets == 2 then
        return subsets[1], subsets[2]
    elseif #subsets == 3 then
        return subsets[1], subsets[2], subsets[3]
    else
        return subsets
    end
end

-- both x and y are torch tensors, first dimension indexes the examples
function babi_data.split_set_tensor(x, y, n_train, n_val, random_shuffle, split_seed)
    split_seed = split_seed or 1023
    local rand_state = torch.getRNGState()
    -- this makes sure each time the same split is used
    torch.manualSeed(split_seed)

    local n_examples = x:size(1)
    local idx
    if random_shuffle then
        idx = torch.randperm(n_examples)
    else
        idx = torch.range(1,n_examples)
    end
    torch.setRNGState(rand_state)

    local x_train = torch.Tensor(n_train, x:size(2)):typeAs(x)
    local y_train = torch.Tensor(n_train):typeAs(y)
    local x_val = torch.Tensor(n_val, x:size(2)):typeAs(x)
    local y_val = torch.Tensor(n_val):typeAs(y)

    for i=1,n_train do
        x_train:narrow(1,i,1):copy(x:narrow(1,idx[i],1))
        y_train:narrow(1,i,1):copy(y:narrow(1,idx[i],1))
    end

    for i=1,n_val do
        x_val:narrow(1,i,1):copy(x:narrow(1,idx[i+n_train],1))
        y_val:narrow(1,i,1):copy(y:narrow(1,idx[i+n_train],1))
    end

    return x_train, y_train, x_val, y_val
end


-- both input and output are lists
function babi_data.split_set_input_output(x, y, n_train, n_val, random_shuffle, split_seed)
    split_seed = split_seed or 1023
    local rand_state = torch.getRNGState()
    -- this makes sure each time the same split is used
    torch.manualSeed(split_seed)

    local n_examples = #x
    local idx
    if random_shuffle then
        idx = torch.randperm(n_examples)
    else
        idx = torch.range(1,n_examples)
    end
    torch.setRNGState(rand_state)

    local x_train = {}
    local y_train = {}
    local x_val = {}
    local y_val = {}

    for i=1,n_train do
        x_train[i] = x[idx[i]]
        y_train[i] = y[idx[i]]
    end

    for i=1,n_val do
        x_val = x[idx[i+n_train]]
        y_val = y[idx[i+n_train]]
    end

    return x_train, y_train, x_val, y_val
end

-- return true if all the targets are of the same length
function babi_data.targets_are_uniform_length(data_list)
    local len = 0
    for _, v in pairs(data_list) do
        for _, t in pairs(v[2]) do
            if len == 0 then
                len = #t
            end
            if len ~= #t then
                return false
            end
        end
    end
    return true
end

----------------- data loaders ------------------------

local DataLoader = torch.class('babi_data.DataLoader')

-- data is a list of {edges, annotations, target} tuples.
function DataLoader:__init(data, shuffle)
    self.data = data
    self.shuffle = shuffle
    self.n_total = #data

    self:reset()
end

function DataLoader:reset()
    if self.shuffle then
        self.idx = torch.randperm(self.n_total)
    else
        self.idx = torch.range(1, self.n_total)
    end
    self.curr = 1
end

function DataLoader:next()
    local res = self.data[self.idx[self.curr]]
    self.curr = self.curr + 1
    if self.curr > self.n_total then
        self:reset()
    end
    return res[1], res[2], res[3]
end

local PairedDataLoader = torch.class('babi_data.PairedDataLoader')

-- both input and output are lists
function PairedDataLoader:__init(input, output, shuffle)
    self.input = input
    self.output = output
    self.shuffle = shuffle
    self.n_total = #input

    self:reset()
end

function PairedDataLoader:reset()
    if self.shuffle then
        self.idx = torch.randperm(self.n_total)
    else
        self.idx = torch.range(1, self.n_total)
    end
    self.curr = 1
end

function PairedDataLoader:next()
    local x = self.input[self.idx[self.curr]]
    local y = self.output[self.idx[self.curr]]
    self.curr = self.curr + 1
    if self.curr > self.n_total then
        self:reset()
    end
    return x, y
end


--------------- this part is copied from my other project --------------------

local MiniBatchLoader = torch.class('babi_data.MiniBatchLoader')

-- input_data: a tensor of size n_example x ...
-- output_data: a tensor of size n_example x ... or nil
-- minibatch_size: an integer
-- shuffle: boolean
function MiniBatchLoader:__init(input_data, output_data, minibatch_size, shuffle)
    if shuffle == nil then shuffle = false end

    self.input_data = input_data
    self.output_data = output_data

    self.curr_input_data = input_data
    self.curr_output_data = output_data
    self.minibatch_size = minibatch_size
    self.shuffle = shuffle

    self.n_data = input_data:size(1)
    self.join_table = nn.JoinTable(1):type(input_data:type())

    self:reset()
end

function MiniBatchLoader:reset()
    self:reorder()
    self.curr_ptr = 1
end

function MiniBatchLoader:reorder()
    if self.shuffle then
        local idx = torch.randperm(self.n_data)
        local new_data = torch.Tensor(self.input_data:size()):typeAs(self.input_data)
        for i=1,self.n_data do
            new_data:narrow(1,i,1):copy(self.input_data:narrow(1,idx[i],1))
        end
        self.curr_input_data = new_data
        if self.output_data ~= nil then
            new_data = torch.Tensor(self.output_data:size()):typeAs(self.output_data)
            for i=1,self.n_data do
                new_data:narrow(1,i,1):copy(self.output_data:narrow(1,idx[i],1))
            end
            self.curr_output_data = new_data
        end
    end
end

function MiniBatchLoader:next()
    local curr_size = 0
    local minibatch_input = {}
    local minibatch_output = {}

    while curr_size < self.minibatch_size do
        local max_chunk_size = self.minibatch_size - curr_size
        if max_chunk_size > self.n_data - self.curr_ptr + 1 then
            max_chunk_size = self.n_data - self.curr_ptr + 1
        end
        table.insert(minibatch_input, self.curr_input_data:narrow(1,self.curr_ptr,max_chunk_size))
        if self.output_data ~= nil then
            table.insert(minibatch_output, self.curr_output_data:narrow(1,self.curr_ptr,max_chunk_size))
        end
        curr_size = curr_size + max_chunk_size
        self.curr_ptr = self.curr_ptr + max_chunk_size
        if self.curr_ptr > self.n_data then
            self:reorder()
            self.curr_ptr = 1
        end

        -- print(minibatch_input)
        -- print(minibatch_output)
    end

    if #minibatch_input == 1 then
        if self.output_data ~= nil then
            return minibatch_input[1], minibatch_output[1]
        else
            return minibatch_input[1]
        end
    else
        if self.output_data ~= nil then
            return self.join_table:forward(minibatch_input):clone(), self.join_table:forward(minibatch_output):clone()
        else
            return self.join_table:forward(minibatch_input)
        end
    end
end


-------------------------- RNN data loading ------------------------------

function babi_data.load_rnn_data_from_file(filename, n_targets)
    n_targets = n_targets or 1
    local dataset = {}
    for line in io.lines(filename) do
        local example = {}
        for d in line:gmatch('%d+') do
            table.insert(example, tonumber(d))
        end
        table.insert(dataset, example)
    end

    local uniform_length = true
    local seq_len = #dataset[1]
    for i=2,#dataset do
        if seq_len ~= #dataset[i] then
            uniform_length = false
            break
        end
    end

    if uniform_length then
        local data = torch.Tensor(dataset)
        local seq = data:narrow(2,1,data:size(2) - n_targets)
        local target = data:narrow(2, data:size(2) - n_targets + 1, n_targets)

        if n_targets == 1 then
            return seq, target
        else    -- extend sequence, append special end target
            local ext_seq = torch.Tensor(seq:size(1), seq:size(2) + n_targets)
            ext_seq:narrow(2,1,seq:size(2)):copy(seq)
            torch.repeatTensor(ext_seq:narrow(2,seq:size(2)+1, n_targets), seq:narrow(2,seq:size(2),1), 1, n_targets)

            local ext_tgt = torch.Tensor(seq:size(1), n_targets + 1)
            ext_tgt:narrow(2,1,n_targets):copy(target)
            ext_tgt:narrow(2,n_targets+1,1):fill(target:max()+1)  -- append special end symbol

            return ext_seq, ext_tgt
        end
    else
        -- sequence length not equal
        local target = {}
        local seq = {}

        local max_target = 0

        for i=1,#dataset do
            local s = torch.Tensor(dataset[i])
            s = s:resize(1, s:nElement())
            if n_targets == 1 then
                seq[i] = s:narrow(2,1,s:size(2) - n_targets)
                target[i] = s:narrow(2,s:size(2) - n_targets + 1, n_targets)
            else
                seq[i] = torch.Tensor(1, s:size(2) + 1)
                seq[i]:narrow(2,1,s:size(2) - n_targets):copy(s:narrow(2,1,s:size(2) - n_targets))
                seq[i]:narrow(2,s:size(2)-n_targets+1,n_targets):fill(s[s:size(2) - n_targets + 1])

                target[i] = torch.Tensor(1, n_targets + 1)
                target[i]:narrow(2,1,n_targets):copy(s:narrow(2,s:size(2)-n_targets+1, n_targets))

                local t_max = target[i]:narrow(2,1,n_targets):max()
                if t_max > max_target then
                    max_target = t_max
                end
            end
        end

        -- append special end symbol
        if n_targets ~= 1 then
            for i=1,#dataset do
                target[i][target[i]:nElement()] = max_target + 1
            end
        end

        return seq, target
    end
end

function babi_data.find_max_in_list_of_tensors(lst)
    local max = -math.huge
    for k,v in pairs(lst) do
        local m = v:max()
        if m > max then
            max = m
        end
    end
    return max
end

---------- end ----------

return babi_data

