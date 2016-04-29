-- Data loading for the two graph sequence tasks.
--
-- Yujia Li, 11/2015
--

local seq_data = {}

-- data_list is a list of {edges, annotations, targets} triplets.
--
-- This function adds an extra isolated node to each graph, and annotate it 
-- as special using the last annotation bit, and also adds this extra node to
-- the end of the targets sequence.
function seq_data.add_end_node(data_list, add_target)
    if add_target == nil then
        add_target = true
    end
    for _, task_data in pairs(data_list) do
        for _, g in pairs(task_data) do
            local edges, annotations, targets = unpack(g)
            local n_nodes = #annotations
            local annotation_dim = #annotations[1]

            local a = {}
            for i=1,annotation_dim do
                a[i] = 0
            end
            a[annotation_dim] = 1
            annotations[n_nodes+1] = a

            if add_target then
                table.insert(targets, n_nodes+1)
            end
        end
    end
    return data_list
end

function seq_data.load_rnn_data_from_file(filename, end_symbol)
    end_symbol = end_symbol or nil
    local input = {}
    local output = {}
    for line in io.lines(filename) do
        local sep_start, sep_end = line:find('  ')
        if sep_start == nil or sep_end == nil then
            error('Error parsing data file, each line must contain two spaces to separate input and output')
        end
        local seq = line:sub(1,sep_start-1)
        local ans = line:sub(sep_end+1)
        local input_line = {}
        for d in seq:gmatch('%d+') do
            table.insert(input_line, tonumber(d))
        end
        table.insert(input, input_line)
        local output_line = {}
        for d in ans:gmatch('%d+') do
            table.insert(output_line, tonumber(d))
        end
        table.insert(output, output_line)
    end

    local uniform_length = true
    local seq_len = #input[1]
    local tgt_len = #output[1]
    for i=2,#input do
        if seq_len ~= #input[i] or tgt_len ~= #output[i] then
            uniform_length = false
            break
        end
    end

    if uniform_length then
        local seq = torch.Tensor(input)
        local target = torch.Tensor(output)

        local n_targets = target:size(2)
        local ext_seq = torch.Tensor(seq:size(1), seq:size(2) + n_targets)
        ext_seq:narrow(2,1,seq:size(2)):copy(seq)
        torch.repeatTensor(ext_seq:narrow(2,seq:size(2)+1, n_targets), seq:narrow(2,seq:size(2),1), 1, n_targets)

        local ext_tgt = torch.Tensor(seq:size(1), n_targets + 1)
        ext_tgt:narrow(2,1,n_targets):copy(target)

        if end_symbol == nil then
            end_symbol = target:max() + 1
        end
        ext_tgt:narrow(2,n_targets+1,1):fill(end_symbol)  -- append special end symbol

        return ext_seq, ext_tgt, end_symbol
    else
        -- sequence length not equal
        local target = {}
        local seq = {}

        local max_target = 0

        for i=1,#input do
            local s = torch.Tensor(input[i])
            local t = torch.Tensor(output[i])

            s:resize(1, s:nElement())
            t:resize(1, t:nElement())

            local n_targets = t:nElement()

            seq[i] = torch.Tensor(1, s:size(2) + n_targets)
            seq[i]:narrow(2,1,s:size(2)):copy(s)
            seq[i]:narrow(2,s:size(2)+1,n_targets):fill(s[1][s:size(2)])

            target[i] = torch.Tensor(1, n_targets + 1)
            target[i]:narrow(2,1,n_targets):copy(t)

            local t_max = t:max()
            if t_max > max_target then
                max_target = t_max
            end
        end

        if end_symbol == nil then
            end_symbol = max_target + 1
        end

        -- append special end symbol
        for i=1,#input do
            target[i][1][target[i]:nElement()] = end_symbol
        end

        return seq, target, end_symbol
    end
end

return seq_data

