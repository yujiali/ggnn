
-- Create a module with the specified module_type and name, and args, then put
-- it into module_dict.  If a module with the same name already exists in 
-- module_dict share its parameter instead of creating a new one.
function rnn.create_or_share(module_type, name, module_dict, args)
    local result
    if module_type == 'Linear' then
        if module_dict[name] then
            result = nn.Linear(1,1)
            result:share(module_dict[name], 'weight', 'gradWeight', 'bias', 'gradBias')
            -- print('Share ' .. name)
        else
            result = nn.Linear(unpack(args))
            module_dict[name] = result
            -- print('Create ' .. name)
        end
    elseif module_type == 'LookupTable' then
        if module_dict[name] then
            result = nn.LookupTable(1,1)
            result:share(module_dict[name], 'weight', 'gradWeight')
        else
            result = nn.LookupTable(unpack(args))
            module_dict[name] = result
        end
    else
        error(string.format('Unsupported module type: %s', module_type))
    end
    return result
end

function rnn.save_rnn_model(filename, params, model_type, vocab_size, embedding_size, hid_size, output_size)
    torch.save(filename, {params=params, model_type=model_type, 
            vocab_size=vocab_size, embedding_size=embedding_size,
            hid_size=hid_size, output_size=output_size})
end

function rnn.load_rnn_model(filename)
    local d = torch.load(filename)
    local net
    if d['model_type'] == 'rnn' then
        net = rnn.RNN(d['vocab_size'], d['embedding_size'], d['hid_size'], d['output_size'])
    elseif d['model_type'] == 'lstm' then
        net = rnn.LSTM(d['vocab_size'], d['embedding_size'], d['hid_size'], d['output_size'])
    end
    local params, _  = net:getParameters()
    params:copy(d['params'])
    return net
end

