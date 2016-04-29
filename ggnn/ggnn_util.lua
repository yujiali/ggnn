-- Some utility functions.
--
-- Yujia Li, 08/2015
--


-- Create a module with the specified module_type and name, and args, then put
-- it into module_dict.  If a module with the same name already exists in 
-- module_dict share its parameter instead of creating a new one.
function ggnn.create_or_share(module_type, name, module_dict, args)
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

-- Construct one propagation net for one edge
-- state_dim: dimensionality of the state vectors
-- hid_sizes: a table of hidden layer sizes used in the propagation net
-- module_dict: a table of name->module mappings, used for reusing modules
-- input: optional, if provided this will be used as input
-- prefix: prefix of the module names to use in the network
-- edge_type: type of the edge, an int
--
-- Return input and output modules for the propagation net, so that they can be
-- used to construct more complicated networks
-- local function construct_propagation_net(state_dim, hid_sizes, module_dict, input, prefix, edge_type)
function ggnn.construct_propagation_net(state_dim, hid_sizes, module_dict, input, prefix, edge_type)
    input = input or nn.Identity()()
    prefix = prefix or PROP_NET_PREFIX
    edge_type = edge_type or 1
    local x = input
    local n_input = state_dim
    for i=1,#hid_sizes do
        x = nn.Tanh()(ggnn.create_or_share('Linear', prefix .. '-' .. edge_type .. '-' .. i, module_dict, {n_input, hid_sizes[i]})(x))
        n_input = hid_sizes[i]
    end
    -- local output = nn.Tanh()(ggnn.create_or_share('Linear', prefix .. '-' .. edge_type .. '-' .. (#hid_sizes+1), module_dict, {n_input, state_dim})(x))
    local output = ggnn.create_or_share('Linear', prefix .. '-' .. edge_type .. '-' .. (#hid_sizes+1), module_dict, {n_input, state_dim})(x)

    return input, output
end

-- Create all adjacency matrices for graph and concatenate them.
--
-- The ordering of the adjacency matrices must be consistent with the ordering
-- of the propagation networks.
function ggnn.create_adjacency_matrix_cat(edges, n_nodes, n_edge_types)
    local a = torch.Tensor(n_nodes, n_nodes*n_edge_types*2):zero()
    for _, e in ipairs(edges) do
        local src_idx, e_type, tgt_idx = unpack(e)
        a[tgt_idx][(e_type-1)*n_nodes+src_idx] = a[tgt_idx][(e_type-1)*n_nodes+src_idx] + 1
        a[src_idx][(e_type-1+n_edge_types)*n_nodes+tgt_idx] = a[src_idx][(e_type-1+n_edge_types)*n_nodes+tgt_idx] + 1
    end
    return a
end


-- net is a GGNN model or GGNN output model
-- params is its parameter vector
function ggnn.save_model_to_file(file_name, net, params)
    local d = net:get_constructor_param_dict()
    d['params'] = params
    torch.save(file_name, d)
end




function ggnn.get_n_types(module_dict, prefix)
    local n_types = 0
    while module_dict[prefix .. '-' .. (n_types+1) .. '-1'] do
        n_types = n_types + 1
    end
    return n_types
end

function ggnn.get_all_layer_sizes(module_dict, prefix, include_first, has_type)
    include_first = include_first or false
    has_type = has_type or false
    local h_sizes = {}
    local layer_id = 1
    local str_mid = has_type and '-1-' or '-'

    while module_dict[prefix .. str_mid .. layer_id] do
        local layer = module_dict[prefix .. str_mid .. layer_id]
        if layer_id == 1 and include_first then
            table.insert(h_sizes, layer.weight:size(2))
        end
        layer = module_dict[prefix .. str_mid .. layer_id]
        table.insert(h_sizes, layer.weight:size(1))
        layer_id = layer_id + 1
    end

    return h_sizes
end

