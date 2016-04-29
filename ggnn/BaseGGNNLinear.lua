-- The base GGNN class
--
-- Exactly the same as BaseGGNN.lua, except that the Tanh nonlinearity is
-- removed.
--
-- Yujia Li, 08/2015
--

local BaseGGNN = torch.class('ggnn.BaseGGNN')

function BaseGGNN:__init(state_dim, annotation_dim, prop_net_h_sizes, n_edge_types, module_dict)
    self.state_dim = state_dim
    self.annotation_dim = annotation_dim

    assert(self.state_dim >= self.annotation_dim, 'state_dim must be no less than annotation_dim')

    self.n_edge_types = n_edge_types

    self.module_dict = module_dict or {}

    self.prop_net_h_sizes = prop_net_h_sizes

    -- all modules that have parameters have to be created at initialization
    -- time, otherwise the module parameters won't be captured by the parameter
    -- vector at the begining of training.
    self:create_propagation_net_modules()
    self:create_node_update_net_modules()
end

-- Creates a copy of this network sharing the same module_dict - i.e. using 
-- exactly the same set of parameters.
function BaseGGNN:create_share_param_copy()
    return ggnn.BaseGGNN(self.state_dim, self.annotation_dim, self.prop_net_h_sizes, self.n_edge_types, self.module_dict)
end

-- Return a dictionary of parameters that can be used to call the constructor
-- to create a GGNN model with the same architecture.  Used when saving a
-- model to file.
function BaseGGNN:get_constructor_param_dict()
    return {
        state_dim=self.state_dim,
        annotation_dim=self.annotation_dim,
        prop_net_h_sizes=self.prop_net_h_sizes,
        n_edge_types=self.n_edge_types
    }
end

-- Load a BaseGGNN model from file, this must be paired with get_constructor_param_dict.
function ggnn.load_base_ggnn_from_file(file_name)
    local d = torch.load(file_name)
    local net = ggnn.BaseGGNN(d['state_dim'], d['annotation_dim'], d['prop_net_h_sizes'], d['n_edge_types'])
    local w = net:getParameters()
    w:copy(d['params'])

    return net
end

-- Create all shared modules of the propagation net.
function BaseGGNN:create_propagation_net_modules()
    local layer_in_dim, layer_out_dim
    for edge_type=1,self.n_edge_types do
        layer_in_dim = self.state_dim
        for i, h_dim in ipairs(self.prop_net_h_sizes) do
            layer_out_dim = h_dim
            ggnn.create_or_share('Linear', ggnn.PROP_NET_PREFIX .. '-' .. edge_type .. '-' .. i, self.module_dict, {layer_in_dim, layer_out_dim})
            ggnn.create_or_share('Linear', ggnn.REVERSE_PROP_NET_PREFIX .. '-' .. edge_type .. '-' .. i, self.module_dict, {layer_in_dim, layer_out_dim})
            layer_in_dim = h_dim
        end
        layer_out_dim = self.state_dim
        ggnn.create_or_share('Linear', ggnn.PROP_NET_PREFIX .. '-' .. edge_type .. '-' .. #self.prop_net_h_sizes+1, self.module_dict, {layer_in_dim, layer_out_dim})
        ggnn.create_or_share('Linear', ggnn.REVERSE_PROP_NET_PREFIX .. '-' .. edge_type .. '-' .. #self.prop_net_h_sizes+1, self.module_dict, {layer_in_dim, layer_out_dim})
    end
end

-- Create all shared modules for the node update network, which takes all the 
-- propagated state updates as input and combine them together into a single
-- node representation.  The node_update_net replaces the previous add_net, as 
-- it is not simply "adding" things together.
function BaseGGNN:create_node_update_net_modules()
    ggnn.create_or_share('Linear', ggnn.NODE_UPDATE_NET_PREFIX .. '-transform', self.module_dict, {self.state_dim * 3, self.state_dim})
    ggnn.create_or_share('Linear', ggnn.NODE_UPDATE_NET_PREFIX .. '-gate', self.module_dict, {self.state_dim * 3, self.state_dim * 2})
end

-- Create a node update net for a single layer.
--
-- Input contains 2*n_edge_types input for each edge type, 1 input for the
-- node representation in the previous layer, and n_graphs adjacency matrices.
function BaseGGNN:create_one_node_update_net(n_nodes_list)
    local n_graphs = #n_nodes_list
    local n_edge_types = self.n_edge_types
    local input = {}
    -- input for each type of edge (2*n_edge_types) + input for self prop net (1) + n_graphs adjacency matrices
    for i=1,n_edge_types*2+1+n_graphs do
        input[i] = nn.Identity()()
    end

    local output = {}
    local idx_offset = 0
    for i=1,n_graphs do
        local temp_input = {}
        for j=1,n_edge_types do
            temp_input[j] = nn.Narrow(1,idx_offset+1,n_nodes_list[i])(input[j])
        end
        if n_edge_types > 1 then
            temp_input = nn.JoinTable(1,2)(temp_input)
        else
            temp_input = temp_input[1]
        end
        local forward_input = nn.MM()({nn.Narrow(2,1,n_nodes_list[i]*n_edge_types)(input[self.n_edge_types*2+1+i]), temp_input})

        temp_input = {}
        for j=1,n_edge_types do
            temp_input[j] = nn.Narrow(1,idx_offset+1,n_nodes_list[i])(input[n_edge_types+j])
        end
        if n_edge_types > 1 then
            temp_input = nn.JoinTable(1,2)(temp_input)
        else
            temp_input = temp_input[1]
        end
        local reverse_input = nn.MM()({nn.Narrow(2,n_nodes_list[i]*n_edge_types+1,n_nodes_list[i]*n_edge_types)(input[n_edge_types*2+1+i]), temp_input})
        local current_state = nn.Narrow(1, idx_offset+1, n_nodes_list[i])(input[2*n_edge_types+1])

        local joined_input = nn.JoinTable(2,2)({forward_input, reverse_input, current_state})
        local gates = nn.Sigmoid()(ggnn.create_or_share('Linear', ggnn.NODE_UPDATE_NET_PREFIX .. '-gate', self.module_dict, {self.state_dim * 3, self.state_dim * 2})(joined_input))

        local update_gate = nn.Narrow(2, 1, self.state_dim)(gates)
        local reset_gate = nn.Narrow(2, self.state_dim+1, self.state_dim)(gates)
        
        joined_input = nn.JoinTable(2,2)({forward_input, reverse_input, nn.CMulTable()({reset_gate, current_state})})
        local transformed_output = ggnn.create_or_share('Linear', ggnn.NODE_UPDATE_NET_PREFIX .. '-transform', self.module_dict, {self.state_dim * 3, self.state_dim})(joined_input)

        output[i] = nn.CAddTable()({current_state, nn.CMulTable()({update_gate, nn.CSubTable()({transformed_output, current_state})})})

        idx_offset = idx_offset + n_nodes_list[i]
    end

    if n_graphs > 1 then
        output = nn.JoinTable(1,2)(output)
    else
        output = output[1]
    end

    return nn.gModule(input, {output})
end

-- Construct the computation graph for the propagation process on a given
-- graph, reusing pre-allocated parameters.  Edges are not used in the 
-- construction, as the edge information will be passed in as adjacency
-- matrices.
--
-- n_steps is the number of full iterations to run in the propagation step
-- n_nodes_list is a list of number of nodes in each graph
function BaseGGNN:construct_network_for_graphs(n_steps, n_nodes_list)
    self.n_nodes_list = n_nodes_list
    self.n_steps = n_steps

    local n_graphs = #n_nodes_list
    self.n_graphs = n_graphs

    local input, output

    if not self.prop_net then
        self.prop_net = {}
    end

    if not self.prop_net[n_steps] then
        for i_step=1,n_steps do
            if not self.prop_net[i_step] then
                self.prop_net[i_step] = {}
                for e_type=1,self.n_edge_types do
                    input, output = ggnn.construct_propagation_net(self.state_dim, self.prop_net_h_sizes, self.module_dict, nil, ggnn.PROP_NET_PREFIX, e_type)
                    self.prop_net[i_step][e_type] = nn.gModule({input}, {output})

                    input, output = ggnn.construct_propagation_net(self.state_dim, self.prop_net_h_sizes, self.module_dict, nil, ggnn.REVERSE_PROP_NET_PREFIX, e_type)
                    self.prop_net[i_step][e_type+self.n_edge_types] = nn.gModule({input}, {output})
                end
            end
        end
    end

    self.add_net = {}
    for i_step=1,n_steps do 
        self.add_net[i_step] = self:create_one_node_update_net(n_nodes_list)
    end
end

-- Return a list of parameters and a list of parameter gradients. This function
-- is inspired by the parameters function in nn/Container.lua
function BaseGGNN:parameters()
    local w = {}
    local dw = {}

    -- sort the keys to make sure the parameters are always in the same order
    local k_list = {}
    for k, v in pairs(self.module_dict) do
        table.insert(k_list, k)
    end
    table.sort(k_list)

    for i=1,#k_list do
        m = self.module_dict[k_list[i]]
        local mw, mdw = m:parameters()
        if mw then
            if type(mw) == 'table' then
                for i=1,#mw do
                    table.insert(w, mw[i])
                    table.insert(dw, mdw[i])
                end
            else
                table.insert(w, mw)
                table.insert(dw, mdw)
            end
        end
    end
    return w, dw
end

-- Return a flattened version of the list of parameters, and a flattened version
-- of the list of parameter gradients.
--
-- This function directly calls nn.Module.flatten
function BaseGGNN:getParameters()
    local params, grad_params = self:parameters()
    return nn.Module.flatten(params), nn.Module.flatten(grad_params)
end

-- adjacency_list is a list of extended adjacency matrices
-- n_steps is the number of propagation steps
-- annotations is a concatenated annotation matrix, with the annotations for 
-- first graph taking the top rows, second graph after that and so on.
function BaseGGNN:forward_with_adjacency_matrices_and_concatenated_annotation_matrix(adjacency_list, n_steps, annotations)
    local n_nodes_list = {}
    local n_total_nodes = 0
    for i, adj in ipairs(adjacency_list) do
        table.insert(n_nodes_list, adj:size(1))
        n_total_nodes = n_total_nodes + adj:size(1)
    end

    self.n_nodes_list = n_nodes_list
    self.n_steps = n_steps

    self:construct_network_for_graphs(n_steps, n_nodes_list)

    self.a_list = adjacency_list

    self.prop_inputs = {}
    self.prop_outputs = {}

    assert(annotations:size(1) == n_total_nodes)

    local input = torch.zeros(n_total_nodes, self.state_dim)
    input:narrow(2,1,self.annotation_dim):copy(annotations)

    self._annotations = annotations -- keep this for use in the future

    self.prop_inputs[1] = input
    for i_step=1,n_steps do
        self.prop_outputs[i_step] = {}
        for i=1,self.n_edge_types*2 do
            self.prop_outputs[i_step][i] = self.prop_net[i_step][i]:forward(self.prop_inputs[i_step])
        end
        self.prop_outputs[i_step][self.n_edge_types*2+1] = self.prop_inputs[i_step]
        for i=1,self.n_graphs do
            table.insert(self.prop_outputs[i_step], self.a_list[i])
        end
        self.prop_inputs[i_step+1] = self.add_net[i_step]:forward(self.prop_outputs[i_step])
    end

    self.n_total_nodes = n_total_nodes

    return self.prop_inputs[n_steps+1]
end

-- adjacency_list is a list of extended adjacency matrices
-- n_steps is the number of propagation steps
-- annotations_list is a list of annotation matrices
function BaseGGNN:forward_with_adjacency_and_annotation_matrices(adjacency_list, n_steps, annotations_list)
    if self._cat_annotations_net == nil then
        self._cat_annotations_net = nn.JoinTable(1)
    end

    local annotations = self._cat_annotations_net:forward(annotations_list)
    return self:forward_with_adjacency_matrices_and_concatenated_annotation_matrix(adjacency_list, n_steps, annotations)
    --]]
end

-- Forward pass, takes the edges_list to construct propagation networks, then
-- take annotation_list to build inputs and push the inputs through the network.
--
-- If adjacency matrix and annotation tensors were already constructed then
-- edges_list can also be used as a list of adjacency matrices, and 
-- annotations_list can be used a list of annotation tensors.
--
-- If edges_list is a list of adjacency matrices, then annotations_list can
-- also just be a single tensor, which is the concatenated annotations for all
-- graphs.
function BaseGGNN:forward(edges_list, n_steps, annotations_list)
    if type(annotations_list) == 'userdata' then   -- single tensor case
        assert(type(edges_list[1]) == 'userdata')  -- edges_list must be a list of adjacency matrices
        return self:forward_with_adjacency_matrices_and_concatenated_annotation_matrix(edges_list, n_steps, annotations_list)
    end

    assert(#edges_list == #annotations_list)

    local annotations_tensor_list = {}
    local adjacency_matrix_list = {}

    for i, annotations in ipairs(annotations_list) do
        if type(annotations) == 'table' then
            annotations_tensor_list[i] = torch.Tensor(annotations)
        else
            annotations_tensor_list[i] = annotations
        end

        if type(edges_list[i]) == 'table' then
            adjacency_matrix_list[i] = ggnn.create_adjacency_matrix_cat(edges_list[i], annotations_tensor_list[i]:size(1), self.n_edge_types)
        else
            adjacency_matrix_list[i] = edges_list[i]
        end
    end

    return self:forward_with_adjacency_and_annotation_matrices(adjacency_matrix_list, n_steps, annotations_tensor_list)
end

-- The backward pass.  output_grad is the gradient tensor obtained from the 
-- loss criterion.  Assumes that forward function has already been called on
-- the same input.
function BaseGGNN:backward(output_grad)
    local layer_grad = output_grad

    for i_step=self.n_steps,1,-1 do
        local prop_grad = self.add_net[i_step]:backward(self.prop_outputs[i_step], layer_grad)
        layer_grad:zero()
        for i=1,self.n_edge_types*2 do
            layer_grad:add(self.prop_net[i_step][i]:backward(self.prop_inputs[i_step], prop_grad[i]))
        end
        layer_grad:add(prop_grad[self.n_edge_types*2+1])
    end

    return layer_grad:narrow(2,1,self.annotation_dim)
end

-- Print out propagation net and the dot-product layer.
function BaseGGNN:str_repr()
    local s = 'Propagation Net: ' .. self.module_dict[ggnn.PROP_NET_PREFIX .. '-1-1'].weight:size(2)
    local layer_id = 1
    while self.module_dict[ggnn.PROP_NET_PREFIX .. '-1-' .. layer_id] do
        s = s .. '-' .. self.module_dict[ggnn.PROP_NET_PREFIX .. '-1-' .. layer_id].weight:size(1)
        layer_id = layer_id + 1
    end
    return s
end

function BaseGGNN:print_model()
    print(self:str_repr())
end


