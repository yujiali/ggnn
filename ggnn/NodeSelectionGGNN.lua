--
-- Gated Graph Neural Network For Node Selection
--
-- Yujia Li, 08/2015
--

local NodeSelectionGGNN, BaseGGNN = torch.class('ggnn.NodeSelectionGGNN', 'ggnn.BaseGGNN')

function NodeSelectionGGNN:__init(state_dim, annotation_dim, prop_net_h_sizes, output_net_h_sizes, n_edge_types, module_dict)
    BaseGGNN.__init(self, state_dim, annotation_dim, prop_net_h_sizes, n_edge_types, module_dict)

    self.output_net_h_sizes = output_net_h_sizes
    self:create_output_net()
end

function NodeSelectionGGNN:get_constructor_param_dict()
    return {
        state_dim=self.state_dim,
        annotation_dim=self.annotation_dim,
        prop_net_h_sizes=self.prop_net_h_sizes,
        output_net_h_sizes=self.output_net_h_sizes,
        n_edge_types=self.n_edge_types
    }
end

function ggnn.load_node_selection_ggnn_from_file(file_name)
    local d = torch.load(file_name)
    local net = ggnn.NodeSelectionGGNN(
        d['state_dim'],
        d['annotation_dim'],
        d['prop_net_h_sizes'],
        d['output_net_h_sizes'],
        d['n_edge_types'],
        nil
    )
    local w = net:getParameters()
    w:copy(d['params'])

    return net
end

-- Creates a copy of this network sharing the same module_dict - i.e. using 
-- exactly the same set of parameters.
function NodeSelectionGGNN:create_share_param_copy()
    return ggnn.NodeSelectionGGNN(self.state_dim, self.annotation_dim, self.prop_net_h_sizes, self.output_net_h_sizes, self.n_edge_types, self.module_dict)
end

-- This network takes the node representations as input and convert
-- these into a real number score.  Node annotations are used as well for
-- gating.
function NodeSelectionGGNN:create_output_net()
    local input = nn.Identity()()
    local layer_input = input
    local in_dim = self.state_dim + self.annotation_dim

    for i, h_dim in ipairs(self.output_net_h_sizes) do
        layer_input = nn.Tanh()(ggnn.create_or_share('Linear', ggnn.OUTPUT_NET_PREFIX .. '-' .. i, self.module_dict, {in_dim, h_dim})(layer_input))
        in_dim = h_dim
    end
    -- final output dimensionality is 1, a single real number score
    local output = ggnn.create_or_share('Linear', ggnn.OUTPUT_NET_PREFIX .. '-' .. (#self.output_net_h_sizes+1), self.module_dict, {in_dim, 1})(layer_input)

    local gate = nn.Sigmoid()(ggnn.create_or_share('Linear', ggnn.GATING_NET_PREFIX, self.module_dict, {self.state_dim + self.annotation_dim, 1})(input))
    local gated_output = nn.CMulTable()({output, gate})

    self.output_net = nn.gModule({input}, {gated_output})
end

-- Forward pass, takes the edges_list to construct propagation networks, then
-- take annotation_list to build inputs and push the inputs through the network
-- to get selected node.
function NodeSelectionGGNN:forward(edges_list, n_steps, annotations_list)
    BaseGGNN.forward(self, edges_list, n_steps, annotations_list)

    self.output_net_input = torch.Tensor(self.n_total_nodes, self.state_dim + self.annotation_dim)
    self.output_net_input:narrow(2,1,self.state_dim):copy(self.prop_inputs[n_steps+1])
    self.output_net_input:narrow(2,self.state_dim+1,self.annotation_dim):copy(self.prop_inputs[1]:narrow(2,1,self.annotation_dim))

    self.output_scores = self.output_net:forward(self.output_net_input)

    return self.output_scores
end

-- This is a convenience function for making predictions.
-- Return a 1-D tensor same size as the number of graphs.
function NodeSelectionGGNN:predict(edges_list, n_steps, annotations_list)
    local score = self:forward(edges_list, n_steps, annotations_list):squeeze()
    local pred = torch.Tensor(self.n_graphs):zero()

    local idx_offset = 0
    for i, n_nodes in ipairs(self.n_nodes_list) do
        _, max_idx = score:narrow(1, idx_offset+1, n_nodes):max(1)
        pred[i] = max_idx
        idx_offset = idx_offset + n_nodes
    end
    return pred
end

-- Prediction for a single graph.
function NodeSelectionGGNN:predict_single_graph(edges, n_steps, annotations, n_nodes_to_select)
    n_nodes_to_select = n_nodes_to_select or 1
    local scores = self:forward({edges}, n_steps, {annotations}):squeeze()
    local _, sorted_idx = scores:sort(1,true)
    local res = {}
    for i=1,n_nodes_to_select do
        res[i] = sorted_idx[i]
    end
    return res
end

function NodeSelectionGGNN:predict_single_graph_threshold(edges, n_steps, annotations, thres)
    thres = thres or 0
    local scores = self:forward({edges}, n_steps, {annotations}):squeeze()
    local res = {}
    for i=1,scores:nElement() do
        if scores[i] >= thres then
            table.insert(res, i)
        end
    end
    return res
end

-- Choose a single arg.  Note the arg can be associated with multiple nodes,
-- the score of all the nodes are summed together.
function NodeSelectionGGNN:predict_single_graph_single_arg(edges, n_steps, annotations, arg_map)
    local scores = self:forward({edges}, n_steps, {annotations}):squeeze()
    local max_score = -math.huge
    local max_arg
    for arg_name, node_ids in pairs(arg_map) do
        local score = 0
        if type(node_ids) == 'table' then
            for _, node_id in pairs(node_ids) do
                score = score + scores[node_id]
            end
        else
            score = scores[node_ids]
        end
        if score > max_score then
            max_score = score
            max_arg = arg_name
        end
    end
    return max_arg
end

-- Choose a single arg.
function NodeSelectionGGNN:predict_graph_batch_single_arg(edges_list, n_steps, annotations_list, arg_map_list)
    local n_graphs = #edges_list
    local score = self:forward(edges_list, n_steps, annotations_list):squeeze()
    local max_score = -math.huge
    local max_arg
    for arg_name, _ in pairs(arg_map_list[1]) do
        local arg_score = 0
        local idx_offset = 0
        for i=1,n_graphs do
            local node_ids = arg_map_list[i][arg_name]
            if type(node_ids) == 'table' then
                for _,node_id in pairs(node_ids) do
                    arg_score = arg_score + score[idx_offset + node_id]
                end
            else
                arg_score = arg_score + score[idx_offset + node_ids]
            end
            idx_offset = idx_offset + self.n_nodes_list[i]
        end
        if arg_score > max_score then
            max_score = arg_score
            max_arg = arg_name
        end
    end
    return max_arg
end

-- The backward pass.  output_grad is the gradient tensor obtained from the 
-- loss criterion.  Assumes that forward function has already been called on
-- the same input.
function NodeSelectionGGNN:backward(output_grad)
    local layer_grad = self.output_net:backward(self.output_net_input, output_grad)
    local annotation_grad = torch.Tensor(self.n_total_nodes, self.annotation_dim):copy(layer_grad:narrow(2,self.state_dim+1,self.annotation_dim))
    layer_grad = layer_grad:narrow(2,1,self.state_dim)
    annotation_grad:add(BaseGGNN.backward(self, layer_grad))
    return annotation_grad
end

-- Print out propagation net and the dot-product layer.
function NodeSelectionGGNN:str_repr()
    local s = BaseGGNN.str_repr(self)
    s = s .. ' | Output Net: ' .. self.module_dict[ggnn.OUTPUT_NET_PREFIX .. '-1'].weight:size(2)
    local layer_id = 1
    while self.module_dict[ggnn.OUTPUT_NET_PREFIX .. '-' .. layer_id] do
        s = s .. '-' .. self.module_dict[ggnn.OUTPUT_NET_PREFIX .. '-' .. layer_id].weight:size(1)
        layer_id = layer_id + 1
    end
    return s
end

-- A convenience function for computing loss and gradient for the output from
-- the GNN.  n_nodes_list must be provided to determine how to split the output.
function ggnn.compute_node_selection_loss_and_grad(criterion, output, targets, n_nodes_list, compute_grad)
    if compute_grad == nil then
        compute_grad = true
    end

    local loss = 0
    local grad

    if compute_grad then
        grad = torch.Tensor(output:size()):zero()
    end

    local idx_offset = 0
    local output_part
    for i, n_nodes in ipairs(n_nodes_list) do
        output_part = output:narrow(1, idx_offset+1, n_nodes)
        loss = loss + criterion:forward(output_part, targets[i])
        if compute_grad then
            grad:narrow(1, idx_offset+1, n_nodes):copy(criterion:backward(output_part, targets[i]))
        end
        idx_offset = idx_offset + n_nodes
    end

    if compute_grad then
        return loss, grad
    else
        return loss
    end
end

-- A convenience function for computing final predictions from output vector.
function ggnn.compute_node_selection_predictions(output, n_nodes_list)
    local pred = {}
    local idx_offset = 0
    output = output:squeeze()
    for i, n_nodes in ipairs(n_nodes_list) do
        local _, p = output:narrow(1, idx_offset+1, n_nodes):max(1)
        pred[i] = p[1]
        idx_offset = idx_offset + n_nodes
    end
    return torch.Tensor(pred):type('torch.LongTensor')
end

-- A multi node version of compute_node_selection_loss_and_grad
-- Here targets is a list of target ID lists.
function ggnn.compute_multi_node_selection_loss_and_grad(criterion, output, targets, n_nodes_list, compute_grad, debug)
    if compute_grad == nil then compute_grad = true end
    if debug == nil then debug = false end

    local loss = 0
    local grad

    local target = torch.Tensor(output:size()):zero()

    -- load target vector
    local idx_offset = 0
    for i, n_nodes in ipairs(n_nodes_list) do
        for _, t in pairs(targets[i]) do
            target[idx_offset + t] = 1
        end
        idx_offset = idx_offset + n_nodes
    end

    if debug then
        print('[Target]')
        for i=1,target:nElement() do
            io.write(string.format('%3d : %.4f %d', i, output[i]:squeeze(), target[i]:squeeze()))
            if target[i]:squeeze() == 1 then io.write('*') end
            io.write('\n')
        end
    end

    local sigmoid = nn.Sigmoid()
    local sigmoid_output = sigmoid:forward(output)

    loss = loss + criterion:forward(sigmoid_output, target)
    if compute_grad then
        return loss, sigmoid:backward(output, criterion:backward(sigmoid_output, target))
    else
        return loss
    end
end

