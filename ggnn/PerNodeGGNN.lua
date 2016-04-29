-- Graph Neural Network for Per Node Prediction
--
-- Yujia Li, 08/2015
--

local PerNodeGGNN, BaseGGNN = torch.class('ggnn.PerNodeGGNN', 'ggnn.BaseGGNN')

-- GGNN model for per node prediction.
--
-- state_dim: dimensionality of the node representations
-- annotation_dim: dimensionality of the annotations
-- prop_net_h_sizes: hidden layer sizes for the propagation net, excluding 
--      input and output layers - they should be the same as state_dim
-- output_net_h_sizes: hidden layer sizes for the per node prediction net,
--      excluding the input layer and the output layer which should be 
--      state_dim+annotation_dim and annotation_dim respectively.
-- n_edge_types: number of different edge types.
-- module_dict: provide this to load a trained model.
function PerNodeGGNN:__init(state_dim, annotation_dim, prop_net_h_sizes, output_net_h_sizes, n_edge_types, module_dict, n_outputs)
    BaseGGNN.__init(self, state_dim, annotation_dim, prop_net_h_sizes, n_edge_types, module_dict)

    self.output_net_h_sizes = output_net_h_sizes
    self.n_outputs = n_outputs or annotation_dim
    self:create_output_net()
end

function PerNodeGGNN:get_constructor_param_dict()
    return {
        state_dim=self.state_dim,
        annotation_dim=self.annotation_dim,
        prop_net_h_sizes=self.prop_net_h_sizes,
        output_net_h_sizes=self.output_net_h_sizes,
        n_edge_types=self.n_edge_types,
        n_outputs=self.n_outputs
    }
end

function ggnn.load_per_node_ggnn_from_file(file_name)
    local d = torch.load(file_name)
    local net = ggnn.PerNodeGGNN(
        d['state_dim'],
        d['annotation_dim'],
        d['prop_net_h_sizes'],
        d['output_net_h_sizes'],
        d['n_edge_types'],
        nil,
        d['n_outputs']
    )
    local w = net:getParameters()
    w:copy(d['params'])

    return net
end

-- Creates a copy of this network sharing the same module_dict - i.e. using 
-- exactly the same set of parameters.
function PerNodeGGNN:create_share_param_copy()
    return ggnn.PerNodeGGNN(self.state_dim, self.annotation_dim, self.prop_net_h_sizes, self.output_net_h_sizes, self.n_edge_types, self.module_dict, self.n_outputs)
end

-- This output net takes the final node representations and initial
-- annotations as input and outputs the updated annotations.  Can be easily
-- changed to make other per-node predictions.
function PerNodeGGNN:create_output_net()
    local input = nn.Identity()()
    local layer_input = input
    local in_dim = self.state_dim + self.annotation_dim

    for i, h_dim in ipairs(self.output_net_h_sizes) do
        layer_input = nn.Tanh()(ggnn.create_or_share('Linear', ggnn.OUTPUT_NET_PREFIX .. '-' .. i, self.module_dict, {in_dim, h_dim})(layer_input))
        in_dim = h_dim
    end

    local output = ggnn.create_or_share('Linear', ggnn.OUTPUT_NET_PREFIX .. '-' .. (#self.output_net_h_sizes+1), self.module_dict, {in_dim, self.n_outputs})(layer_input)
    
    -- Need to do a sigmoid transformation to use this with binary cross entropy loss.
    local final_output = nn.Sigmoid()(output)

    self.output_net = nn.gModule({input}, {final_output})
end

-- Forward pass, takes the edges_list to construct propagation networks, then
-- take annotation_list to build inputs and push the inputs through the network
-- to get selected node.
function PerNodeGGNN:forward(edges_list, n_steps, annotations_list)
    BaseGGNN.forward(self, edges_list, n_steps, annotations_list)

    self.output_net_input = torch.Tensor(self.n_total_nodes, self.state_dim + self.annotation_dim)
    self.output_net_input:narrow(2,1,self.state_dim):copy(self.prop_inputs[n_steps+1])
    self.output_net_input:narrow(2,self.state_dim+1,self.annotation_dim):copy(self.prop_inputs[1]:narrow(2,1,self.annotation_dim))

    self.output = self.output_net:forward(self.output_net_input)

    return self.output
end

-- Call this only after a forward pass.
function PerNodeGGNN:breakdown_annotations(annotation_tensor)
    local new_annotations_list = {}
    local idx_offset = 0
    for i, n_nodes in ipairs(self.n_nodes_list) do
        new_annotations_list[i] = annotation_tensor:narrow(1,idx_offset+1,n_nodes):totable()
        idx_offset = idx_offset + n_nodes
    end
    return new_annotations_list
end

function PerNodeGGNN:predict_graph_batch(edges_list, n_steps, annotations_list)
    local all_annotations = self:forward(edges_list, n_steps, annotations_list):ge(0.5)
    local new_annotations_list = {}
    local idx_offset = 0
    for i, n_nodes in ipairs(self.n_nodes_list) do
        new_annotations_list[i] = all_annotations:narrow(1,idx_offset+1,n_nodes):totable()
        idx_offset = idx_offset + n_nodes
    end
    return new_annotations_list
end

-- Make predictions for a single graph. Return binarized annotation predictions.
function PerNodeGGNN:predict_single_graph(edges, n_steps, annotations)
    return self:forward({edges}, n_steps, {annotations}):ge(0.5):totable()
end

-- The backward pass.  output_grad is the gradient tensor obtained from the 
-- loss criterion.  Assumes that forward function has already been called on
-- the same input.
function PerNodeGGNN:backward(output_grad)
    local layer_grad = self.output_net:backward(self.output_net_input, output_grad)
    local annotation_grad = torch.Tensor(self.n_total_nodes, self.annotation_dim):copy(layer_grad:narrow(2,self.state_dim+1,self.annotation_dim))
    layer_grad = layer_grad:narrow(2,1,self.state_dim)

    annotation_grad:add(BaseGGNN.backward(self, layer_grad))
    return annotation_grad
end

-- Print out propagation net and the dot-product layer.
function PerNodeGGNN:str_repr()
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
-- the GNN.  
--
-- targets is a list of target annotations.
function ggnn.compute_annotation_ggnn_loss_and_grad(criterion, output, targets, compute_grad)
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
    for i, target in ipairs(targets) do
        local n_nodes = #target

        output_part = output:narrow(1, idx_offset+1, n_nodes)
        target = torch.Tensor(target):type(output:type())
        loss = loss + criterion:forward(output_part, target)
        if compute_grad then
            grad:narrow(1, idx_offset+1, n_nodes):copy(criterion:backward(output_part, target))
        end
        idx_offset = idx_offset + n_nodes
    end
    loss = loss / #targets

    if compute_grad then
        grad:mul(1 / #targets)
        return loss, grad
    else
        return loss
    end
end

