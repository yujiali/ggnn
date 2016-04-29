-- Gated Graph Neural Networks for Graph Level Prediction
--
-- Yujia Li, 08/2015
--

local GraphLevelGGNN, BaseGGNN = torch.class('ggnn.GraphLevelGGNN', 'ggnn.BaseGGNN')

-- state_dim: dimensionality of the state vectors
-- prop_net_h_sizes: number of hidden units on each layer
--      for the propagation net and classification net separately
-- output_net_sizes: number of units on each layer of the classification net,
--      the first number in the list is the input size, and the last number is 
--      the output size.
function GraphLevelGGNN:__init(state_dim, annotation_dim, prop_net_h_sizes, output_net_sizes, n_edge_types, module_dict)
    BaseGGNN.__init(self, state_dim, annotation_dim, prop_net_h_sizes, n_edge_types, module_dict)

    self.output_net_sizes = output_net_sizes
    self.n_classes = output_net_sizes[#output_net_sizes]
    self.class_net_in_dim = output_net_sizes[1]

    self:create_aggregation_net_modules()
    self:create_output_net()
end

function GraphLevelGGNN:get_constructor_param_dict()
    return {
        state_dim=self.state_dim,
        annotation_dim=self.annotation_dim,
        prop_net_h_sizes=self.prop_net_h_sizes,
        output_net_sizes=self.output_net_sizes,
        n_edge_types=self.n_edge_types
    }
end

function ggnn.load_graph_level_ggnn_from_file(file_name)
    local d = torch.load(file_name)
    local net = ggnn.GraphLevelGGNN(
        d['state_dim'],
        d['annotation_dim'],
        d['prop_net_h_sizes'],
        d['output_net_sizes'],
        d['n_edge_types'],
        nil
    )
    local w = net:getParameters()
    w:copy(d['params'])

    return net
end


-- Creates a copy of this network sharing the same module_dict - i.e. using 
-- exactly the same set of parameters.
function GraphLevelGGNN:create_share_param_copy()
    return ggnn.GraphLevelGGNN(self.state_dim, self.annotation_dim, self.prop_net_h_sizes, self.output_net_sizes, self.n_edge_types, self.module_dict)
end

-- Create all shared modules of the aggregation net.  Actually just two modules
-- the linear module linked to the gates and the linear module to the
-- transformed outputs. Here to be more efficient we construct the two linear
-- modules as one and then use Narrow modules to get separate modules out of
-- the results.
function GraphLevelGGNN:create_aggregation_net_modules()
    ggnn.create_or_share('Linear', ggnn.AGGREGATION_NET_PREFIX .. '-input', self.module_dict, {self.state_dim+self.annotation_dim, 2*self.class_net_in_dim})
end


-- create the output network, which will not change across graphs
function GraphLevelGGNN:create_output_net()
    local layer_in_dim, layer_out_dim
    local input = nn.Identity()()
    local x_input = input
    for i=1,#self.output_net_sizes-2 do
        x_input = nn.Tanh()(ggnn.create_or_share('Linear', ggnn.OUTPUT_NET_PREFIX .. '-' .. i, 
                self.module_dict, {self.output_net_sizes[i], self.output_net_sizes[i+1]})(x_input))
    end

    if #self.output_net_sizes > 1 then
        x_input = ggnn.create_or_share('Linear', ggnn.OUTPUT_NET_PREFIX .. '-' .. #self.output_net_sizes-1, self.module_dict,
                {self.output_net_sizes[#self.output_net_sizes-1], self.output_net_sizes[#self.output_net_sizes]})(x_input)
    end

    -- Sigmoid for two class problems
    -- local output = nn.Sigmoid()(x_input)
    
    -- Otherwise the linear layer output is the output
    local output = x_input

    self.output_net = nn.gModule({input}, {output})
end

-- This is a variant of the original aggregation net architecture.  Where the
-- input is a concatenation of the final node representations and the initial
-- node annotations.  Then both the gates and the transformed representations
-- get input from the concatenated node representations.
--
-- As a comarison, the previous approach uses both the node representations
-- and initial annotations when computing gate activations, but only node
-- representations in computing transformed representations.
function GraphLevelGGNN:create_aggregation_net(n_nodes_list)
    local input = nn.Identity()()

    local act = ggnn.create_or_share('Linear', ggnn.AGGREGATION_NET_PREFIX .. '-input', self.module_dict, {self.state_dim+self.annotation_dim, 2*self.class_net_in_dim})(input)
    local gates = nn.Sigmoid()(nn.Narrow(2, 1, self.class_net_in_dim)(act))
    local h = nn.Tanh()(nn.Narrow(2, self.class_net_in_dim+1, self.class_net_in_dim)(act))

    local gated_h = nn.CMulTable()({gates, h})

    local summed_act = {}
    local idx_offset = 0
    for i, n_nodes in ipairs(n_nodes_list) do
        table.insert(summed_act, nn.Reshape(1,self.class_net_in_dim, false)(nn.Sum(1)(nn.Narrow(1, idx_offset+1, n_nodes)(gated_h))))
        idx_offset = idx_offset + n_nodes
    end

    local output
    if #summed_act > 1 then
        output = nn.Tanh()(nn.JoinTable(1,2)(summed_act))
    else
        output = nn.Tanh()(summed_act[1])
    end

    self.aggregation_net = nn.gModule({input}, {output})
end

-- The forward pass, pass through the propagation net and the output net
-- return the final outputs.
function GraphLevelGGNN:forward(edges_list, n_steps, annotations_list)
    BaseGGNN.forward(self, edges_list, n_steps, annotations_list)

    self:create_aggregation_net(self.n_nodes_list)

    self.aggregation_net_input = torch.Tensor(self.n_total_nodes, self.state_dim + self.annotation_dim)
    self.aggregation_net_input:narrow(2,1,self.state_dim):copy(self.prop_inputs[n_steps+1])
    self.aggregation_net_input:narrow(2,self.state_dim+1,self.annotation_dim):copy(self.prop_inputs[1]:narrow(2,1,self.annotation_dim))

    self.c_input = self.aggregation_net:forward(self.aggregation_net_input)
    return self.output_net:forward(self.c_input)
end

-- A convinience function for making predictions.
function GraphLevelGGNN:predict(edges_list, n_steps, annotations_list)
    local output = self:forward(edges_list, n_steps, annotations_list)
    local _, pred = torch.max(output, 2)
    return pred
end

function GraphLevelGGNN:predict_graph_batch(edges_list, n_steps, annotations_list)
    local output = self:forward(edges_list, n_steps, annotations_list)
    local scores = output:sum(1)
    local _, pred = scores:max(2)
    return pred[1][1]
end

function GraphLevelGGNN:predict_single_graph(edges, n_steps, annotations)
    local pred = self:predict({edges}, n_steps, {annotations})
    return pred[1][1]
end

-- The backward pass.  output_grad is the gradient tensor obtained from the 
-- loss criterion.  Assumes that forward function has already been called on
-- the same input.
function GraphLevelGGNN:backward(output_grad)
    local layer_grad = self.output_net:backward(self.c_input, output_grad)
    layer_grad = self.aggregation_net:backward(self.aggregation_net_input, layer_grad)
    local annotation_grad = torch.Tensor(self.n_total_nodes, self.annotation_dim):copy(layer_grad:narrow(2,self.state_dim+1,self.annotation_dim))
    layer_grad = layer_grad:narrow(2,1,self.state_dim)

    annotation_grad:add(BaseGGNN.backward(self, layer_grad))
    return annotation_grad
end

-- Print out propagation net and classification net structure.
function GraphLevelGGNN:str_repr()
    local s = BaseGGNN.str_repr(self)
    s = s .. ' | Aggregation Net: ' .. self.module_dict[ggnn.AGGREGATION_NET_PREFIX .. '-input'].weight:size(2) .. '-' .. self.class_net_in_dim
    s = s .. ' | Classification Net: ' .. self.module_dict[ggnn.OUTPUT_NET_PREFIX .. '-1'].weight:size(2)
    layer_id = 1
    while self.module_dict[ggnn.OUTPUT_NET_PREFIX .. '-' .. layer_id] do
        s = s .. '-' .. self.module_dict[ggnn.OUTPUT_NET_PREFIX .. '-' .. layer_id].weight:size(1)
        layer_id = layer_id + 1
    end
    return s
end


