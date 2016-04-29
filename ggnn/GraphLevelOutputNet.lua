-- Output network for making graph level predictions.
--
-- Yujia Li, 03/2016
--

local GraphLevelOutputNet, BaseOutputNet = torch.class('ggnn.GraphLevelOutputNet', 'ggnn.BaseOutputNet')

-- output_net_sizes: number of units on each layer of the classification net,
--      the first number in the list is the input size, and the last number is 
--      the output size.
function GraphLevelOutputNet:__init(state_dim, annotation_dim, output_net_sizes, module_dict)
    BaseOutputNet.__init(self, state_dim, annotation_dim, module_dict)

    self.output_net_sizes = output_net_sizes
    self.n_classes = output_net_sizes[#output_net_sizes]
    self.class_net_in_dim = output_net_sizes[1]

    self:create_aggregation_net_modules()
    self:create_output_net()
end

-- Creates a copy of this network sharing the same module_dict - i.e. using 
-- exactly the same set of parameters.
function GraphLevelOutputNet:create_share_param_copy()
    return ggnn.GraphLevelOutputNet(self.state_dim, self.annotation_dim, self.output_net_sizes, self.module_dict)
end

function GraphLevelOutputNet:get_constructor_param_dict()
    return {
        state_dim=self.state_dim,
        annotation_dim=self.annotation_dim,
        output_net_sizes=self.output_net_sizes
    }
end

function ggnn.load_graph_level_output_net_from_file(file_name)
    local d = torch.load(file_name)
    local net = ggnn.GraphLevelOutputNet(
        d['state_dim'],
        d['annotation_dim'],
        d['output_net_sizes']
    )
    local w = net:getParameters()
    w:copy(d['params'])

    return net
end

-- Create all shared modules of the aggregation net.  Actually just two modules
-- the linear module linked to the gates and the linear module to the
-- transformed outputs. Here to be more efficient we construct the two linear
-- modules as one and then use Narrow modules to get separate modules out of
-- the results.
function GraphLevelOutputNet:create_aggregation_net_modules()
    ggnn.create_or_share('Linear', ggnn.AGGREGATION_NET_PREFIX .. '-input', self.module_dict, {self.state_dim+self.annotation_dim, 2*self.class_net_in_dim})
end

-- Output net does not change across graphs.
function GraphLevelOutputNet:create_output_net()
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

-- The input is a concatenation of the final node representations and the initial
-- node annotations.  Then both the gates and the transformed representations
-- get input from the concatenated input.
--
-- Aggregation net creates graph level representations for each graph, these 
-- networks need to be dynamically created for each graph as different graphs
-- have different sizes.
function GraphLevelOutputNet:create_aggregation_net(n_nodes_list)
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

function GraphLevelOutputNet:forward(node_representations, node_annotations, n_nodes_list)
    self:create_aggregation_net(n_nodes_list)
    self.aggregation_net_input = self:prepare_input(node_representations, node_annotations)
    self.graph_representations = self.aggregation_net:forward(self.aggregation_net_input)
    return self.output_net:forward(self.graph_representations)
end

function GraphLevelOutputNet:backward(output_grad)
    local graph_rep_grad = self.output_net:backward(self.graph_representations, output_grad)
    local input_grad = self.aggregation_net:backward(self.aggregation_net_input, graph_rep_grad)
    return self:separate_gradients(input_grad)
end

function GraphLevelOutputNet:str_repr()
    return '<GraphLevelOutputNet>'
end
