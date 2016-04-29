-- Output network for node selection.
--
-- Yujia Li, 03/2016
--

local NodeSelectionOutputNet, BaseOutputNet = torch.class('ggnn.NodeSelectionOutputNet', 'ggnn.BaseOutputNet')

function NodeSelectionOutputNet:__init(state_dim, annotation_dim, output_net_h_sizes, module_dict)
    BaseOutputNet.__init(self, state_dim, annotation_dim, module_dict)

    self.output_net_h_sizes = output_net_h_sizes
    self:create_output_net()
end

-- Creates a copy of this network sharing the same module_dict - i.e. using 
-- exactly the same set of parameters.
function NodeSelectionOutputNet:create_share_param_copy()
    return ggnn.NodeSelectionOutputNet(self.state_dim, self.annotation_dim, self.output_net_h_sizes, self.module_dict)
end

function NodeSelectionOutputNet:get_constructor_param_dict()
    return {
        state_dim=self.state_dim,
        annotation_dim=self.annotation_dim,
        output_net_h_sizes=self.output_net_h_sizes
    }
end

function ggnn.load_node_selection_output_net_from_file(file_name)
    local d = torch.load(file_name)
    local net = ggnn.NodeSelectionOutputNet(
        d['state_dim'],
        d['annotation_dim'],
        d['output_net_h_sizes']
    )
    local w = net:getParameters()
    w:copy(d['params'])

    return net
end

-- This network takes the node representations as input and convert
-- these into a real number score.  Node annotations are used as well for
-- gating.
function NodeSelectionOutputNet:create_output_net()
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

function NodeSelectionOutputNet:str_repr()
    return '<NodeSelectionOutputNet>'
end
