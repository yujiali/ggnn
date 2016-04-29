-- Output net for making per node predictions.
--
-- Yujia Li, 03/2016
--

local PerNodeOutputNet, BaseOutputNet = torch.class('ggnn.PerNodeOutputNet', 'ggnn.BaseOutputNet')

function PerNodeOutputNet:__init(state_dim, annotation_dim, output_net_h_sizes, module_dict, n_outputs)
    BaseOutputNet.__init(self, state_dim, annotation_dim, module_dict)

    self.output_net_h_sizes = output_net_h_sizes
    self.n_outputs = n_outputs or annotation_dim
    self:create_output_net()
end

-- Creates a copy of this network sharing the same module_dict - i.e. using 
-- exactly the same set of parameters.
function PerNodeOutputNet:create_share_param_copy()
    return ggnn.PerNodeOutputNet(self.state_dim, self.annotation_dim, self.output_net_h_sizes, self.module_dict, self.n_outputs)
end

function PerNodeOutputNet:get_constructor_param_dict()
    return {
        state_dim=self.state_dim,
        annotation_dim=self.annotation_dim,
        output_net_h_sizes=self.output_net_h_sizes,
        n_outputs=self.n_outputs
    }
end

function ggnn.load_per_node_output_net_from_file(file_name)
    local d = torch.load(file_name)
    local net = ggnn.PerNodeOutputNet(
        d['state_dim'],
        d['annotation_dim'],
        d['output_net_h_sizes'],
        nil,
        d['n_outputs']
    )
    local w = net:getParameters()
    w:copy(d['params'])

    return net
end


-- This output net takes the final node representations and initial
-- annotations as input and outputs the updated annotations.  Can be easily
-- changed to make other per-node predictions.
function PerNodeOutputNet:create_output_net()
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

function PerNodeOutputNet:str_repr()
    return '<PerNodeOutputNet>'
end
