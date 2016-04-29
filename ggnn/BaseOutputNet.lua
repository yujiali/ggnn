-- Base for all output networks.
--
-- An output network takes the final output of a BaseGGNN (propagation network)
-- and the initial annotations as input and predicts outputs.
--
-- Yujia Li, 03/2016
--

require 'nn'

local BaseOutputNet = torch.class('ggnn.BaseOutputNet')

function BaseOutputNet:__init(state_dim, annotation_dim, module_dict)
    self.state_dim = state_dim
    self.annotation_dim = annotation_dim
    self.module_dict = module_dict or {}
end

-- Creates a copy of this network sharing the same module_dict - i.e. using 
-- exactly the same set of parameters.
function BaseOutputNet:create_share_param_copy()
    return ggnn.BaseOutputNet(self.state_dim, self.annotation_dim, self.module_dict)
end

function BaseOutputNet:get_constructor_param_dict()
    return {
        state_dim=self.state_dim,
        annotation_dim=self.annotation_dim
    }
end

function ggnn.load_base_output_net_from_file(file_name)
    local d = torch.load(file_name)
    local net = ggnn.BaseOutputNet(
        d['state_dim'],
        d['annotation_dim']
    )
    local w = net:getParameters()
    w:copy(d['params'])

    return net
end

-- Return a list of parameters and a list of parameter gradients. This function
-- is inspired by the parameters function in nn/Container.lua
function BaseOutputNet:parameters()
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

function BaseOutputNet:getParameters()
    local params, grad_params = self:parameters()
    return nn.Module.flatten(params), nn.Module.flatten(grad_params)
end

-- Create the output network, which takes the concatenation of final node
-- representations and annotations as input and map that to outputs.
function BaseOutputNet:create_output_net()
    -- Subclasses should overwrite this function.
end

function BaseOutputNet:prepare_input(node_representations, node_annotations)
    assert(node_representations:size(2) == self.state_dim, 'node representation dimensionality mismatch')
    assert(node_annotations:size(2) == self.annotation_dim, 'node annotation dimensionality mismatch')
    assert(node_representations:size(1) == node_annotations:size(1), 'node representations and annotations should have the same number of nodes')

    self.n_total_nodes = node_representations:size(1)

    if self.output_net_input == nil then
        self.output_net_input = torch.Tensor():type(node_representations:type())
    end
    self.output_net_input:resize(self.n_total_nodes, self.state_dim + self.annotation_dim)
    self.output_net_input:narrow(2,1,self.state_dim):copy(node_representations)
    self.output_net_input:narrow(2,self.state_dim+1,self.annotation_dim):copy(node_annotations)

    return self.output_net_input
end

-- node_representations: NxD node representations matrix
-- node_annotations: NxA node annotations matrix
-- n_nodes_list: a list of n_nodes for each graph - this is useful for some
--      outputs that rely on some graph structure properties.
-- 
-- This function implements the default behavior, can be overwritten.
function BaseOutputNet:forward(node_representations, node_annotations, n_nodes_list)
    self:prepare_input(node_representations, node_annotations)
    self.output = self.output_net:forward(self.output_net_input)
    return self.output
end

-- output_net_input_grad: Nx(D+A) gradient matrix, first NxD part is the 
-- gradient for node representations and the last NxA part is the gradient
-- for node annoatations.
--
-- This function splits the two and returns two matrices of size NxD and NxA
-- for the two parts separately.
function BaseOutputNet:separate_gradients(output_net_input_grad)
    return output_net_input_grad:narrow(2,1,self.state_dim), output_net_input_grad:narrow(2,self.state_dim+1,self.annotation_dim)
end

-- default behavior, can be overwritten
function BaseOutputNet:backward(output_grad)
    return self:separate_gradients(self.output_net:backward(self.output_net_input, output_grad))
end

function BaseOutputNet:str_repr()
    return '<BaseOutputNet>'
end

function BaseOutputNet:print_model()
    print(self:str_repr())
end
