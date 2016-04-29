-- This base class handles some basic elements shared by all RNN types.
--
-- Yujia Li, 10/2015
--

require 'nn'

local BaseRNN = torch.class('rnn.BaseRNN')

function BaseRNN:__init(module_dict)
    self.module_dict = module_dict or {}
end

function BaseRNN:parameters()
    local w = {}
    local gw = {}

    -- sort the keys to make sure the parameters are always in the same order
    local k_list = {}
    for k, v in pairs(self.module_dict) do
        table.insert(k_list, k)
    end
    table.sort(k_list)

    for i=1,#k_list do
        m = self.module_dict[k_list[i]]
        local mw, mgw = m:parameters()
        if mw then
            if type(mw) == 'table' then
                for i=1,#mw do
                    table.insert(w, mw[i])
                    table.insert(gw, mgw[i])
                end
            else
                table.insert(w, mw)
                table.insert(gw, mgw)
            end
        end
    end
    return w, gw
end

function BaseRNN:getParameters()
    local params, grad_params = self:parameters()
    return nn.Module.flatten(params), nn.Module.flatten(grad_params)
end

function BaseRNN:print_model()
    print('BaseRNN')
end
