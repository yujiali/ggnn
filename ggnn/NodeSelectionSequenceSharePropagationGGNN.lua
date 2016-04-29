-- GNN for sequences of nodes problems, designed for the two graph path
-- problems.
--
-- This is a variant where the graph level output and the node annotation 
-- output share the same propagation network.
--
-- Yujia Li, 04/2016
--

local NodeSelectionSequenceSharePropagationGGNN = torch.class('ggnn.NodeSelectionSequenceSharePropagationGGNN')

function NodeSelectionSequenceSharePropagationGGNN:__init(propagation_net, node_selection_output_net, annotation_output_net)
    self.propagation_net = propagation_net
    self.node_selection_output_net = node_selection_output_net
    self.annotation_output_net = annotation_output_net

    assert(propagation_net.annotation_dim == node_selection_output_net.annotation_dim)
    assert(propagation_net.annotation_dim == annotation_output_net.annotation_dim)
    assert(propagation_net.state_dim == node_selection_output_net.state_dim)
    assert(propagation_net.state_dim == annotation_output_net.state_dim)
end

-- n_pred_steps is the number of prediction steps
-- n_prop_steps is the number of propagation steps for each prediction
--
function NodeSelectionSequenceSharePropagationGGNN:forward(edges_list, n_pred_steps, n_prop_steps, annotations_list)
    self.n_pred_steps = n_pred_steps
    self.n_prop_steps = n_prop_steps

    if self.slices == nil then
        self.slices = {}
    end

    if #self.slices < n_pred_steps then
        for i=#self.slices+1, n_pred_steps do
            local pnet = self.propagation_net:create_share_param_copy()
            local nsnet = self.node_selection_output_net:create_share_param_copy()
            local anet = self.annotation_output_net:create_share_param_copy()
            self.slices[i] = {pnet=pnet, nsnet=nsnet, anet=anet}
        end
    end

    self.node_selection_output = {}

    local annotations_list_input = annotations_list
    for t=1,n_pred_steps do
        local nodereps = self.slices[t].pnet:forward(edges_list, n_prop_steps, annotations_list_input)  -- node representations
        local nodeanno = self.slices[t].pnet._annotations   -- node annotation matrix
        local n_nodes_list = self.slices[t].pnet.n_nodes_list

        local ns_out = self.slices[t].nsnet:forward(nodereps, nodeanno, n_nodes_list)
        if t < n_pred_steps then
            local a_out = self.slices[t].anet:forward(nodereps, nodeanno, n_nodes_list)
            annotations_list_input = a_out
        end

        if t == 1 then
            edges_list = self.slices[t].pnet.a_list    -- get the adjacency matrices to save time
            self.n_graphs = self.slices[t].pnet.n_graphs
            self.n_nodes_list = self.slices[t].pnet.n_nodes_list   -- keep this for reference
        end
        self.node_selection_output[t] = ns_out
    end

    return self.node_selection_output
end

function NodeSelectionSequenceSharePropagationGGNN:predict(edges_list, n_pred_steps, n_prop_steps, annotations_list)
    local output = self:forward(edges_list, n_pred_steps, n_prop_steps, annotations_list)
    local idx = {}
    for t=1,n_pred_steps do
        idx[t] = ggnn.compute_node_selection_predictions(output[t], self.slices[t].pnet.n_nodes_list)
        idx[t]:resize(idx[t]:nElement(), 1)
    end
    if self._jt == nil then
        self._jt = nn.JoinTable(2)
    end
    idx = self._jt:forward(idx)

    return idx
end

function NodeSelectionSequenceSharePropagationGGNN:backward(node_select_grad)
    assert(#node_select_grad == self.n_pred_steps)

    local a_grad, r_grad    -- annotation and node representation gradients
    for t=self.n_pred_steps,1,-1 do
        local ns_r_grad, ns_a_grad = self.slices[t].nsnet:backward(node_select_grad[t])
        if t < self.n_pred_steps then
            local a_r_grad, a_a_grad = self.slices[t].anet:backward(a_grad)
            ns_r_grad:add(a_r_grad)
            ns_a_grad:add(a_a_grad)
        end
        r_grad = ns_r_grad
        a_grad = ns_a_grad
        a_grad:add(self.slices[t].pnet:backward(r_grad))
    end
    return a_grad
end

function NodeSelectionSequenceSharePropagationGGNN:getParameters()
    local params, grad_params = self:parameters()
    return nn.Module.flatten(params), nn.Module.flatten(grad_params)
end

function NodeSelectionSequenceSharePropagationGGNN:parameters()
    local w, gw = self.propagation_net:parameters()
    local aw, agw = self.node_selection_output_net:parameters()
    for i=1,#aw do
        table.insert(w, aw[i])
        table.insert(gw, agw[i])
    end
    aw, agw = self.annotation_output_net:parameters()
    for i=1,#aw do
        table.insert(w, aw[i])
        table.insert(gw, agw[i])
    end
    return w, gw
end

function NodeSelectionSequenceSharePropagationGGNN:print_model()
    print('[Propagation Net]:')
    self.propagation_net:print_model()
    print('[NodeSelection Output Net]:')
    self.node_selection_output_net:print_model()
    print('[Annotation Output Net]:')
    self.annotation_output_net:print_model()
end

---------- model I/O -----------

function NodeSelectionSequenceSharePropagationGGNN:get_constructor_param_dict()
    return {
        state_dim        = self.propagation_net.state_dim,
        annotation_dim   = self.propagation_net.annotation_dim,
        prop_net_h_sizes = self.propagation_net.prop_net_h_sizes,
        n_edge_types     = self.propagation_net.n_edge_types,
        ns_output_net_h_sizes  = self.node_selection_output_net.output_net_h_sizes,
        a_output_net_h_sizes   = self.annotation_output_net.output_net_h_sizes
    }
end

-- 
function ggnn.load_node_selection_seq_share_prop_ggnn_from_file(model_file)
    local d = torch.load(model_file)
    local propagation_net = ggnn.BaseGGNN(
        d['state_dim'],
        d['annotation_dim'],
        d['prop_net_h_sizes'],
        d['n_edge_types'])
    local node_selection_output_net = ggnn.NodeSelectionOutputNet(
        d['state_dim'],
        d['annotation_dim'], 
        d['ns_output_net_h_sizes'])
    local annotation_output_net = ggnn.PerNodeOutputNet(
        d['state_dim'],
        d['annotation_dim'],
        d['a_output_net_h_sizes'])

    local nsseqnet = ggnn.NodeSelectionSequenceSharePropagationGGNN(propagation_net, node_selection_output_net, annotation_output_net)
    local w, _ = nsseqnet:getParameters()
    w:copy(d['params'])

    return nsseqnet
end


