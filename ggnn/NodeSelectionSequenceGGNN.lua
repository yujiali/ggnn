-- GNN for sequences of nodes problems, designed for the two graph path
-- problems.
--
-- Yujia Li, 11/2015
--

local NodeSelectionSequenceGGNN = torch.class('ggnn.NodeSelectionSequenceGGNN')

function NodeSelectionSequenceGGNN:__init(node_selection_net, annotation_net)
    self.node_selection_net = node_selection_net
    self.annotation_net = annotation_net

    assert(node_selection_net.annotation_dim == annotation_net.annotation_dim)
    assert(node_selection_net.n_edge_types == annotation_net.n_edge_types)
end

-- n_pred_steps is the number of prediction steps
-- n_prop_steps is the number of propagation steps for each prediction
--
function NodeSelectionSequenceGGNN:forward(edges_list, n_pred_steps, n_prop_steps, annotations_list)
    self.n_pred_steps = n_pred_steps
    self.n_prop_steps = n_prop_steps

    if self.slices == nil then
        self.slices = {}
    end

    if #self.slices < n_pred_steps then
        for i=#self.slices+1, n_pred_steps do
            local nsnet = self.node_selection_net:create_share_param_copy()
            local anet = self.annotation_net:create_share_param_copy()
            self.slices[i] = {nsnet=nsnet, anet=anet}
        end
    end

    self.node_selection_output = {}

    local annotations_list_input = annotations_list
    for t=1,n_pred_steps do
        local ns_out = self.slices[t].nsnet:forward(edges_list, n_prop_steps, annotations_list_input)
        local a_out = self.slices[t].anet:forward(edges_list, n_prop_steps, annotations_list_input)
        -- annotations_list_input = self.slices[t].anet:breakdown_annotations(a_out)
        annotations_list_input = a_out

        if t == 1 then
            edges_list = self.slices[t].nsnet.a_list    -- get the adjacency matrices to save time
            self.n_graphs = self.slices[t].nsnet.n_graphs
            self.n_nodes_list = self.slices[t].nsnet.n_nodes_list   -- keep this for reference
        end
        self.node_selection_output[t] = ns_out
    end

    return self.node_selection_output
end

function NodeSelectionSequenceGGNN:predict(edges_list, n_pred_steps, n_prop_steps, annotations_list)
    local output = self:forward(edges_list, n_pred_steps, n_prop_steps, annotations_list)
    local idx = {}
    for t=1,n_pred_steps do
        idx[t] = ggnn.compute_node_selection_predictions(output[t], self.slices[t].nsnet.n_nodes_list)
        idx[t]:resize(idx[t]:nElement(), 1)
    end
    if self._jt == nil then
        self._jt = nn.JoinTable(2)
    end
    idx = self._jt:forward(idx)

    return idx
end

function NodeSelectionSequenceGGNN:backward(node_select_grad)
    assert(#node_select_grad == self.n_pred_steps)

    local a_grad
    for t=self.n_pred_steps,1,-1 do
        local a_ns_grad = self.slices[t].nsnet:backward(node_select_grad[t])
        if t < self.n_pred_steps then
            a_grad = self.slices[t].anet:backward(a_grad)
            a_grad:add(a_ns_grad)
        else
            a_grad = a_ns_grad
        end
    end
    return a_grad
end

function NodeSelectionSequenceGGNN:getParameters()
    local params, grad_params = self:parameters()
    return nn.Module.flatten(params), nn.Module.flatten(grad_params)
end

function NodeSelectionSequenceGGNN:parameters()
    local w, gw = self.node_selection_net:parameters()
    local aw, agw = self.annotation_net:parameters()
    for i=1,#aw do
        table.insert(w, aw[i])
        table.insert(gw, agw[i])
    end
    return w, gw
end

function NodeSelectionSequenceGGNN:print_model()
    print('[NodeSelectionGGNN]:')
    self.node_selection_net:print_model()
    print('[AnnotationGGNN]:')
    self.annotation_net:print_model()
end

---------- model I/O -----------

function NodeSelectionSequenceGGNN:get_constructor_param_dict()
    return {
        annotation_dim  = self.node_selection_net.annotation_dim,
        n_edge_types    = self.node_selection_net.n_edge_types,
        ns_state_dim           = self.node_selection_net.state_dim,
        ns_prop_net_h_sizes    = self.node_selection_net.prop_net_h_sizes,
        ns_output_net_h_sizes  = self.node_selection_net.output_net_h_sizes,
        a_state_dim            = self.annotation_net.state_dim,
        a_prop_net_h_sizes     = self.annotation_net.prop_net_h_sizes,
        a_output_net_h_sizes   = self.annotation_net.output_net_h_sizes
    }
end

-- 
function ggnn.load_node_selection_seq_ggnn_from_file(model_file)
    local d = torch.load(model_file)
    local node_selection_net = ggnn.NodeSelectionGGNN(
        d['ns_state_dim'],
        d['annotation_dim'], 
        d['ns_prop_net_h_sizes'],
        d['ns_output_net_h_sizes'],
        d['n_edge_types'])
    local annotation_net = ggnn.PerNodeGGNN(
        d['a_state_dim'],
        d['annotation_dim'],
        d['a_prop_net_h_sizes'], 
        d['a_output_net_h_sizes'],
        d['n_edge_types'])

    local nsseqnet = ggnn.NodeSelectionSequenceGGNN(node_selection_net, annotation_net)
    local w, _ = nsseqnet:getParameters()
    w:copy(d['params'])

    return nsseqnet
end

----------------- loss ------------------

-- outputs is a list of n_pred_steps elements, each is the output of one step.
-- targets is a list of n_graphs targets, each is one output sequence, all of
-- them must have the same length, or a n_graphs x n_pred_steps tensor.
function ggnn.compute_node_selection_seq_ggnn_loss_and_grad(criterion, outputs, targets, n_nodes_list, compute_grad)
    if compute_grad == nil then
        compute_grad = true
    end

    if type(targets) == 'table' then
        targets = torch.Tensor(targets)
    end

    assert(targets:size(2) == #outputs)

    local n_pred_steps = targets:size(2)

    local loss = 0
    local output_grad = {}

    for t=1,n_pred_steps do
        local t_loss, t_grad = ggnn.compute_node_selection_loss_and_grad(criterion, outputs[t], targets:select(2,t), n_nodes_list, compute_grad)
        loss = loss + t_loss
        if compute_grad then
            output_grad[t] = t_grad:mul(1 / n_pred_steps)
        end
    end

    if compute_grad then
        return loss / n_pred_steps, output_grad
    else
        return loss / n_pred_steps
    end
end

