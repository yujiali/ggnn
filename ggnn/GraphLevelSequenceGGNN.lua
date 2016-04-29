-- GGNN for making a sequence of graph level predictions. Designed for 
-- bAbI task 19.
--
-- Yujia Li, 11/2015
--

local GraphLevelSequenceGGNN = torch.class('ggnn.GraphLevelSequenceGGNN')

-- graph_level_net is a GraphLevelGGNN
-- annotation_net is a PerNodeGGNN
function GraphLevelSequenceGGNN:__init(graph_level_net, annotation_net)
    self.graph_level_net = graph_level_net
    self.annotation_net = annotation_net

    assert(graph_level_net.annotation_dim == annotation_net.annotation_dim)
    assert(graph_level_net.n_edge_types == annotation_net.n_edge_types)

    self.n_classes = self.graph_level_net.n_classes
end

-- n_pred_steps is the number of prediction steps
-- n_prop_steps is the number of propagation steps for each prediction
--
function GraphLevelSequenceGGNN:forward(edges_list, n_pred_steps, n_prop_steps, annotations_list)
    self.n_pred_steps = n_pred_steps
    self.n_prop_steps = n_prop_steps

    if self.slices == nil then
        self.slices = {}
    end

    if #self.slices < n_pred_steps then
        for i=#self.slices+1, n_pred_steps do
            local glnet = self.graph_level_net:create_share_param_copy()
            local anet = self.annotation_net:create_share_param_copy()
            self.slices[i] = {glnet=glnet, anet=anet}
        end
    end

    if self.graph_level_output == nil then
        self.graph_level_output = torch.Tensor()
    end

    local annotations_list_input = annotations_list
    for t=1,n_pred_steps do
        local gl_out = self.slices[t].glnet:forward(edges_list, n_prop_steps, annotations_list_input)
        local a_out = self.slices[t].anet:forward(edges_list, n_prop_steps, annotations_list_input)
        -- annotations_list_input = self.slices[t].anet:breakdown_annotations(a_out)

        annotations_list_input = a_out

        if t == 1 then
            edges_list = self.slices[t].glnet.a_list    -- get the adjacency matrices to save time
            self.n_graphs = self.slices[t].glnet.n_graphs
            self.graph_level_output:resize(self.n_graphs, self.n_classes * n_pred_steps)
        end
        self.graph_level_output:narrow(2, (t-1) * self.n_classes + 1, self.n_classes):copy(gl_out)
    end

    return self.graph_level_output
end

function GraphLevelSequenceGGNN:predict(edges_list, n_pred_steps, n_prop_steps, annotations_list)
    local output = self:forward(edges_list, n_pred_steps, n_prop_steps, annotations_list)
    output:resize(self.n_graphs * n_pred_steps, self.n_classes)

    local _, idx = output:max(2)
    return idx:resize(self.n_graphs, n_pred_steps)
end

function GraphLevelSequenceGGNN:backward(graph_level_grad)
    assert(graph_level_grad:size(1) == self.n_graphs)
    assert(graph_level_grad:size(2) == self.n_classes * self.n_pred_steps)

    local a_grad
    for t=self.n_pred_steps,1,-1 do
        local gl_grad = graph_level_grad:narrow(2,(t-1) * self.n_classes + 1, self.n_classes)
        local a_gl_grad = self.slices[t].glnet:backward(gl_grad)
        if t < self.n_pred_steps then
            a_grad = self.slices[t].anet:backward(a_grad)
            a_grad:add(a_gl_grad)
        else
            a_grad = a_gl_grad
        end
    end
    return a_grad
end

function GraphLevelSequenceGGNN:getParameters()
    local params, grad_params = self:parameters()
    return nn.Module.flatten(params), nn.Module.flatten(grad_params)
end

function GraphLevelSequenceGGNN:parameters()
    local w, gw = self.graph_level_net:parameters()
    local aw, agw = self.annotation_net:parameters()
    for i=1,#aw do
        table.insert(w, aw[i])
        table.insert(gw, agw[i])
    end
    return w, gw
end

function GraphLevelSequenceGGNN:print_model()
    print('[GraphLevelGGNN]:')
    self.graph_level_net:print_model()
    print('[AnnotationGGNN]:')
    self.annotation_net:print_model()
end

---------- model I/O -----------

function GraphLevelSequenceGGNN:get_constructor_param_dict()
    return {
        annotation_dim  = self.graph_level_net.annotation_dim,
        n_edge_types    = self.graph_level_net.n_edge_types,
        gl_state_dim          = self.graph_level_net.state_dim,
        gl_prop_net_h_sizes   = self.graph_level_net.prop_net_h_sizes,
        gl_output_net_sizes   = self.graph_level_net.output_net_sizes,
        a_state_dim           = self.annotation_net.state_dim,
        a_prop_net_h_sizes    = self.annotation_net.prop_net_h_sizes,
        a_output_net_h_sizes  = self.annotation_net.output_net_h_sizes
    }
end

function ggnn.load_graph_level_seq_ggnn_from_file(model_file)
    local d = torch.load(model_file)
    local graph_level_net = ggnn.GraphLevelGGNN(
        d['gl_state_dim'],
        d['annotation_dim'], 
        d['gl_prop_net_h_sizes'],
        d['gl_output_net_sizes'],
        d['n_edge_types'])
    local annotation_net = ggnn.PerNodeGGNN(
        d['a_state_dim'],
        d['annotation_dim'],
        d['a_prop_net_h_sizes'], 
        d['a_output_net_h_sizes'],
        d['n_edge_types'])

    local glseqnet = ggnn.GraphLevelSequenceGGNN(graph_level_net, annotation_net)
    local w, _ = glseqnet:getParameters()
    w:copy(d['params'])

    return glseqnet
end

----------------- loss ------------------

-- outputs is a n_graphs x (n_classes * n_pred_steps) tensor
-- targets is a n_graphs x n_pred_steps tensor, or a n_graphs x n_pred_steps table.
function ggnn.compute_graph_level_seq_ggnn_loss_and_grad(criterion, outputs, targets, compute_grad)
    if compute_grad == nil then
        compute_grad = true
    end

    if type(targets) == 'table' then
        targets = torch.Tensor(targets)
    end

    assert(targets:size(1) == outputs:size(1))
    assert(outputs:size(2) % targets:size(2) == 0)
    local n_pred_steps = targets:size(2)
    local n_graphs = targets:size(1)
    local n_classes = outputs:size(2) / targets:size(2)

    targets = torch.reshape(targets, n_graphs * n_pred_steps)
    outputs = torch.reshape(outputs, n_graphs * n_pred_steps, n_classes)

    local loss = criterion:forward(outputs, targets)
    if not compute_grad then
        return loss
    else
        local output_grad = criterion:backward(outputs, targets)
        output_grad = torch.reshape(output_grad, n_graphs, n_pred_steps * n_classes)
        return loss, output_grad
    end
end

