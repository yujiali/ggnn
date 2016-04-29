-- GGNN for making a sequence of graph level predictions. Designed for 
-- bAbI task 19.
--
-- This is a variant where the graph level output and the node annotation 
-- output share the same propagation network.
--
-- Yujia Li, 04/2016
--

local GraphLevelSequenceSharePropagationGGNN = torch.class('ggnn.GraphLevelSequenceSharePropagationGGNN')

-- propagation_net is a BaseGGNN
-- graph_level_output_net is a GraphLevelOutputNet
-- annotation_output_net is a PerNodeOutputNet
function GraphLevelSequenceSharePropagationGGNN:__init(propagation_net, graph_level_output_net, annotation_output_net)
    self.propagation_net = propagation_net
    self.graph_level_output_net = graph_level_output_net
    self.annotation_output_net = annotation_output_net

    assert(propagation_net.annotation_dim == graph_level_output_net.annotation_dim)
    assert(propagation_net.annotation_dim == annotation_output_net.annotation_dim)
    assert(propagation_net.state_dim == graph_level_output_net.state_dim)
    assert(propagation_net.state_dim == annotation_output_net.state_dim)

    self.n_classes = self.graph_level_output_net.n_classes
end

-- n_pred_steps is the number of prediction steps
-- n_prop_steps is the number of propagation steps for each prediction
--
function GraphLevelSequenceSharePropagationGGNN:forward(edges_list, n_pred_steps, n_prop_steps, annotations_list)
    self.n_pred_steps = n_pred_steps
    self.n_prop_steps = n_prop_steps

    if self.slices == nil then
        self.slices = {}
    end

    if #self.slices < n_pred_steps then
        for i=#self.slices+1, n_pred_steps do
            local pnet = self.propagation_net:create_share_param_copy()
            local glnet = self.graph_level_output_net:create_share_param_copy()
            local anet = self.annotation_output_net:create_share_param_copy()
            self.slices[i] = {pnet=pnet, glnet=glnet, anet=anet}
        end
    end

    if self.graph_level_output == nil then
        self.graph_level_output = torch.Tensor()
    end

    local annotations_list_input = annotations_list
    for t=1,n_pred_steps do
        local nodereps = self.slices[t].pnet:forward(edges_list, n_prop_steps, annotations_list_input)  -- node representations
        local nodeanno = self.slices[t].pnet._annotations   -- node annotation matrix
        local n_nodes_list = self.slices[t].pnet.n_nodes_list

        local gl_out = self.slices[t].glnet:forward(nodereps, nodeanno, n_nodes_list)

        if t < n_pred_steps then
            local a_out = self.slices[t].anet:forward(nodereps, nodeanno, n_nodes_list)
            annotations_list_input = a_out
        end

        if t == 1 then
            edges_list = self.slices[t].pnet.a_list    -- get the adjacency matrices to save time
            self.n_graphs = self.slices[t].pnet.n_graphs
            self.graph_level_output:resize(self.n_graphs, self.n_classes * n_pred_steps)
        end
        self.graph_level_output:narrow(2, (t-1) * self.n_classes + 1, self.n_classes):copy(gl_out)
    end

    return self.graph_level_output
end

function GraphLevelSequenceSharePropagationGGNN:predict(edges_list, n_pred_steps, n_prop_steps, annotations_list)
    local output = self:forward(edges_list, n_pred_steps, n_prop_steps, annotations_list)
    output:resize(self.n_graphs * n_pred_steps, self.n_classes)

    local _, idx = output:max(2)
    return idx:resize(self.n_graphs, n_pred_steps)
end

function GraphLevelSequenceSharePropagationGGNN:backward(graph_level_grad)
    assert(graph_level_grad:size(1) == self.n_graphs)
    assert(graph_level_grad:size(2) == self.n_classes * self.n_pred_steps)

    local a_grad, r_grad    -- annotation and node representation gradients
    for t=self.n_pred_steps,1,-1 do
        local gl_grad = graph_level_grad:narrow(2,(t-1) * self.n_classes + 1, self.n_classes)
        local gl_r_grad, gl_a_grad = self.slices[t].glnet:backward(gl_grad)
        if t < self.n_pred_steps then
            local a_r_grad, a_a_grad = self.slices[t].anet:backward(a_grad)
            gl_r_grad:add(a_r_grad)
            gl_a_grad:add(a_a_grad)
        end
        r_grad = gl_r_grad
        a_grad = gl_a_grad
        a_grad:add(self.slices[t].pnet:backward(r_grad))
    end
    return a_grad
end

function GraphLevelSequenceSharePropagationGGNN:getParameters()
    local params, grad_params = self:parameters()
    return nn.Module.flatten(params), nn.Module.flatten(grad_params)
end

function GraphLevelSequenceSharePropagationGGNN:parameters()
    local w, gw = self.propagation_net:parameters()
    local aw, agw = self.graph_level_output_net:parameters()
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

function GraphLevelSequenceSharePropagationGGNN:print_model()
    print('[Propagation Net]')
    self.propagation_net:print_model()
    print('[GraphLevel Output Net]:')
    self.graph_level_output_net:print_model()
    print('[Annotation Output Net]:')
    self.annotation_output_net:print_model()
end

---------- model I/O -----------

function GraphLevelSequenceSharePropagationGGNN:get_constructor_param_dict()
    return {
        state_dim        = self.propagation_net.state_dim,
        annotation_dim   = self.propagation_net.annotation_dim,
        prop_net_h_sizes = self.propagation_net.prop_net_h_sizes,
        n_edge_types     = self.propagation_net.n_edge_types,
        gl_output_net_sizes   = self.graph_level_output_net.output_net_sizes,
        a_output_net_h_sizes  = self.annotation_output_net.output_net_h_sizes
    }
end

function ggnn.load_graph_level_seq_share_prop_ggnn_from_file(model_file)
    local d = torch.load(model_file)
    local propagation_net = ggnn.BaseGGNN(
        d['state_dim'],
        d['annotation_dim'],
        d['prop_net_h_sizes'],
        d['n_edge_types'])
    local graph_level_output_net = ggnn.GraphLevelOutputNet(
        d['state_dim'],
        d['annotation_dim'], 
        d['gl_output_net_sizes'])
    local annotation_output_net = ggnn.PerNodeOutputNet(
        d['state_dim'],
        d['annotation_dim'],
        d['a_output_net_h_sizes'])

    local glseqnet = ggnn.GraphLevelSequenceSharePropagationGGNN(propagation_net, graph_level_output_net, annotation_output_net)
    local w, _ = glseqnet:getParameters()
    w:copy(d['params'])

    return glseqnet
end

