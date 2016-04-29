-- Gated Graph Neural Network Testing
--
-- Yujia Li, 02/2016
--

require 'ggnn'
require 'rnn'
require 'nn'

color = require 'trepl.colorize'

TOLERANCE = 1e-5
EPS = 1e-8

---------------------------- utils ---------------------------------

function print_test_name(test_name)
    print(color.blue(test_name))
end

function print_vector(v, v_name, precision)
    precision = precision or 8
    io.write(color.yellow(v_name) .. ': [')
    for i=1,v:nElement() do
        io.write(string.format('%' .. (precision + 4) .. '.' .. precision .. 'f', v[i]))
    end
    io.write(' ]\n')
end

function check_grad(g1, g1_name, g2, g2_name)
    local max_name_len = string.len(g1_name)
    if max_name_len < string.len(g2_name) then max_name_len = string.len(g2_name) end

    local err = torch.abs(g1 - g2)
    local err_name = 'Error'
    if max_name_len < string.len(err_name) then max_name_len = string.len(err_name) end

    print_vector(g1, string.format('%' .. max_name_len .. 's', g1_name))
    print_vector(g2, string.format('%' .. max_name_len .. 's', g2_name))
    print_vector(err, string.format('%' .. max_name_len .. 's', err_name))

    if err:nElement() == 0 or err:max() < TOLERANCE then
        print('\27[42m[Success]\27[0m')
        return true
    else
        print('\27[41m[Fail]\27[0m')
        return false
    end
end

-- f is a function that takes a vector w as input and output a single number
-- This function computes the finite difference gradient of f at w.
function compute_fdiff_grad(w, f)
    local grad = torch.Tensor(w:size()):zero()
    local n_dims = w:nElement()
    for i=1,n_dims do
        w[i] = w[i] + EPS
        local f_plus = f(w)
        w[i] = w[i] - 2 * EPS
        local f_minus = f(w)
        grad[i] = (f_plus - f_minus) / (2 * EPS)
        w[i] = w[i] + EPS
    end
    return grad
end

function time_func(f, ...)
    local timer = torch.Timer()
    f(...)
    print(string.format('Time: %.2fs', timer:time().real))
    print('')
end

--------------------- test cases ------------------------

function test_print_vector()
    print_test_name('[Test Print Vector]')
    local v = torch.range(1,10)
    print_vector(v, 'Test Vector', 8)
end

function test_checkgrad()
    print_test_name('[Test checkgrad]')
    local f = function(w) return w[1]^2 + w[2]^2 end
    local w = torch.Tensor({1,2})
    local dw = torch.Tensor({2,4})
    local fdiff_dw = compute_fdiff_grad(w, f)

    check_grad(dw, 'Backprop Grad', fdiff_dw, 'Fdiff Grad')
end

function create_sample_graph()
    local edges = {{{1,1,2}, {2,1,3}, {1,2,3}}, {{1,1,2}, {1,2,2}, {2,2,1}}}
    local annotations = {{{0,1}, {1,1}, {1,0}}, {{0,0}, {1,0}}}
    local n_edge_types = 2
    local n_total_nodes = #annotations[1] + #annotations[2]

    return edges, annotations, n_edge_types, n_total_nodes
end

function create_random_annotations_from_input_annotations(annotations, n_outputs)
    n_outputs = n_outputs or #annotations[1][1]
    local new_annotations = {}
    for i, a in ipairs(annotations) do
        new_annotations[i] = torch.rand(#a, n_outputs):ge(0.5):totable()
    end
    return new_annotations
end

function create_random_per_node_outputs_from_n_nodes_list(n_nodes_list, n_outputs)
    local outputs = {}
    for i, n_nodes in ipairs(n_nodes_list) do
        outputs[i] = torch.rand(n_nodes, n_outputs):ge(0.5):totable()
    end
    return outputs
end

function create_random_node_selections(annotations)
    local n_graphs = #annotations
    local target = torch.Tensor(n_graphs)

    for i=1,n_graphs do
        target[i] = math.ceil(math.random() * #annotations[i])
    end
    return target
end

function create_random_node_selections_from_n_nodes_list(n_nodes_list)
    local n_graphs = #n_nodes_list
    local target = torch.Tensor(n_graphs)

    for i=1,n_graphs do
        target[i] = math.ceil(math.random() * n_nodes_list[i])
    end
    return target
end

function test_base_ggnn()
    print_test_name('[Test BaseGGNN]')
    local edges, annotations, n_edge_types, n_total_nodes = create_sample_graph()
    local n_steps = 3
    local annotation_dim = #annotations[1][1]
    local state_dim = 3

    local target = torch.randn(n_total_nodes, state_dim)
    local c = nn.MSECriterion()

    local net = ggnn.BaseGGNN(state_dim, annotation_dim, {}, n_edge_types)
    local w, dw = net:getParameters()
    dw:zero()

    local y = net:forward(edges, n_steps, annotations)
    c:forward(y, target)
    local dy = c:backward(y, target)
    net:backward(dy)

    local function f(params) 
        if params ~= w then
            params:copy(w)
        end
        return c:forward(net:forward(edges, n_steps, annotations), target)
    end

    local fdiff_dw = compute_fdiff_grad(w, f)

    check_grad(dw, 'Backprop Grad', fdiff_dw, 'Fdiff Grad')
end

function test_base_ggnn_alternative()
    print_test_name('[Test BaseGGNN Alternative]')
    local edges, annotations, n_edge_types, n_total_nodes = create_sample_graph()
    local n_steps = 3
    local annotation_dim = #annotations[1][1]
    local state_dim = 3

    local target = torch.randn(n_total_nodes, state_dim)
    local c = nn.MSECriterion()

    local adjacency_matrices = {}
    local annotations_tensors = {}
    for i, edge_list in ipairs(edges) do
        adjacency_matrices[i] = ggnn.create_adjacency_matrix_cat(edge_list, #annotations[i], n_edge_types)
        annotations_tensors[i] = torch.Tensor(annotations[i])
    end

    local net = ggnn.BaseGGNN(state_dim, annotation_dim, {}, n_edge_types)
    local w, dw = net:getParameters()
    dw:zero()

    local y = net:forward(adjacency_matrices, n_steps, annotations_tensors)
    c:forward(y, target)
    local dy = c:backward(y, target)
    net:backward(dy)

    local function f(params) 
        if params ~= w then
            params:copy(w)
        end
        return c:forward(net:forward(adjacency_matrices, n_steps, annotations_tensors), target)
    end

    local fdiff_dw = compute_fdiff_grad(w, f)

    check_grad(dw, 'Backprop Grad', fdiff_dw, 'Fdiff Grad')
end

function test_base_ggnn_alternative2()
    print_test_name('[Test BaseGGNN Alternative #2]')
    local edges, annotations, n_edge_types, n_total_nodes = create_sample_graph()
    local n_steps = 3
    local annotation_dim = #annotations[1][1]
    local state_dim = 3

    local target = torch.randn(n_total_nodes, state_dim)
    local c = nn.MSECriterion()

    local adjacency_matrices = {}
    local annotations_tensors = {}
    for i, edge_list in ipairs(edges) do
        adjacency_matrices[i] = ggnn.create_adjacency_matrix_cat(edge_list, #annotations[i], n_edge_types)
        annotations_tensors[i] = torch.Tensor(annotations[i])
    end

    local annotations = nn.JoinTable(1):forward(annotations_tensors)

    local net = ggnn.BaseGGNN(state_dim, annotation_dim, {}, n_edge_types)
    local w, dw = net:getParameters()
    dw:zero()

    local y = net:forward(adjacency_matrices, n_steps, annotations)
    c:forward(y, target)
    local dy = c:backward(y, target)
    net:backward(dy)

    local function f(params) 
        if params ~= w then
            params:copy(w)
        end
        return c:forward(net:forward(adjacency_matrices, n_steps, annotations), target)
    end

    local fdiff_dw = compute_fdiff_grad(w, f)

    check_grad(dw, 'Backprop Grad', fdiff_dw, 'Fdiff Grad')
end

function test_per_node_annotation_loss()
    print_test_name('[Test PerNodeGGNN annotation loss]')
    local annotation_dim = 2
    local target = {torch.rand(3,annotation_dim):ge(0.5):totable(), torch.rand(2, annotation_dim):ge(0.5):totable()}
    local n_total_nodes = #target[1] + #target[2]
    local w = nn.Sigmoid():forward(torch.randn(n_total_nodes * annotation_dim))
    local x = w:view(n_total_nodes, annotation_dim)

    local c = nn.BCECriterion()

    local loss, dx = ggnn.compute_annotation_ggnn_loss_and_grad(c, x, target, true)
    local dw = dx:view(-1)
    
    local function f(params)
        if params ~= w then
            params:copy(w)
        end
        return ggnn.compute_annotation_ggnn_loss_and_grad(c, x, target, false)
    end

    local fdiff_dw = compute_fdiff_grad(w, f)
    check_grad(dw, 'Backprop Grad', fdiff_dw, 'Fdiff Grad')
end

function test_per_node_annotation_ggnn()
    print_test_name('[Test PerNodeGGNN]')
    local edges, annotations, n_edge_types, n_total_nodes = create_sample_graph()
    local n_steps = 3
    local annotation_dim = #annotations[1][1]
    local state_dim = 3

    local target = create_random_annotations_from_input_annotations(annotations)
    local c = nn.BCECriterion()

    local net = ggnn.PerNodeGGNN(state_dim, annotation_dim, {}, {}, n_edge_types)
    local w, dw = net:getParameters()
    dw:zero()

    local y = net:forward(edges, n_steps, annotations)
    local loss, dy = ggnn.compute_annotation_ggnn_loss_and_grad(c, y, target, true)
    net:backward(dy)

    local function f(params)
        if params ~= w then
            params:copy(w)
        end
        return ggnn.compute_annotation_ggnn_loss_and_grad(c, net:forward(edges, n_steps, annotations), target, false)
    end

    local fdiff_dw = compute_fdiff_grad(w, f)
    check_grad(dw, 'Backprop Grad', fdiff_dw, 'Fdiff Grad')
end

function test_per_node_annotation_ggnn_alternative_loss()
    print_test_name('[Test PerNodeGGNN Alternative Loss]')
    local edges, annotations, n_edge_types, n_total_nodes = create_sample_graph()
    local n_steps = 3
    local annotation_dim = #annotations[1][1]
    local state_dim = 3

    local target = torch.randn(n_total_nodes, annotation_dim)
    local c = nn.MSECriterion()

    local net = ggnn.PerNodeGGNN(state_dim, annotation_dim, {}, {}, n_edge_types)
    local w, dw = net:getParameters()
    dw:zero()

    local y = net:forward(edges, n_steps, annotations)
    c:forward(y, target)
    local dy = c:backward(y, target)
    net:backward(dy)

    local function f(params)
        if params ~= w then
            params:copy(w)
        end
        return c:forward(net:forward(edges, n_steps, annotations), target)
    end

    local fdiff_dw = compute_fdiff_grad(w, f)
    check_grad(dw, 'Backprop Grad', fdiff_dw, 'Fdiff Grad')
end

function test_per_node_alternative_output()
    print_test_name('[Test PerNodeGGNN Alternative Output]')
    local edges, annotations, n_edge_types, n_total_nodes = create_sample_graph()
    local n_steps = 3
    local annotation_dim = #annotations[1][1]
    local state_dim = 3
    local n_outputs = 1

    local target = create_random_annotations_from_input_annotations(annotations, n_outputs)
    local c = nn.BCECriterion()

    local net = ggnn.PerNodeGGNN(state_dim, annotation_dim, {}, {}, n_edge_types, nil, n_outputs)
    local w, dw = net:getParameters()
    dw:zero()

    local y = net:forward(edges, n_steps, annotations)
    local loss, dy = ggnn.compute_annotation_ggnn_loss_and_grad(c, y, target, true)
    net:backward(dy)

    local function f(params)
        if params ~= w then
            params:copy(w)
        end
        return ggnn.compute_annotation_ggnn_loss_and_grad(c, net:forward(edges, n_steps, annotations), target, false)
    end

    local fdiff_dw = compute_fdiff_grad(w, f)
    check_grad(dw, 'Backprop Grad', fdiff_dw, 'Fdiff Grad')
end

function test_node_selection_loss()
    print_test_name('[Test NodeSelectionGGNN loss]')

    local n_nodes_list = {3, 2}
    local n_total_nodes = 5
    local w = torch.rand(n_total_nodes)
    local x = w:view(n_total_nodes, 1)
    local target = {2, 1}
    local c = nn.CrossEntropyCriterion()

    local loss, dx = ggnn.compute_node_selection_loss_and_grad(c, x, target, n_nodes_list, true)
    local dw = dx:view(-1)
    
    local function f(params)
        if params ~= w then
            params:copy(w)
        end
        return ggnn.compute_node_selection_loss_and_grad(c, x, target, n_nodes_list, false)
    end

    local fdiff_dw = compute_fdiff_grad(w, f)
    check_grad(dw, 'Backprop Grad', fdiff_dw, 'Fdiff Grad')
end

function test_node_selection_ggnn()
    print_test_name('[Test NodeSelectionGGNN]')
    local edges, annotations, n_edge_types, n_total_nodes = create_sample_graph()
    local n_steps = 3
    local annotation_dim = #annotations[1][1]
    local state_dim = 3

    local target = create_random_node_selections(annotations)
    local c = nn.CrossEntropyCriterion()

    local net = ggnn.NodeSelectionGGNN(state_dim, annotation_dim, {}, {}, n_edge_types)
    local w, dw = net:getParameters()
    dw:zero()

    local y = net:forward(edges, n_steps, annotations)
    local loss, dy = ggnn.compute_node_selection_loss_and_grad(c, y, target, net.n_nodes_list, true)
    net:backward(dy)

    local function f(params)
        if params ~= w then
            params:copy(w)
        end
        local y = net:forward(edges, n_steps, annotations)
        return ggnn.compute_node_selection_loss_and_grad(c, y, target, net.n_nodes_list, false)
    end

    local fdiff_dw = compute_fdiff_grad(w, f)
    check_grad(dw, 'Backprop Grad', fdiff_dw, 'Fdiff Grad')
end

function test_graph_level_ggnn()
    print_test_name('[Test GraphLevelGGNN]')
    local edges, annotations, n_edge_types, n_total_nodes = create_sample_graph()
    local n_steps = 3
    local annotation_dim = #annotations[1][1]
    local state_dim = 3
    local n_classes = 3

    local target = torch.rand(#edges):mul(n_classes):ceil()
    local c = nn.CrossEntropyCriterion()

    local net = ggnn.GraphLevelGGNN(state_dim, annotation_dim, {}, {state_dim, n_classes}, n_edge_types)
    local w, dw = net:getParameters()
    dw:zero()

    local y = net:forward(edges, n_steps, annotations)
    c:forward(y, target)
    local dy = c:backward(y, target)
    net:backward(dy)

    local function f(params)
        if params ~= w then
            params:copy(w)
        end
        return c:forward(net:forward(edges, n_steps, annotations), target)
    end

    local fdiff_dw = compute_fdiff_grad(w, f)
    check_grad(dw, 'Backprop Grad', fdiff_dw, 'Fdiff Grad')
end

-- f_output_net_constructor takes state_dim and annotation_dim as input and 
-- creates an instance of the output net.
--
-- f_loss_and_grad takes criterion, prediction, target, compute_grad (boolean) 
-- as input and computes the loss and optionally the prediction gradient.
function test_output_net_grad(f_output_net_constructor, output_net_name, criterion, f_create_target, f_loss_and_grad)
    print_test_name('[Test ' .. output_net_name .. ' parameter gradient]')

    local n_nodes_list = {2, 3, 4}
    local n_total_nodes = torch.Tensor(n_nodes_list):sum()
    local state_dim = 3
    local annotation_dim = 2

    local n_outputs = annotation_dim

    local node_representations = torch.rand(n_total_nodes, state_dim)
    local node_annotations = torch.rand(n_total_nodes, annotation_dim)

    local target = f_create_target(n_nodes_list, n_outputs)

    local net = f_output_net_constructor(state_dim, annotation_dim)
    local w, dw = net:getParameters()
    dw:zero()

    local y = net:forward(node_representations, node_annotations, n_nodes_list)
    local loss, dy = f_loss_and_grad(criterion, y, target, n_nodes_list, true)
    net:backward(dy)

    local function f(params)
        if params ~= w then
            params:copy(w)
        end
        return f_loss_and_grad(criterion, net:forward(node_representations, node_annotations, n_nodes_list), target, n_nodes_list, false)
    end

    local fdiff_dw = compute_fdiff_grad(w, f)
    check_grad(dw, 'Backprop Grad', fdiff_dw, 'Fdiff Grad')
end

function test_output_net_input_grad(f_output_net_constructor, output_net_name, criterion, f_create_target, f_loss_and_grad)
    print_test_name('[Test ' .. output_net_name .. ' input gradients]')
    local n_nodes_list = {2, 3, 4}
    local n_total_nodes = torch.Tensor(n_nodes_list):sum()
    local state_dim = 3
    local annotation_dim = 2

    local n_outputs = annotation_dim -- used only by GraphLevelOutputNet

    local w = torch.rand(n_total_nodes * (annotation_dim + state_dim))
    local reshaped_w = w:view(n_total_nodes, annotation_dim + state_dim)
    local node_representations = reshaped_w:narrow(2,1,state_dim)
    local node_annotations = reshaped_w:narrow(2,state_dim+1, annotation_dim)

    local target = f_create_target(n_nodes_list, n_outputs)

    local net = f_output_net_constructor(state_dim, annotation_dim, n_outputs)
    local params = net:getParameters()
    torch.randn(params, params:size())

    local y = net:forward(node_representations, node_annotations, n_nodes_list)
    local loss, dy = f_loss_and_grad(criterion, y, target, n_nodes_list, true)
    local drep, danno = net:backward(dy)
    local dw = torch.cat(drep, danno, 2):view(-1)

    local function f(params)
        if params ~= w then
            params:copy(w)
        end
        return f_loss_and_grad(criterion, net:forward(node_representations, node_annotations, n_nodes_list), target, n_nodes_list, false)
    end

    local fdiff_dw = compute_fdiff_grad(w, f)
    check_grad(dw, 'Backprop Grad', fdiff_dw, 'Fdiff Grad')
end

function test_per_node_output_net_grad()
    test_output_net_grad(
        function(state_dim, annotation_dim, n_outputs) return ggnn.PerNodeOutputNet(state_dim, annotation_dim, {}, {}) end,
        'PerNodeOutputNet',
        nn.BCECriterion(),
        create_random_per_node_outputs_from_n_nodes_list,
        function(c, y, t, n_nodes_list, compute_grad) return ggnn.compute_annotation_ggnn_loss_and_grad(c, y, t, compute_grad) end
    )
end

function test_per_node_output_net_input_grad()
    test_output_net_input_grad(
        function(state_dim, annotation_dim, n_outputs) return ggnn.PerNodeOutputNet(state_dim, annotation_dim, {}, {}) end,
        'PerNodeOutputNet',
        nn.BCECriterion(),
        create_random_per_node_outputs_from_n_nodes_list,
        function(c, y, t, n_nodes_list, compute_grad) return ggnn.compute_annotation_ggnn_loss_and_grad(c, y, t, compute_grad) end
    )
end

function test_node_selection_output_net_grad()
    test_output_net_grad(
        function(state_dim, annotation_dim, n_outputs) return ggnn.NodeSelectionOutputNet(state_dim, annotation_dim, {}, {}) end,
        'NodeSelectionOutputNet',
        nn.CrossEntropyCriterion(),
        create_random_node_selections_from_n_nodes_list,
        ggnn.compute_node_selection_loss_and_grad
    )
end

function test_node_selection_output_net_input_grad()
    test_output_net_input_grad(
        function(state_dim, annotation_dim, n_outputs) return ggnn.NodeSelectionOutputNet(state_dim, annotation_dim, {}, {}) end,
        'NodeSelectionOutputNet',
        nn.CrossEntropyCriterion(),
        create_random_node_selections_from_n_nodes_list,
        ggnn.compute_node_selection_loss_and_grad
    )
end

function test_graph_level_output_net_grad()
    test_output_net_grad(
        function(state_dim, annotation_dim, n_outputs) return ggnn.GraphLevelOutputNet(state_dim, annotation_dim, {state_dim, n_outputs}, {}) end,
        'GraphLevelOutputNet',
        nn.CrossEntropyCriterion(),
        function(n_nodes_list, n_outputs) return torch.rand(#n_nodes_list):mul(n_outputs):ceil() end,
        function(c, y, t, n_nodes_list, compute_grad)
            local loss = c:forward(y, t)
            local dy = c:backward(y, t)
            return loss, dy
        end
    )
end

function test_graph_level_output_net_input_grad()
    test_output_net_input_grad(
        function(state_dim, annotation_dim, n_outputs) return ggnn.GraphLevelOutputNet(state_dim, annotation_dim, {state_dim, n_outputs}, {}) end,
        'GraphLevelOutputNet',
        nn.CrossEntropyCriterion(),
        function(n_nodes_list, n_outputs) return torch.rand(#n_nodes_list):mul(n_outputs):ceil() end,
        function(c, y, t, n_nodes_list, compute_grad)
            local loss = c:forward(y, t)
            local dy = c:backward(y, t)
            return loss, dy
        end
    )
end

function test_ggnn_io(net, f_load_net, model_name)
    print_test_name('[Test ' .. model_name .. ' I/O]')

    local temp_file_name = '_temp_'

    local w = net:getParameters()
    ggnn.save_model_to_file(temp_file_name, net, w)
    local net2 = f_load_net(temp_file_name)
    local w2 = net2:getParameters()

    os.execute('rm -f ' .. temp_file_name)

    check_grad(w, 'Original', w2, 'Loaded')
end

function test_base_ggnn_io()
    test_ggnn_io(ggnn.BaseGGNN(2,1,{},1), ggnn.load_base_ggnn_from_file, 'BaseGGNN')
end

function test_per_node_ggnn_io()
    test_ggnn_io(ggnn.PerNodeGGNN(2,1,{},{},1,nil,2), ggnn.load_per_node_ggnn_from_file, 'PerNodeGGNN')
end

function test_node_selection_ggnn_io()
    test_ggnn_io(ggnn.NodeSelectionGGNN(2,1,{},{},1), ggnn.load_node_selection_ggnn_from_file, 'NodeSelectionGGNN')
end

function test_graph_level_ggnn_io()
    test_ggnn_io(ggnn.GraphLevelGGNN(2,1,{},{2,2},1), ggnn.load_graph_level_ggnn_from_file, 'GraphLevelGGNN')
end

function test_base_output_io()
    test_ggnn_io(ggnn.BaseOutputNet(2,1), ggnn.load_base_output_net_from_file, 'BaseOutputNet')
end

function test_per_node_output_io()
    test_ggnn_io(ggnn.PerNodeOutputNet(2,1,{},nil,2), ggnn.load_per_node_output_net_from_file, 'PerNodeOutputNet')
end

function test_node_selection_output_io()
    test_ggnn_io(ggnn.NodeSelectionOutputNet(2,1,{}), ggnn.load_node_selection_output_net_from_file, 'NodeSelectionOutputNet')
end

function test_graph_level_output_io()
    test_ggnn_io(ggnn.GraphLevelOutputNet(2,1,{2,3}), ggnn.load_graph_level_output_net_from_file, 'GraphLevelOutputNet')
end


function test_graph_level_seq_ggnn()
    print_test_name('[Test GraphLevelSequenceGGNN]')

    local state_dim = 2
    local annotation_dim = 2
    local n_classes = 3
    local n_edge_types = 2

    local edge_list = {{{1,1,2}, {2,2,3}, {3,1,1}}, {{1,1,2}, {1,2,2}, {2,1,1}}}
    local annotations_list = {{{0,1}, {1,0}, {0,0}}, {{1,0}, {0,1}}}
    local targets = {{1, 2}, {1, 3}}

    local n_pred_steps = 2
    local n_prop_steps = 2

    local glnet = ggnn.GraphLevelGGNN(state_dim, annotation_dim, {}, {2, n_classes}, n_edge_types)
    local anet = ggnn.PerNodeGGNN(state_dim, annotation_dim, {}, {2}, n_edge_types)
    local glseqnet = ggnn.GraphLevelSequenceGGNN(glnet, anet)

    local w, dw= glseqnet:getParameters()
    print('#params=' .. w:nElement())
    dw:zero()

    local criterion = nn.CrossEntropyCriterion()

    local y = glseqnet:forward(edge_list, n_pred_steps, n_prop_steps, annotations_list)
    local _, dy = ggnn.compute_graph_level_seq_ggnn_loss_and_grad(criterion, y, targets, true)
    glseqnet:backward(dy)

    local function f(params)
        if w ~= params then
            w:copy(params)
        end
        return ggnn.compute_graph_level_seq_ggnn_loss_and_grad(criterion, glseqnet:forward(edge_list, n_pred_steps, n_prop_steps, annotations_list), targets, false)
    end
    local fdiff_dw = compute_fdiff_grad(w, f)
    check_grad(dw, 'Backprop Grad', fdiff_dw, 'Fdiff Grad')
end

function test_graph_level_seq_ggnn_io()
    test_ggnn_io(
        ggnn.GraphLevelSequenceGGNN(
            ggnn.GraphLevelGGNN(2,1,{},{2,2},1),
            ggnn.PerNodeGGNN(2,1,{},{},1,nil)
        ), 
        ggnn.load_graph_level_seq_ggnn_from_file,
        'GraphLevelSequenceGGNN'
    )
end

function test_node_selection_seq_ggnn()
    print_test_name('[Test NodeSelectionSequenceGGNN]')

    local state_dim = 2
    local annotation_dim = 2
    local n_edge_types = 2

    local edge_list = {{{1,1,2}, {2,2,3}, {3,1,1}}, {{1,1,2}, {1,2,2}, {2,1,1}}}
    local annotations_list = {{{0,1}, {1,0}, {0,0}}, {{1,0}, {0,1}}}
    local targets = {{1, 3}, {2, 2}}

    local n_pred_steps = 2
    local n_prop_steps = 2

    local nsnet = ggnn.NodeSelectionGGNN(state_dim, annotation_dim, {}, {2}, n_edge_types)
    local anet = ggnn.PerNodeGGNN(state_dim, annotation_dim, {}, {2}, n_edge_types)
    local nsseqnet = ggnn.NodeSelectionSequenceGGNN(nsnet, anet)

    local w, dw= nsseqnet:getParameters()
    print('#params=' .. w:nElement())
    dw:zero()

    local criterion = nn.CrossEntropyCriterion()

    local y = nsseqnet:forward(edge_list, n_pred_steps, n_prop_steps, annotations_list)
    local _, dy = ggnn.compute_node_selection_seq_ggnn_loss_and_grad(criterion, y, targets, nsseqnet.n_nodes_list, true)
    nsseqnet:backward(dy)

    local function f(params)
        if w ~= params then
            w:copy(params)
        end
        return ggnn.compute_node_selection_seq_ggnn_loss_and_grad(criterion, nsseqnet:forward(edge_list, n_pred_steps, n_prop_steps, annotations_list), targets, nsseqnet.n_nodes_list, false)
    end
    local fdiff_dw = compute_fdiff_grad(w, f)
    check_grad(dw, 'Backprop Grad', fdiff_dw, 'Fdiff Grad')
end

function test_node_selection_seq_ggnn_io()
    test_ggnn_io(
        ggnn.NodeSelectionSequenceGGNN(
            ggnn.NodeSelectionGGNN(2,1,{},{2},1),
            ggnn.PerNodeGGNN(2,1,{},{},1,nil)
        ), 
        ggnn.load_node_selection_seq_ggnn_from_file,
        'NodeSelectionSequenceGGNN'
    )
end

function test_graph_level_share_prop_seq_ggnn()
    print_test_name('[Test GraphLevelSequenceSharePropagationGGNN]')

    local state_dim = 2
    local annotation_dim = 2
    local n_classes = 3
    local n_edge_types = 2

    local edge_list = {{{1,1,2}, {2,2,3}, {3,1,1}}, {{1,1,2}, {1,2,2}, {2,1,1}}}
    local annotations_list = {{{0,1}, {1,0}, {0,0}}, {{1,0}, {0,1}}}
    local targets = {{1, 2}, {1, 3}}

    local n_pred_steps = 2
    local n_prop_steps = 2

    local pnet = ggnn.BaseGGNN(state_dim, annotation_dim, {}, n_edge_types)
    local glnet = ggnn.GraphLevelOutputNet(state_dim, annotation_dim, {2, n_classes})
    local anet = ggnn.PerNodeOutputNet(state_dim, annotation_dim, {2})
    local glseqnet = ggnn.GraphLevelSequenceSharePropagationGGNN(pnet, glnet, anet)

    local w, dw= glseqnet:getParameters()
    print('#params=' .. w:nElement())
    dw:zero()

    local criterion = nn.CrossEntropyCriterion()

    local y = glseqnet:forward(edge_list, n_pred_steps, n_prop_steps, annotations_list)
    local _, dy = ggnn.compute_graph_level_seq_ggnn_loss_and_grad(criterion, y, targets, true)
    glseqnet:backward(dy)

    local function f(params)
        if w ~= params then
            w:copy(params)
        end
        return ggnn.compute_graph_level_seq_ggnn_loss_and_grad(criterion, glseqnet:forward(edge_list, n_pred_steps, n_prop_steps, annotations_list), targets, false)
    end
    local fdiff_dw = compute_fdiff_grad(w, f)
    check_grad(dw, 'Backprop Grad', fdiff_dw, 'Fdiff Grad')
end

function test_graph_level_share_prop_seq_ggnn_io()
    test_ggnn_io(
        ggnn.GraphLevelSequenceSharePropagationGGNN(
            ggnn.BaseGGNN(2,1,{},1),
            ggnn.GraphLevelOutputNet(2,1,{2,2}),
            ggnn.PerNodeOutputNet(2,1,{})
        ), 
        ggnn.load_graph_level_seq_share_prop_ggnn_from_file,
        'GraphLevelSequenceSharePropagationGGNN'
    )
end

function test_node_selection_share_prop_seq_ggnn()
    print_test_name('[Test NodeSelectionSequenceSharePropagationGGNN]')

    local state_dim = 2
    local annotation_dim = 2
    local n_edge_types = 2

    local edge_list = {{{1,1,2}, {2,2,3}, {3,1,1}}, {{1,1,2}, {1,2,2}, {2,1,1}}}
    local annotations_list = {{{0,1}, {1,0}, {0,0}}, {{1,0}, {0,1}}}
    local targets = {{1, 3}, {2, 2}}

    local n_pred_steps = 2
    local n_prop_steps = 2

    local pnet = ggnn.BaseGGNN(state_dim, annotation_dim, {}, n_edge_types)
    local nsnet = ggnn.NodeSelectionOutputNet(state_dim, annotation_dim, {2})
    local anet = ggnn.PerNodeOutputNet(state_dim, annotation_dim, {2})
    local nsseqnet = ggnn.NodeSelectionSequenceSharePropagationGGNN(pnet, nsnet, anet)

    local w, dw= nsseqnet:getParameters()
    print('#params=' .. w:nElement())
    dw:zero()

    local criterion = nn.CrossEntropyCriterion()

    local y = nsseqnet:forward(edge_list, n_pred_steps, n_prop_steps, annotations_list)
    local _, dy = ggnn.compute_node_selection_seq_ggnn_loss_and_grad(criterion, y, targets, nsseqnet.n_nodes_list, true)
    nsseqnet:backward(dy)

    local function f(params)
        if w ~= params then
            w:copy(params)
        end
        return ggnn.compute_node_selection_seq_ggnn_loss_and_grad(criterion, nsseqnet:forward(edge_list, n_pred_steps, n_prop_steps, annotations_list), targets, nsseqnet.n_nodes_list, false)
    end
    local fdiff_dw = compute_fdiff_grad(w, f)
    check_grad(dw, 'Backprop Grad', fdiff_dw, 'Fdiff Grad')
end

function test_node_selection_share_prop_seq_ggnn_io()
    test_ggnn_io(
        ggnn.NodeSelectionSequenceSharePropagationGGNN(
            ggnn.BaseGGNN(2,1,{},1),
            ggnn.NodeSelectionOutputNet(2,1,{2}),
            ggnn.PerNodeOutputNet(2,1,{})
        ), 
        ggnn.load_node_selection_seq_share_prop_ggnn_from_file,
        'NodeSelectionSequenceSharePropagationGGNN'
    )
end

function test_rnn(n_outputs)
    print_test_name('[Test RNN ' .. n_outputs .. ' outputs]')
    local vocab_size = 5
    local embed_size = 2
    local hid_size = 3
    local output_size = 1

    local n_examples = 5
    local n_steps = 5
    local n_steps_2 = 4

    local x = torch.rand(n_examples, n_steps):mul(vocab_size):ceil()
    local t = torch.randn(n_examples * n_outputs, output_size)
    local x2 = torch.rand(n_examples, n_steps_2):mul(vocab_size):ceil()
    local t2 = torch.randn(n_examples * n_outputs, output_size)

    local net = rnn.RNN(vocab_size, embed_size, hid_size, output_size)
    local c = nn.MSECriterion()

    local params, grad_params = net:getParameters()
    grad_params:zero()

    net:create_fixed_components()

    net:print_model()
    print(tostring(params:nElement()) .. ' parameters')

    local y = net:forward(x, n_outputs)
    c:forward(y, t)
    net:backward(c:backward(y, t))

    y = net:forward(x2, n_outputs)
    c:forward(y, t2)
    net:backward(c:backward(y, t2))

    local function f(w)
        if w ~= params then
            params:copy(w)
        end
        return c:forward(net:forward(x, n_outputs), t) + c:forward(net:forward(x2, n_outputs), t2)
    end

    local fdiff_grad = compute_fdiff_grad(params, f)
    check_grad(grad_params, 'Backprop Grad', fdiff_grad, 'Fdiff Grad')
end

function test_lstm(n_outputs)
    print_test_name('[Test LSTM ' .. n_outputs .. ' outputs]')
    local vocab_size = 5
    local embed_size = 2
    local hid_size = 3
    local output_size = 1

    local n_examples = 5
    local n_steps = 5
    local n_steps_2 = 4

    local x = torch.rand(n_examples, n_steps):mul(vocab_size):ceil()
    local t = torch.randn(n_examples * n_outputs, output_size)
    local x2 = torch.rand(n_examples, n_steps_2):mul(vocab_size):ceil()
    local t2 = torch.randn(n_examples * n_outputs, output_size)

    local net = rnn.LSTM(vocab_size, embed_size, hid_size, output_size)
    local c = nn.MSECriterion()

    local params, grad_params = net:getParameters()
    grad_params:zero()

    net:create_fixed_components()

    net:print_model()
    print(tostring(params:nElement()) .. ' parameters')

    local y = net:forward(x, n_outputs)
    c:forward(y, t)
    net:backward(c:backward(y, t))

    y = net:forward(x2, n_outputs)
    c:forward(y, t2)
    net:backward(c:backward(y, t2))

    local function f(w)
        if w ~= params then
            params:copy(w)
        end
        return c:forward(net:forward(x, n_outputs), t) + c:forward(net:forward(x2, n_outputs), t2)
        -- return c:forward(net:forward(x), t)
    end

    local fdiff_grad = compute_fdiff_grad(params, f)
    check_grad(grad_params, 'Backprop Grad', fdiff_grad, 'Fdiff Grad')
end

function test_rnn_all()
    test_rnn(1)
    test_rnn(2)
end

function test_lstm_all()
    test_lstm(1)
    test_lstm(2)
end






-------------------- run tests -------------------------

time_func(test_print_vector)
time_func(test_checkgrad)

time_func(test_base_ggnn)
time_func(test_base_ggnn_alternative)
time_func(test_base_ggnn_alternative2)


time_func(test_per_node_annotation_loss)
time_func(test_per_node_annotation_ggnn)
time_func(test_per_node_annotation_ggnn_alternative_loss)
time_func(test_per_node_alternative_output)

time_func(test_node_selection_loss)
time_func(test_node_selection_ggnn)

time_func(test_graph_level_ggnn)

time_func(test_per_node_output_net_grad)
time_func(test_per_node_output_net_input_grad)
time_func(test_node_selection_output_net_grad)
time_func(test_node_selection_output_net_input_grad)
time_func(test_graph_level_output_net_grad)
time_func(test_graph_level_output_net_input_grad)

time_func(test_base_ggnn_io)
time_func(test_per_node_ggnn_io)
time_func(test_node_selection_ggnn_io)
time_func(test_graph_level_ggnn_io)

time_func(test_base_output_io)
time_func(test_per_node_output_io)
time_func(test_node_selection_output_io)
time_func(test_graph_level_output_io)

time_func(test_graph_level_seq_ggnn)
time_func(test_graph_level_seq_ggnn_io)

time_func(test_node_selection_seq_ggnn)
time_func(test_node_selection_seq_ggnn_io)

time_func(test_graph_level_share_prop_seq_ggnn)
time_func(test_graph_level_share_prop_seq_ggnn_io)
time_func(test_node_selection_share_prop_seq_ggnn)
time_func(test_node_selection_share_prop_seq_ggnn_io)

time_func(test_rnn_all)
time_func(test_lstm_all)
