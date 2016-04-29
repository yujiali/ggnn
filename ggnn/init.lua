-- Gated Graph Neural Network model.
--
-- Yujia Li, 02/2016
--

require 'nn'
require 'nngraph'
require 'torch'

ggnn = {}

ggnn.use_gpu = false

ggnn.PROP_NET_PREFIX = 'prop'
ggnn.REVERSE_PROP_NET_PREFIX = 'reverse-prop'
ggnn.AGGREGATION_NET_PREFIX = 'aggregation'
ggnn.DOT_PRODUCT_LAYER_PREFIX = 'dot-product'
ggnn.OUTPUT_NET_PREFIX = 'output'
ggnn.SELF_PROP_NET_PREFIX = 'self-prop'
ggnn.GATING_NET_PREFIX = 'gate'
ggnn.NODE_UPDATE_NET_PREFIX = 'node-update'

include('ggnn_util.lua')

-- 4 alternatives for the propagation network are provided:
--
-- BaseGGNN and BaseGGNNLinear are the gated version.
-- BaseGGNNSimple and BaseGGNNSimpleLinear are the "vanilla RNN" version.
-- BaseGGNN and BaseGGNNSimple both use tanh nonlinearity.
-- BaseGGNNLinear and BaseGGNNSimpleLinear do not use tanh nonlinearity.
--

include('BaseGGNN.lua')
-- include('BaseGGNNLinear.lua')
-- include('BaseGGNNSimple.lua')
-- include('BaseGGNNSimpleLinear.lua')

include('PerNodeGGNN.lua')
include('NodeSelectionGGNN.lua')
include('GraphLevelGGNN.lua')

include('BaseOutputNet.lua')
include('PerNodeOutputNet.lua')
include('NodeSelectionOutputNet.lua')
include('GraphLevelOutputNet.lua')

include('GraphLevelSequenceGGNN.lua')
include('NodeSelectionSequenceGGNN.lua')

include('GraphLevelSequenceSharePropagationGGNN.lua')
include('NodeSelectionSequenceSharePropagationGGNN.lua')

return ggnn

