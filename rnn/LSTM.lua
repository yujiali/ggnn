-- Long Short-Term Memory Network
--
-- Yujia Li, 10/2015
--

local LSTM, BaseRNN = torch.class('rnn.LSTM', 'rnn.BaseRNN') 

function LSTM:__init(vocab_size, embedding_size, hid_size, output_size, module_dict)
    BaseRNN.__init(self, module_dict)

    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.hid_size = hid_size
    self.output_size = output_size

    self:create_modules()
end

function LSTM:create_modules()
    rnn.create_or_share('LookupTable', 'input-lookup', self.module_dict, {self.vocab_size, self.embedding_size})
    rnn.create_or_share('Linear', 'ih2h', self.module_dict, {self.embedding_size + self.hid_size * 2, self.hid_size * 4})
    rnn.create_or_share('Linear', 'h2o', self.module_dict, {self.hid_size, self.output_size})
end

-- this part does not need to be rebuilt for different sequences
function LSTM:create_fixed_components()
    local h_final = nn.Identity()()
    local output = rnn.create_or_share('Linear', 'h2o', self.module_dict, {self.hid_size, self.output_size})(h_final)
    self.output_net = nn.gModule({h_final}, {output})
end

function LSTM:create_network(n_steps)
    -- create all time slices
    if self.time_slices == nil then
        self.time_slices = {}
    end

    for i_step=#self.time_slices+1,n_steps do
        local input = nn.Identity()()
        local h_prev = nn.Identity()()
        local c_prev = nn.Identity()()

        local embedding = rnn.create_or_share('LookupTable', 'input-lookup', self.module_dict, {self.vocab_size, self.embedding_size})(input)
        local joint = nn.JoinTable(2)({embedding, h_prev, c_prev})
        local g = rnn.create_or_share('Linear', 'ih2h', self.module_dict, {self.embedding_size + self.hid_size * 2, self.hid_size * 4})(joint)
        local c_update = nn.Tanh()(nn.Narrow(2,1,self.hid_size)(g))
        local gates = nn.Sigmoid()(nn.Narrow(2,1+self.hid_size, self.hid_size*3)(g))
        local input_gate = nn.Narrow(2,1,self.hid_size)(gates)
        local forget_gate = nn.Narrow(2,1+self.hid_size, self.hid_size)(gates)
        local output_gate = nn.Narrow(2,1+2*self.hid_size, self.hid_size)(gates)

        local c_out = nn.CAddTable()({nn.CMulTable()({c_prev, forget_gate}), nn.CMulTable()({c_update, input_gate})})
        local h_out = nn.CMulTable()({nn.Tanh()(c_out), output_gate})

        self.time_slices[i_step] = nn.gModule({input, h_prev, c_prev}, {h_out, c_out})
    end
end

-- input is a matrix of size n_examples x n_steps. Each row is a sequence
-- example, each column is one step.  All sequences should have the same length.
function LSTM:forward(input, n_outputs)
    n_outputs = n_outputs or 1

    self.input = input
    self.n_seqs = input:size(1)
    self.seq_len = input:size(2)

    self.n_outputs = n_outputs

    if self.output_net == nil then
        self:create_fixed_components()
    end
    self:create_network(self.seq_len)

    self.h_init = torch.zeros(self.n_seqs, self.hid_size)
    self.c_init = torch.zeros(self.n_seqs, self.hid_size)
    self.h = {}
    self.c = {}

    -- first step is special
    self.h[1], self.c[1] = unpack(self.time_slices[1]:forward({input:select(2,1), self.h_init, self.c_init}))
    for i_step=2,self.seq_len do
        self.h[i_step], self.c[i_step] = unpack(self.time_slices[i_step]:forward({input:select(2,i_step), self.h[i_step-1], self.c[i_step-1]}))
    end

    if n_outputs == 1 then
        self.output = self.output_net:forward(self.h[self.seq_len])
    else
        if self.h_out == nil then
            self.h_out = torch.Tensor()
        end
        self.h_out:resize(self.n_seqs, self.hid_size * n_outputs)

        for i=1,n_outputs do
            self.h_out:narrow(2,(i-1)*self.hid_size+1, self.hid_size):copy(self.h[self.seq_len - n_outputs + i])
        end
        self.h_out:resize(self.n_seqs * n_outputs, self.hid_size)
        self.output = self.output_net:forward(self.h_out)
    end
    return self.output
end

function LSTM:backward(output_grad)
    if self.n_outputs == 1 then
        local h_grad = self.output_net:backward(self.h[self.seq_len], output_grad)
        local c_grad = torch.zeros(self.n_seqs, self.hid_size):typeAs(output_grad)
        for i_step=self.seq_len,2,-1 do
            _, h_grad, c_grad = unpack(self.time_slices[i_step]:backward({self.input:select(2,i_step), self.h[i_step-1], self.c[i_step-1]}, {h_grad, c_grad}))
        end
        self.time_slices[1]:backward({self.input:select(2,1), self.h_init, self.c_init}, {h_grad, c_grad})
    else
        local h_out_grad = self.output_net:backward(self.h_out, output_grad)
        h_out_grad:resize(self.n_seqs, self.n_outputs * self.hid_size)

        local h_grad = torch.zeros(self.n_seqs, self.hid_size):typeAs(h_out_grad)
        local c_grad = torch.zeros(self.n_seqs, self.hid_size):typeAs(h_out_grad)
        for i_step=self.seq_len,2,-1 do
            if self.seq_len - i_step + 1 <= self.n_outputs then
                h_grad:add(h_out_grad:narrow(2, (self.n_outputs - (self.seq_len-i_step+1)) * self.hid_size+1, self.hid_size))
            end
            _, h_grad, c_grad = unpack(self.time_slices[i_step]:backward({self.input:select(2,i_step), self.h[i_step-1], self.c[i_step-1]}, {h_grad, c_grad}))
        end
        self.time_slices[1]:backward({self.input:select(2,1), self.h_init, self.c_init}, {h_grad, c_grad})
    end
end

function LSTM:predict(input, n_outputs)
    n_outputs = n_outputs or 1
    local output = self:forward(input, n_outputs)
    local _, pred = output:max(2)

    if n_outputs > 1 then
        return pred:resize(pred:size(1) / n_outputs, n_outputs)
    else
        return pred
    end
end

function LSTM:print_model()
    print(string.format('LSTM: Lookup %d->%d | Input to Hid %d->%d | Hid to Output %d->%d', 
            self.vocab_size, self.embedding_size, self.embedding_size, self.hid_size, self.hid_size, self.output_size))
end

