require 'cudnn'
require 'nn'
require 'extracunn'
require 'image'

local ConvGRU, parent = torch.class('nn.ConvGRU', 'nn.GRU')

function ConvGRU:__init(inputSize, outputSize, rho, kc, km, stride)
    self.kc = kc
    self.km = km
    self.padc = torch.floor(kc/2)
    self.padm = torch.floor(km/2)
    self.stride = stride or 1

    parent.__init(self, inputSize, outputSize, rho or 10)
end

function ConvGRU:buildGate()
    -- Note : Input is : {input(t), output(t-1)}
    local gate = nn.Sequential()
    local input2gate = cudnn.SpatialConvolution(self.inputSize, self.outputSize, self.kc, self.kc, self.stride, self.stride, self.padc, self.padc)
    local output2gate = nn.SpatialConvolutionNoBias(self.outputSize, self.outputSize, self.km, self.km, self.stride, self.stride, self.padm, self.padm)
    local para = nn.ParallelTable()
    para:add(input2gate):add(output2gate)
    gate:add(para)
    gate:add(nn.CAddTable())
    gate:add(cudnn.Sigmoid())
    return gate
end

-------------------------- factory methods -----------------------------
function ConvGRU:buildModel()
    -- input : {input, prevOutput}
    -- output : {output}

    self.inputGate = self:buildGate()
    self.resetGate = self:buildGate()

    local concat = nn.ConcatTable():add(nn.Identity()):add(self.inputGate):add(self.resetGate)
    local seq = nn.Sequential()
    seq:add(concat)
    seq:add(nn.FlattenTable()) -- x(t), s(t-1), r, z

    -- Rearrange to x(t), s(t-1), r, z, s(t-1)
    local concat = nn.ConcatTable()  --
    concat:add(nn.NarrowTable(1,4)):add(nn.SelectTable(2))
    seq:add(concat):add(nn.FlattenTable())

    -- h
    local t1 = nn.Sequential()
    t1:add(nn.SelectTable(1))
    local t2 = nn.Sequential()
    t2:add(nn.NarrowTable(2,2)):add(nn.CMulTable())
    t1:add(cudnn.SpatialConvolution(self.inputSize, self.outputSize, self.kc, self.kc, self.stride, self.stride, self.padc, self.padc))
    t2:add(nn.SpatialConvolutionNoBias(self.outputSize, self.outputSize, self.km, self.km, self.stride, self.stride, self.padm, self.padm))

    local concat = nn.ConcatTable()
    concat:add(t1):add(t2)
    local hidden = nn.Sequential()
    hidden:add(concat):add(nn.CAddTable()):add(nn.Tanh())

    -- 1-z
    local z1 = nn.Sequential()
    z1:add(nn.SelectTable(4))
    z1:add(nn.SAdd(-1, true))  -- Scalar add & negation

    -- z * h
    local z2 = nn.Sequential()
    z2:add(nn.NarrowTable(4,2))
    z2:add(nn.CMulTable())

    -- (1 - z) * h
    local o1 = nn.Sequential()
    local concat = nn.ConcatTable()
    concat:add(hidden):add(z1)
    o1:add(concat):add(nn.CMulTable())

    local o2 = nn.Sequential()
    local concat = nn.ConcatTable()
    concat:add(o1):add(z2)
    o2:add(concat):add(nn.CAddTable())

    seq:add(o2)

    return seq
end

------------------------- forward backward -----------------------------
function ConvGRU:updateOutput(input)
    local prevOutput

    if self.step == 1 then
        prevOutput = self.userPrevOutput or self.zeroTensor
        self.zeroTensor:resize(self.outputSize, input:size(2), input:size(3)):zero()
    else
        -- previous output and memory of this module
        prevOutput = self.output
    end

    -- output(t) = gru{input(t), output(t-1)}
    local output
    if self.train ~= false then
        self:recycle()
        local recurrentModule = self:getStepModule(self.step)
        -- the actual forward propagation
        output = recurrentModule:updateOutput{input, prevOutput}
    else
        output = self.recurrentModule:updateOutput{input, prevOutput}
    end

    self.outputs[self.step] = output

    self.output = output

    self.step = self.step + 1
    self.gradPrevOutput = nil
    self.updateGradInputStep = nil
    self.accGradParametersStep = nil
    self.gradParametersAccumulated = false
    -- note that we don't return the cell, just the output
    return self.output
end

function ConvGRU:initBias(forgetBias, otherBias)
    local oBias = otherBias or 0
    local rBias = forgetBias or 1
    self.inputGate.modules[1].modules[1].bias:fill(oBias)
    self.resetGate.modules[1].modules[1].bias:fill(rBias)
end