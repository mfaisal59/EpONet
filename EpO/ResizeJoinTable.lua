require 'nn'

local ResizeJoinTable, parent = torch.class('nn.ResizeJoinTable', 'nn.Module')

function ResizeJoinTable:__init(dimension)
    parent.__init(self)
    self.size = torch.LongStorage()
    self.dimension = dimension
    self.gradInput = {}

    self.join = nn.JoinTable(dimension, nil)

    self.model = nn.Sequential()
    local params = {owidth = 1; oheight = 1}
    local parallel = nn.ParallelTable()
    parallel:add(nn.SpatialUpSamplingBilinear(params))
    parallel:add(nn.Identity())
    self.model:add(parallel)
    self.model:add(self.join)

    self.model:float()
    self.model:training()
    self.model:cuda()
end

function ResizeJoinTable:_getPositiveDimension(input)
    return self.join:_getPositiveDimension(input)
end

function ResizeJoinTable:updateOutput(input)
    local second = input[2]

    self.model.modules[1].modules[1].owidth = second:size(4)
    self.model.modules[1].modules[1].oheight = second:size(3)

    return self.model:updateOutput(input)
end

function ResizeJoinTable:clearState()
    self.model:clearState();
end

function ResizeJoinTable:updateGradInput(input, gradOutput)
    self.gradInput = self.model:updateGradInput(input, gradOutput)
    return self.gradInput
end

function ResizeJoinTable:type(type, tensorCache)
    self.gradInput = {}
    return parent.type(self, type, tensorCache)
end
