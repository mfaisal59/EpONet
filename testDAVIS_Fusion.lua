require 'lfs'
require 'segmentFrame_Fusion'
require 'cutorch'
require 'nnx'

cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'GPU id')
cmd:option('-motionModel', '', 'Motion model to test')
cmd:option('-memoryModel', '', 'Memory model to test')

local params = cmd:parse(arg)
--change this path to your davis directory
local davisPath = '/home/cvlab/FAISAL/DAVIS_Dataset/DAVIS-trainval-480p'

os.execute('mkdir -p ' .. davisPath .. '/Results/EpO+')

cutorch.setDevice(params.gpu + 1)
torch.setdefaulttensortype('torch.FloatTensor')

local motionModel = torch.load('models/' .. params.motionModel):float()
motionModel = motionModel:cuda()
motionModel:evaluate()

local memoryModel = torch.load('models/' .. params.memoryModel):float()
memoryModel = memoryModel:cuda()
memoryModel:evaluate()
memoryModel:remember('both')
memoryModel:forget()

local deeplab = torch.load('models/deeplab.net')

local mean_pixel = deeplab['meanpixel']

local appModel = deeplab['net']

appModel = appModel['features']

appModel:remove(39)
appModel:remove(38)

appModel:remove(37)
appModel:remove(36)
appModel:remove(35)

appModel:add(nn.SpatialUpSamplingBilinear({oheight=60, owidth=107}))

appModel:float()
appModel:evaluate()
appModel = appModel:cuda()

local inputTable = {}
local frameNames = {}

local file = io.open('testSeqFrames.txt')
for line in file:lines() do
    --testVideo(motionModel, line, appModel, mean_pixel, memoryModel)
    table.insert(frameNames, line)
    local featureMap, pred = segment(motionModel, line, appModel, mean_pixel)
    if pred then
    	table.insert(inputTable, {featureMap:float():clone(), pred:clone()})
    end
end    


motionModel:clearState()
appModel:clearState()
appModel:forward(torch.zeros(1, 3, 32, 32):float():cuda())
collectgarbage();

for i = 1, #frameNames do
    inputTable[i][1] = inputTable[i][1]:cuda()
    inputTable[i][2] = inputTable[i][2]:cuda()
end


local memoeyOutputs = memoryModel:updateOutput(inputTable)

for i = 1, #frameNames do
    local output = nn.utils.recursiveType(memoeyOutputs[i], 'torch.FloatTensor')
    output = output[1]

    output = image.scale(output, 854, 480);

    local path = string.gsub(frameNames[i], 'png', 'jpg')
    local resultPath = string.gsub(path, 'JPEGImages/480p', 'Results/EpO+');
    local resultPath = string.gsub(resultPath, 'jpg', 'png');
    local resultDir = string.gsub(resultPath, '%d+.png', '');
    os.execute("mkdir -p " .. resultDir)

    local rawPath = string.gsub(resultPath, '(%d+).png', 'raw_%1.png');
    image.save(rawPath, output);

    local inputPath = string.gsub(resultPath, '(%d+).png', 'input_%1.png');
    local motion = nn.utils.recursiveType(inputTable[i][2], 'torch.FloatTensor')
    motion = image.scale(motion, 854, 480);
    image.save(inputPath, motion[{{1}, {}, {}}])
end

memoryModel:forget()
memoryModel:clearState()
collectgarbage();


