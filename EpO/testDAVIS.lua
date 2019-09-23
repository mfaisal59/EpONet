require 'lfs'
require 'segmentDAVIS'
require 'cutorch'
require 'image'
require 'ResizeJoinTable'

cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'GPU id')
cmd:option('-model', '', 'model to test')

local params = cmd:parse(arg)
local davisDir = '/home/cvlab/FAISAL/DAVIS_Dataset/DAVIS_Dataset/DAVIS-trainval-480p'

os.execute("mkdir -p " .. davisDir .. "/Results/EpO/")

cutorch.setDevice(params.gpu + 1)
local model = torch.load('models/' .. params.model):float()
model = model:cuda()
model:evaluate()

local file = io.open('testSeqFrames.txt')
for line in file:lines() do
    local input, label = line:match("([^ ]+) ([^ ]+)")
    segment(model, line)
end

print('Finished processing')
