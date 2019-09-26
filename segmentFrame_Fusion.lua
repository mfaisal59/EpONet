require 'image'
require 'lfs'
require 'cunn'
require 'nngraph'
require 'rnn'
require 'ConvGRU'
require 'ResizeJoinTable'

function segment(motionModel, path, appModel, mean_pixel)
    local motionPath = string.gsub(path, 'JPEGImages', 'motionImages');

    local resized_width = 424
    local resized_height = 232
    local batch = torch.Tensor(1, 3, resized_height, resized_width);

    local motionFrame = image.load(motionPath)
    local rgb = image.load(path)

    local preprocessed_input = rgb:clone()
    -- Convert to BGR
    preprocessed_input[{{1},{},{}}] = rgb[{{3},{},{}}]:clone()
    preprocessed_input[{{3},{},{}}] = rgb[{{1},{},{}}]:clone()

    preprocessed_input = preprocessed_input:mul(255)
    local mean_image = torch.repeatTensor(mean_pixel:view(3,1,1), 1, preprocessed_input:size()[2], preprocessed_input:size()[3]):float()
    preprocessed_input = preprocessed_input:add(-mean_image)

    local featureMap = appModel:forward(preprocessed_input:float():cuda())

    motionFrame = image.scale(motionFrame, resized_width, resized_height, 'simple');

    batch[1] = motionFrame;

    batch = batch:float():cuda()

    local outputs = motionModel:forward(batch)

    local preds
    local pred = outputs[1];
    local resHeight = featureMap:size(2)
    local resWidth = featureMap:size(3)
    pred = nn.utils.recursiveType(pred, 'torch.FloatTensor')

    pred = image.scale(pred:float(), resWidth, resHeight);
    return featureMap, pred
end
