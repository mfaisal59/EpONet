require 'image'
require 'lfs'
require 'cunn'
require 'nngraph'

function segment(model, rgbPath)
    local motionPath = string.gsub(rgbPath, 'JPEGImages', 'motionImages');
    local resized_width = 424
    local resized_height = 232
    local batch = torch.Tensor(1, 3, resized_height, resized_width);

    print(motionPath)
    local motionFrame = image.load(motionPath)
    motionFrame = image.scale(motionFrame, resized_width, resized_height, 'simple');

    batch[1] = motionFrame;

    batch = batch:float():cuda()

    local outputs = model:forward(batch)

    local preds
    preds = torch.Tensor(1, 480, 854)
    local pred = outputs[1];
    pred = nn.utils.recursiveType(pred, 'torch.DoubleTensor')
    pred = image.scale(pred, 854, 480)

    local predRaw = pred
    pred = torch.round(pred)
    preds[1] = pred

    local resultPath = string.gsub(rgbPath, 'JPEGImages/480p', 'Results/EpO');
    local resultPath = string.gsub(resultPath, 'jpg', 'png');
    local resultDir = string.gsub(resultPath, '%d+.png', '');
    print(resultDir)
    if not path.exists(resultDir) then
        os.execute("mkdir -p " .. resultDir)
    end
    image.save(resultPath, torch.round(preds));
    local resultPathRaw = string.gsub(resultPath, '(%d+).png', 'raw_%1.png');
    image.save(resultPathRaw, predRaw);
end
