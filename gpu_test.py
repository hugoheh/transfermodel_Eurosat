import torch

#torch.cuda.is_available()
#torch.cuda.device_count()
#torch.cuda.current_device()
#torch.cuda.device()
#torch.cuda.get_device_name()

#torch.zeros(1).cuda()

print('__CUDA VERSION:', torch.version.cuda)
print('__CUDNN VERSION:', torch.backends.cudnn.version())


