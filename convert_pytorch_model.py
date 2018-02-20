import torch, torchvision
from webdnn.frontend.pytorch import PyTorchConverter

model = torchvision.models.alexnet(pretrained=True)
dummy_input = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
graph = PyTorchConverter().convert(model, dummy_input)

from webdnn.backend import generate_descriptor

exec_info = generate_descriptor("webgl", graph)  # also "webassembly", "webgl", "fallback" are available.
exec_info.save("./output_pytorch_alexnet")
