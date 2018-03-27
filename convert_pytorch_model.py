import torch, torchvision
from webdnn.frontend.pytorch import PyTorchConverter
import traceback

# Alexnet
model = torchvision.models.alexnet(pretrained=True)
dummy_input = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
graph = PyTorchConverter().convert(model, dummy_input)

# resnet18
model = torchvision.models.resnet18(pretrained=True)
print(
[x for x in torchvision.models.__dict__.items() if x[0][0]!='_']
)
success = []
fail = []
for name, modelf in [x for x in torchvision.models.__dict__.items() if x[0][0]!='_']:
    if name[0].upper() == name[0]:
        continue
    print (type(modelf))
    if type(modelf) != type(lambda: 0):
        continue
    print('######################################################')
    print('Converting {}'.format(name))
    print('######################################################')
    try:
        model = modelf(pretrained=False)
        dummy_input = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
        graph = PyTorchConverter().convert(model, dummy_input)
        success.append(name)
    except Exception as e:
        print(e)
        print('{} failed'.format(name))
        fail.append((name, e))

print ('----------------------------------------' )
print ('Successes: {}'.format(len(success))       )
print ('  {}'.format(success)                      )
print ('----------------------------------------' )
print ('Failures: {}'.format(len(fail))           )
print ('  {}'.format(fail)                        )
print ('----------------------------------------' )

exit(1)

from webdnn.backend import generate_descriptor

exec_info = generate_descriptor("webgl", graph)  # also "webassembly", "webgl", "fallback" are available.
exec_info.save("./output_pytorch_alexnet")
