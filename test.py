import torch 
from torchvision.transforms import Resize
image = torch.rand(1600, 1200, 3)

image = Resize((224, 224))(image)
print(image.size())
