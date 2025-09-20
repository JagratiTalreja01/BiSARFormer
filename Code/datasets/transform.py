from torchvision import transforms

def get_default_transforms():
    return transforms.Compose([
        transforms.ToTensor()
    ])
