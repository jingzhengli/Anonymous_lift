from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
import torchvision.transforms as transforms
from timm.data import create_transform
import torch

dataset_list = {
    "oxford_pets": OxfordPets,
    "eurosat": EuroSAT,
    "ucf101": UCF101,
    "sun397": SUN397,
    "caltech101": Caltech101,
    "dtd": DescribableTextures,
    "fgvc": FGVCAircraft,
    "food101": Food101,
    "oxford_flowers": OxfordFlowers,
    "stanford_cars": StanfordCars,
}


def build_dataset(dataset, root_path, shots):
    return dataset_list[dataset](root_path, shots)


class DataAugmentation:
    def __init__(self, weak_transform, strong_transform):
        self.transforms = [weak_transform, strong_transform]

        # self.mask_generator = MaskGenerator(
        #     input_size=args.input_size,
        #     mask_patch_size=train_config['mask_patch_size'],
        #     model_patch_size=train_config['model_patch_size'],
        #     mask_ratio=train_config['mask_ratio'],
        # )

    def __call__(self, x):
        images_weak = self.transforms[0](x)
        images_strong = self.transforms[1](x)

        # return images_weak, images_strong, self.mask_generator()
        return images_weak, images_strong


def build_transform(is_train):
    if is_train:
        weak_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(224),
            # transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor((0.48145466, 0.4578275, 0.40821073)),
                std=torch.tensor((0.26862954, 0.26130258, 0.27577711)))
        ])

        strong_transform = create_transform(
            input_size=224,
            scale=(0.3, 1),
            is_training=True,
            color_jitter=0,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        transform = DataAugmentation(weak_transform, strong_transform)

        return transform

    else:
        transform = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(args.image_mean),
                std=torch.tensor(args.image_std))
        ])
        return transform