import os
from torchvision import datasets, transforms
from dataloader.Eyepacs import Eyepacs
from dataloader.Messidor import Messidor1Dataset,Messidor2Dataset
from dataloader.Apots import Apots
from dataloader.RFMid import RFMiD
from dataloader.ODIR import ODIR
from dataloader.Rsnr import RSNR
from dataloader.MICCAI import MICCAI
from dataloader.EyeQ import EyeQ
from dataloader.IDRID import IDRID
from dataloader.UWF4DR import UWF4DR, MuCaRD, MURED

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100


    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    elif args.data_set == "EyePacs":
        if is_train:
            dataset = Eyepacs(image_dir='dataset/EyePACS/train_crop',file_dir='dataset/EyePACS/train_crop.csv',split='train',val_test='None',transform=transform)
        else:
            dataset = Eyepacs(image_dir='dataset/EyePACS/test_crop',file_dir='dataset/EyePACS/test_crop.csv',split='val',val_test='Private',transform=transform)
        nb_classes = args.nb_classes
        
    elif args.data_set == "UWF4DR":
        if is_train:
            dataset = UWF4DR(image_dir='/scratch/xinli38/data/UWF4DR/train',label_dir=args.fold_train,transform=transform)
            # dataset = UWF4DR(args.fold_train, transform=transform)
        else:
            dataset = UWF4DR(image_dir='/scratch/xinli38/data/UWF4DR/test',label_dir=args.fold_test,transform=transform)
            # dataset = UWF4DR(args.fold_test, transform=transform)
        nb_classes = args.nb_classes
        
    elif args.data_set == "messidor1":
        if is_train: 
            dataset = Messidor1Dataset(image_dir='/scratch/xinli38/data/Messidor/image',label_dir='/scratch/xinli38/data/Messidor/train5.csv',transform=transform)
        else:
            dataset = Messidor1Dataset(image_dir='/scratch/xinli38/data/Messidor/image',label_dir='/scratch/xinli38/data/Messidor/test5.csv',transform=transform)
        nb_classes = args.nb_classes
    elif args.data_set == 'messidor2':
        if is_train:
            dataset = Messidor2Dataset(image_dir='dataset/Messidor-2/crop',label_dir='dataset/Messidor-2/train.csv',transform=transform)
        else:
            dataset = Messidor2Dataset(image_dir='dataset/Messidor-2/crop',label_dir='dataset/Messidor-2/test.csv',transform=transform)
        nb_classes = args.nb_classes
    elif args.data_set == 'rfmid':
        if is_train:
            dataset = RFMiD(image_dir='/scratch/xinli38/data/RFMiD/train',label_dir=args.fold_train,transform=transform)
        else:
            dataset = RFMiD(image_dir='/scratch/xinli38/data/RFMiD/test',label_dir=args.fold_test,transform=transform)
        nb_classes = args.nb_classes
    elif args.data_set == 'apots':
        if is_train:
            dataset = Apots(image_dir='/scratch/xinli38/data/APTOS/images',label_dir=args.fold_train,transform=transform)
        else:
            dataset = Apots(image_dir='/scratch/xinli38/data/APTOS/images',label_dir=args.fold_test,transform=transform)
        nb_classes = args.nb_classes

    elif args.data_set == 'EyeQ':
        if is_train:
            dataset = EyeQ(image_dir='/scratch/xinli38/data/EyeQ/image_256',label_dir=args.fold_train,transform=transform)
        else:
            dataset = EyeQ(image_dir='/media/xinli38/T7 Touch/EyePACS/Contextual_unpaired_2', label_dir=args.fold_test,transform=transform)
        nb_classes = args.nb_classes

    elif args.data_set == 'idrid':
        if is_train:
            dataset = IDRID(image_dir='/scratch/xinli38/data/IDRID/image/train_256',label_dir=args.fold_train,transform=transform)
        else:
            dataset = IDRID(image_dir='/scratch/xinli38/data/IDRID/image/test_256', label_dir=args.fold_test,transform=transform)
        nb_classes = args.nb_classes   

    elif args.data_set == 'rsna':
        if is_train:
            dataset = RSNR(image_dir='dataset/APOTS/crop',label_dir='dataset/APOTS/train_1.csv',transform=transform)
        else:
            dataset = RSNR(image_dir='dataset/APOTS/crop',label_dir='dataset/APOTS/train_1.csv',transform=transform)
        nb_classes = args.nb_classes


    elif args.data_set == 'MICCAI':
        if is_train:
            dataset = MICCAI(image_dir=args.data_path,label_dir=args.fold_train,transform=transform)
        else:
            dataset = MICCAI(image_dir=args.data_path,label_dir=args.fold_test,transform=transform)
        nb_classes = args.nb_classes

    elif args.data_set == 'odir':
        if is_train:
            dataset = ODIR(image_dir="/scratch/xinli38/data/ODIR/images",label_dir=args.fold_train,transform=transform)
        else:
            dataset = ODIR(image_dir="/scratch/xinli38/data/ODIR/images",label_dir=args.fold_test,transform=transform)
        nb_classes = args.nb_classes

    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    input_size = args.input_size 
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if args.data_set == "MICCAI" or args.data_set == "odir":
        if is_train:
            if args.data_set == "no":
                return transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(
                        degrees=20,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        fill=0
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5]
                    )
                ])

            else:

                return transforms.Compose([
                    transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0), ratio=(0.75, 1.3333),interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    #T.RandomApply([T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET)],p=0.3),
                    transforms.RandomRotation(degrees=[-180, 180],
                                    fill=0,interpolation=transforms.InterpolationMode.BICUBIC),
                    #transforms.RandomAffine(degrees=0, translate=[0.15, 0.15], fill=0,interpolation=transforms.InterpolationMode.BICUBIC),
                    #T.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(brightness=0.4,
                                    contrast=0.4,
                                    saturation=0.4,
                                    hue=0.4
                                    ),
                    #transforms.RandomApply([transforms.GaussianBlur(kernel_size=(7,13),sigma=(9,11))],p=0.5),
                    #transforms.RandomAutocontrast(p=0.3),
                    transforms.ToTensor(),
                    # messidor 
                    # transforms.Normalize([0.425753653049469, 0.29737451672554016, 0.21293757855892181], [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]),
                    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
                    # T.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
                    transforms.RandomErasing(p=0.4)
                    ])
        else:
            args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            
            return transforms.Compose([
                transforms.Resize(size=size,interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                # transforms.Normalize([0.425753653049469, 0.29737451672554016, 0.21293757855892181], [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])
        
        
    else:
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std,
            )
            if not resize_im:
                transform.transforms[0] = transforms.RandomCrop(
                    args.input_size, padding=4)
            return transform

        t = []
        if resize_im:
            # warping (no cropping) when evaluated at 384 or larger
            if args.input_size >= 384:  
                t.append(
                transforms.Resize((args.input_size, args.input_size), 
                                interpolation=transforms.InterpolationMode.BICUBIC), 
            )
                print(f"Warping {args.input_size} size input images...")
            else:
                if args.crop_pct is None:
                    args.crop_pct = 224 / 256
                size = int(args.input_size / args.crop_pct)
                t.append(
                    # to maintain same ratio w.r.t. 224 images
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
                )
                t.append(transforms.CenterCrop(args.input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)