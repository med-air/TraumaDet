import os, torch, random
import numpy as np
import pandas as pd
from monai import transforms
from torch.utils.data import DataLoader
from prompt_trainer import trainer
from dataset.FusionDataset import FusionDataset
from models.GLFF_CA_Prompt import Global_with_Local_UnetClassification, Global_with_Local_Prompt_Fusion_UnetClassification, Local_Branch_Prompt_Fused2, Global_with_Local_noFusion, Global_with_Local_UnetClassification_Global_Prompt_Embedding, Global_Prompt
from optimizers.lr_scheduler import WarmupCosineSchedule,LinearWarmupCosineAnnealingLR



def get_data_loader(args):

    local_file_root = "/research/d1/rshr/qxhu/PublicDataset/RSNA2023/preprocessed_data/our_methods"
    global_file_root = "/research/d1/rshr/qxhu/PublicDataset/RSNA2023/preprocessed_data/baseline_methods"

    labels_df = pd.read_csv('/research/d1/rshr/qxhu/PublicDataset/RSNA2023/preprocessed_data/label.csv', index_col="ID")   

    train_samples = []
    with open('/research/d1/rshr/jxyu/projects/MICCAI2024_LocalGlobal/data_preprocessing/train_data.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            sample = line.strip()
            train_samples.append(sample)

    val_samples   = []
    with open('/research/d1/rshr/jxyu/projects/MICCAI2024_LocalGlobal/data_preprocessing/val_data.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            sample = line.strip()
            val_samples.append(sample)

    local_train_images = []
    global_train_images = []
    train_labels = []
    for sample in train_samples:
        name = int(sample.split('_')[0])
        local_train_images.append(os.path.join(local_file_root, sample))
        global_train_images.append(os.path.join(global_file_root, sample))
        train_labels.append(labels_df.loc[name].values)
    train_labels = np.array(train_labels, dtype=float)
    train_labels_list = np.any(train_labels, axis=1).astype(int).tolist()
    train_labels = torch.FloatTensor(train_labels)

    local_val_images = []
    global_val_images = []
    val_labels = []
    for sample in val_samples:
        name = int(sample.split('_')[0])
        local_val_images.append(os.path.join(local_file_root, sample))
        global_val_images.append(os.path.join(global_file_root, sample))
        val_labels.append(labels_df.loc[name].values)
    val_labels = np.array(val_labels, dtype=float)
    val_labels = torch.FloatTensor(val_labels)  

    x, y, z = args.resize_x, args.resize_y, args.resize_z

    local_train_img_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            # transforms.Resize(spatial_size=(x, y, z), mode="area"),
            transforms.RandFlip(prob=0.2, spatial_axis=0),
            transforms.RandFlip(prob=0.2, spatial_axis=1),
            transforms.RandFlip(prob=0.2, spatial_axis=2),
            transforms.RandRotate90(prob=0.2, max_k=3),
            transforms.RandScaleIntensity(factors=0.15, prob=0.3),
            transforms.RandShiftIntensity(offsets=0.15, prob=0.3),

            # Add more intensity-based transform
            transforms.RandAdjustContrast(prob=0.2),
            # transforms.RandHistogramShift(prob=0.2),
            # transforms.RandGibbsNoise(prob=0.2),
            # transforms.RandKSpaceSpikeNoise(prob=0.2)
            
        ]
    )
    # add more intensity-based transform.
    local_train_seg_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            # transforms.Resize(spatial_size=(x, y, z), mode="nearest"),
            transforms.RandFlip(prob=0.2, spatial_axis=0),
            transforms.RandFlip(prob=0.2, spatial_axis=1),
            transforms.RandFlip(prob=0.2, spatial_axis=2),
            transforms.RandRotate90(prob=0.2, max_k=3),
        ]
    )

    local_val_img_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            # transforms.Resize(spatial_size=(x, y, z), mode="area"),
        ]
    )
    local_val_seg_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            # transforms.Resize(spatial_size=(x, y, z), mode="nearest"),
        ]
    )

    global_train_img_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            transforms.Resize(spatial_size=(x, y, z), mode="area"),
            transforms.RandFlip(prob=0.2, spatial_axis=0),
            transforms.RandFlip(prob=0.2, spatial_axis=1),
            transforms.RandFlip(prob=0.2, spatial_axis=2),
            transforms.RandRotate90(prob=0.2, max_k=3),
            transforms.RandScaleIntensity(factors=0.15, prob=0.3),
            transforms.RandShiftIntensity(offsets=0.15, prob=0.3),

            # Add more intensity-based transform
            transforms.RandAdjustContrast(prob=0.2),
            # transforms.RandHistogramShift(prob=0.2),
            # transforms.RandGibbsNoise(prob=0.2),
            # transforms.RandKSpaceSpikeNoise(prob=0.2)
            
        ]
    )
    # add more intensity-based transform.
    global_train_seg_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            transforms.Resize(spatial_size=(x, y, z), mode="nearest"),
            transforms.RandFlip(prob=0.2, spatial_axis=0),
            transforms.RandFlip(prob=0.2, spatial_axis=1),
            transforms.RandFlip(prob=0.2, spatial_axis=2),
            transforms.RandRotate90(prob=0.2, max_k=3),
        ]
    )

    global_val_img_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            transforms.Resize(spatial_size=(x, y, z), mode="area"),
        ]
    )
    global_val_seg_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            transforms.Resize(spatial_size=(x, y, z), mode="nearest"),
        ]
    )

    train_ds = FusionDataset(local_npz_files=local_train_images, global_npz_files=global_train_images, labels=train_labels, local_img_transforms=local_train_img_transform, local_seg_transforms=None, global_img_transforms=global_train_img_transform, global_seg_transforms=None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=6)
    # train_loader = DataLoader(train_ds, batch_size=args.batch_size,
    #                           sampler=ImbalancedDatasetSampler(dataset=train_ds, labels=train_labels_list),
    #                           )

    val_ds = FusionDataset(local_npz_files=local_val_images, global_npz_files=global_val_images, labels=val_labels, local_img_transforms=local_val_img_transform, local_seg_transforms=None, global_img_transforms=global_val_img_transform, global_seg_transforms=None)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)


    return train_loader, val_loader

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def _get_models(args):

    if args.model_name == "local_global":

        model = Global_with_Local_UnetClassification(out_channels = 3, local_prompt = False)
        model.load_params(torch.load(args.pretrain, map_location='cpu')['net']) # load pretrain model
    
    elif args.model_name == "local_prompt_global":

        model = Global_with_Local_UnetClassification(out_channels = 3, local_prompt = True)
        model.load_params(torch.load(args.pretrain, map_location='cpu')['net']) # load pretrain model
        word_embedding = torch.load("four_organ.pth")
        model.localbranch.organ_embedding.data = word_embedding.float()
        print('load word embedding')
    
    # elif args.model_name == "local_prompt_global_prompt_0.25":
    #     model = Global_with_Local_UnetClassification(out_channels = 3, local_prompt = True)
    #     model.load_params(torch.load(args.pretrain, map_location='cpu')['net']) # load pretrain model
    #     word_embedding = torch.load("four_organ.pth")
    #     model.localbranch.organ_embedding.data = word_embedding.float()
    #     print('load word embedding')
    
    elif args.model_name == "local_prompt_global_prompt":

        model = Global_with_Local_UnetClassification(out_channels = 3, local_prompt = True)
        model.load_params(torch.load(args.pretrain, map_location='cpu')['net']) # load pretrain model
        word_embedding = torch.load("four_organ.pth")
        model.localbranch.organ_embedding.data = word_embedding.float()
        print('load word embedding')
        # args.prompt_loss = True
    
    # elif args.model_name == "local_prompt_fusion_global_prompt":
    #     model = Global_with_Local_Prompt_Fusion_UnetClassification(out_channels = 3, local_prompt = True)
    #     model.load_params(torch.load(args.pretrain, map_location='cpu')['net']) # load pretrain model
    #     word_embedding = torch.load("four_organ.pth")
    #     model.localbranch.organ_embedding.data = word_embedding.float()
    #     print('load word embedding')

    # elif args.model_name == "local_prompt_fusionOneWay":
    #     model = Local_Branch_Prompt_Fused2(out_channels=3, local_prompt=True)
    #     model.load_params(torch.load(args.pretrain, map_location='cpu')['net']) # load pretrain model
    #     word_embedding = torch.load("four_organ.pth")
    #     model.localbranch.organ_embedding.data = word_embedding.float()
    #     print('load word embedding')
    
    elif args.model_name == "Global_with_Local_noFusion":
        model = Global_with_Local_noFusion(out_channels = 3, local_prompt = False)
        model.load_params(torch.load(args.pretrain, map_location='cpu')['net']) # load pretrain model
    
    # elif args.model_name == "local_prompt_global_singleFusion":
    #     model = Global_with_Local_UnetClassification(out_channels = 3, local_prompt = True, CrossAttention = False)
    #     model.load_params(torch.load(args.pretrain, map_location='cpu')['net']) # load pretrain model
    #     word_embedding = torch.load("four_organ.pth")
    #     model.localbranch.organ_embedding.data = word_embedding.float()
    #     print('load word embedding')
    # elif args.model_name == "local_prompt_global_prompt_singleFusion_embedding":
    #     model = Global_with_Local_UnetClassification_Global_Prompt_Embedding(out_channels = 3, local_prompt = True, CrossAttention= False)
    #     model.load_params(torch.load(args.pretrain, map_location='cpu')['net']) # load pretrain model
    #     word_embedding = torch.load("four_organ.pth")
    #     model.localbranch.organ_embedding.data = word_embedding.float()
    #     print('load word embedding')
 
    elif args.model_name == "Global_Prompt":
        model = Global_Prompt(out_channels = 3)

    else:
        raise RuntimeError("Do not support the method!")

    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description='medical segmentation contest')
    parser.add_argument('--max_epochs', default=400, type=int)
    parser.add_argument('--val_every', default=10, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    
    parser.add_argument('--log_dir', default="runs", type=str)
    parser.add_argument('--model_name', default=f"GLFF_eph400_lr5e-4_Trans6_Global128_debug", type=str)
    parser.add_argument('--pretrain', default=f"./unet.pth", type=str)
    parser.add_argument('--resize_x', default=128, type=int)
    parser.add_argument('--resize_y', default=128, type=int)
    parser.add_argument('--resize_z', default=128, type=int)

    parser.add_argument('--alfa', default=1, type=float)
    parser.add_argument('--prompt_loss', default=False, type=bool)
    
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    args.device = device

    # Print All Config
    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')

    # loader
    train_loader, val_loader = get_data_loader(args)

    model = _get_models(args)
    # model = Global_with_Local_UnetClassification(out_channels = 3)
    # model.load_params(torch.load(args.pretrain, map_location='cpu')['net']) # load pretrain model
    # word_embedding = torch.load("four_organ.pth")
    # model.localbranch.organ_embedding.data = word_embedding.float()
    # print('load word embedding')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=10, max_epochs=args.max_epochs
        )
    
    loss_function = torch.nn.BCEWithLogitsLoss()
    prompt_loss = args.prompt_loss

    trainer(model, train_loader, val_loader, optimizer, scheduler, loss_function, prompt_loss, args)

if __name__ == "__main__":
    setup_seed()
    main()


# python GLFF_train.py --model_name Global_prompt --alfa 0.9 --prompt_loss True

# python GLFF_train.py --model_name local_prompt_global_prompt --alfa 0.8 --prompt_loss True
# python GLFF_train.py --model_name local_prompt_global_prompt_0.25 --alfa 0.75 --prompt_loss Truesqu