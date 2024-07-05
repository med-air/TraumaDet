import os
import time 
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import monai.transforms as transforms
from dataset.FusionDataset import FusionDataset
from models.GLFF_CA_Prompt import Global_with_Local_UnetClassification, Global_with_Local_Prompt_Fusion_UnetClassification, Global_with_Local_UnetClassification_Global_Prompt_Embedding, Global_Prompt
from torch.utils.data import DataLoader

def _get_models(args):

    if args.model_name == "local_global":

        model = Global_with_Local_UnetClassification(out_channels = 3, local_prompt = False)
        model.load_params(torch.load(args.pretrain, map_location='cpu')['net']) # load pretrain model
        model.load_state_dict(torch.load("/research/d1/rshr/jxyu/projects/MICCAI2024_LocalGlobal/baseline_main/runs/local_branch_network_epoch400_lr5e-4_bs4_debug/model_best.pt"))
    
    elif args.model_name == "local_prompt_global":

        model = Global_with_Local_UnetClassification(out_channels = 3, local_prompt = True)
        model.load_params(torch.load(args.pretrain, map_location='cpu')['net']) # load pretrain model
        word_embedding = torch.load("four_organ.pth")
        model.localbranch.organ_embedding.data = word_embedding.float()
        print('load word embedding')
        model.load_state_dict(torch.load("/research/d1/rshr/jxyu/projects/MICCAI2024_LocalGlobal/baseline_main/runs/local_branch_network_epoch400_lr5e-4_bs4_debug/model_best.pt"))
    
    elif args.model_name == "local_prompt_global_prompt":

        model = Global_with_Local_UnetClassification(out_channels = 3, local_prompt = True)
        model.load_params(torch.load(args.pretrain, map_location='cpu')['net']) # load pretrain model
        word_embedding = torch.load("four_organ.pth")
        model.localbranch.organ_embedding.data = word_embedding.float()
        print('load word embedding')
        model.load_state_dict(torch.load("/research/d1/rshr/jxyu/projects/MICCAI2024_LocalGlobal/baseline_main/runs/GLFF_eph400_lr5e-4_Sampler_Trans6_Global128_debug/model_best.pt"))
    
    # elif args.model_name == "local_prompt_fusion_global_prompt":
    #     model = Global_with_Local_Prompt_Fusion_UnetClassification(out_channels = 3, local_prompt = True)
    #     model.load_params(torch.load(args.pretrain, map_location='cpu')['net']) # load pretrain model
    #     word_embedding = torch.load("four_organ.pth")
    #     model.localbranch.organ_embedding.data = word_embedding.float()
    #     print('load word embedding')
    #     model.load_state_dict(torch.load("/research/d1/rshr/jxyu/projects/MICCAI2024_LocalGlobal/baseline_main/runs/local_prompt_fusion_global_prompt/model_best.pt"))

    # elif args.model_name == "local_prompt_fusionOneWay":
    #     model = Local_Branch_Prompt_Fused2(out_channels=3, local_prompt=True)
    #     model.load_params(torch.load(args.pretrain, map_location='cpu')['net']) # load pretrain model
    #     word_embedding = tyjn67orch.load("four_organ.pth")
    #     model.localbranch.organ_embedding.data = word_embedding.float()
    #     print('load word embedding')
    #     model.load_state_dict(torch.load("/research/d1/rshr/jxyu/projects/MICCAI2024_LocalGlobal/baseline_main/runs/local_branch_network_epoch400_lr5e-4_bs4_debug/model_best.pt"))
    
    # elif args.model_name == "Changed_local_prompt_global_prompt":
    #     model = Global_with_Local_UnetClassification(out_channels = 3, local_prompt = True)
    #     model.load_params(torch.load(args.pretrain, map_location='cpu')['net']) # load pretrain model
    #     word_embedding = torch.load("four_organ.pth")
    #     model.localbranch.organ_embedding.data = word_embedding.float()
    #     print('load word embedding')
    #     model.load_state_dict(torch.load("/research/d1/rshr/jxyu/projects/MICCAI2024_LocalGlobal/baseline_main/runs/Changed_local_prompt_global_prompt/model_best.pt"))

    # elif args.model_name == "local_prompt_global_prompt_singleFusion_embedding":
    #     model = Global_with_Local_UnetClassification_Global_Prompt_Embedding(out_channels = 3, local_prompt = True, CrossAttention= False)
    #     model.load_params(torch.load(args.pretrain, map_location='cpu')['net']) # load pretrain model
    #     word_embedding = torch.load("four_organ.pth")
    #     model.localbranch.organ_embedding.data = word_embedding.float()
    #     print('load word embedding')
    #     model.load_state_dict(torch.load("/research/d1/rshr/jxyu/projects/MICCAI2024_LocalGlobal/baseline_main/runs/local_prompt_global_prompt_singleFusion_embedding/model_259.pt"))
    
    elif args.model_name == "Global_Prompt":
        model = Global_Prompt(out_channels = 3)
        model.load_state_dict(torch.load("/research/d1/rshr/jxyu/projects/MICCAI2024_LocalGlobal/baseline_main/runs/Global_Prompt/model_best.pt"))

    else:
        raise RuntimeError("Do not support the method!")

    return model

def get_data_loader(args):

    local_file_root = "/research/d1/rshr/qxhu/PublicDataset/RSNA2023/preprocessed_data/our_methods"
    global_file_root = "/research/d1/rshr/qxhu/PublicDataset/RSNA2023/preprocessed_data/baseline_methods"
    labels_df = pd.read_csv('/research/d1/rshr/qxhu/PublicDataset/RSNA2023/preprocessed_data/label.csv', index_col="ID")

    val_samples   = []
    with open('/research/d1/rshr/jxyu/projects/MICCAI2024_LocalGlobal/data_preprocessing/test_data.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            sample = line.strip()
            val_samples.append(sample)

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

    val_ds = FusionDataset(local_npz_files=local_val_images, global_npz_files=global_val_images, labels=val_labels, local_img_transforms=local_val_img_transform, local_seg_transforms=None, global_img_transforms=global_val_img_transform, global_seg_transforms=None)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)


    return val_loader

def cls_score(pred, label):
    """
    pred:  [[1, 0, 0, 0], [0, 0, 0, 1]]
    label: [[0, 0, 1, 1], [1, 1, 1, 1]]
    """

    # case level
    case_pred = np.array([np.any(item) for item in pred], dtype=int)
    case_true = np.array([np.any(item) for item in label], dtype=int)
    tn, fp, fn, tp = confusion_matrix(case_true, case_pred).ravel()

    case_acc = np.sum(case_pred == case_true) / (len(case_true) + 1e-9)
    case_sen = tp / (tp + fn + 1e-9)
    case_prec = tp / (tp + fp + 1e-9)
    case_f1  = (2*case_prec*case_sen) / (case_prec + case_sen + 1e-9)


    # organ level
    organ_pred = np.array(pred).ravel()
    organ_true = np.array(label).ravel()
    tn, fp, fn, tp = confusion_matrix(organ_true, organ_pred).ravel()

    organ_acc = np.sum(organ_pred == organ_true) / (len(organ_true) + 1e-9)
    organ_sen = tp / (tp + fn + 1e-9)
    organ_prec = tp / (tp + fp + 1e-9)
    organ_f1   = (2*organ_prec*organ_sen) / (organ_prec + organ_sen + 1e-9)

    liver_pred = np.array([item[0] for item in pred])
    spleen_pred = np.array([item[1] for item in pred])
    kidney_pred = np.array([item[2] for item in pred])
    liver_true = np.array([item[0] for item in label])
    spleen_true = np.array([item[1] for item in label])
    kidney_true = np.array([item[2] for item in label])
    tn1, fp1, fn1, tp1 = confusion_matrix(liver_true, liver_pred).ravel()
    tn2, fp2, fn2, tp2 = confusion_matrix(spleen_true, spleen_pred).ravel()
    tn3, fp3, fn3, tp3 = confusion_matrix(kidney_true, kidney_pred).ravel()

    liver_acc = np.sum(liver_pred == liver_true) / (len(liver_true) + 1e-9)
    liver_sen = tp1 / (tp1 + fn1 + 1e-9)
    liver_prec = tp1 / (tp1 + fp1 + 1e-9)
    liver_f1   = (2*liver_prec*liver_sen) / (liver_prec + liver_sen + 1e-9)

    spleen_acc = np.sum(spleen_pred == spleen_true) / (len(spleen_true) + 1e-9)
    spleen_sen = tp2 / (tp2 + fn2 + 1e-9)
    spleen_prec = tp2 / (tp2 + fp2 + 1e-9)
    spleen_f1   = (2*spleen_prec*spleen_sen) / (spleen_prec + spleen_sen + 1e-9)

    kidney_acc = np.sum(kidney_pred == kidney_true) / (len(kidney_true) + 1e-9)
    kidney_sen = tp3 / (tp3 + fn3 + 1e-9)
    kidney_prec = tp3 / (tp3 + fp3 + 1e-9)
    kidney_f1   = (2*kidney_prec*kidney_sen) / (kidney_prec + kidney_sen + 1e-9)

    score_table = {
        "case_acc": case_acc,
        "case_sensitive": case_sen,
        "case_precision": case_prec,
        "case_f1": case_f1,
        "organ_acc": organ_acc,
        "organ_sensitive": organ_sen,
        "organ_precision": organ_prec,
        "organ_f1": organ_f1,

        "liver_acc": liver_acc,
        "liver_sensitive": liver_sen,
        "liver_precision": liver_prec,
        "liver_f1": liver_f1,
        "spleen_acc": spleen_acc,
        "spleen_sensitive": spleen_sen,
        "spleen_precision": spleen_prec,
        "spleen_f1": spleen_f1,
        "kidney_acc": kidney_acc,
        "kidney_sensitive": kidney_sen,
        "kidney_precision": kidney_prec,
        "kidney_f1": kidney_f1

    }
    score = 0.3 * case_sen + 0.2 * case_f1 + 0.1 * case_acc + 0.2 * organ_acc + 0.2 * organ_f1

    return score, score_table


def inference_cls(model, val_loader, args):

    device = args.device
    y_true = []
    y_pred = []

    for val_data in val_loader:
        val_liver        = val_data['liver'].to(device)
        val_spleen        = val_data['spleen'].to(device)
        val_left_kidney  = val_data['left_kidney'].to(device)
        val_right_kidney = val_data['right_kidney'].to(device)

        val_abdominal = val_data['abdominal'].to(device)

        val_labels  = val_data['label'].to(device)
        names   = val_data['name']
        if args.model_name == "Global_Prompt":
            val_preds, _ = model(val_abdominal)
        else:
            val_preds, _ = model(val_abdominal, val_liver, val_spleen, val_left_kidney, val_right_kidney)
        
        val_preds[val_preds >= 0] = 1
        val_preds[val_preds < 0] = 0
        val_labels = val_labels.cpu().numpy()
        val_preds  = val_preds.detach().cpu().numpy()
        y_true.extend(val_labels)
        y_pred.extend(val_preds)
        print("names", names, "val labels:",val_labels, "preds:", val_preds)
        

    score, score_table = cls_score(y_pred, y_true)
    metrics = (score + score_table['organ_f1']) * 0.5
    print("Metrics", metrics, 'Score', score, "Organ F1", score_table['organ_f1'])
    print(score_table)

def main():

    import argparse
    parser = argparse.ArgumentParser(description='medical segmentation contest')
    parser.add_argument('--model_name', default="", type=str)
    parser.add_argument('--resize_x', default=128, type=int)
    parser.add_argument('--resize_y', default=128, type=int)
    parser.add_argument('--resize_z', default=128, type=int)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    args.device = device
    args.pretrain = "./unet.pth"
    
    # Print All Config
    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')

    # loader
    val_loader = get_data_loader(args)
    model = _get_models(args)
    model.to(device)

    inference_cls(model, val_loader, args)

if __name__ == "__main__":

    main()


