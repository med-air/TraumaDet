import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.for_resnet.resnet import generate_model
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D
from einops import rearrange, repeat, reduce
from models.transformer_decoder import TransformerDecoder,TransformerDecoderLayer

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth+1)),act)
        layer2 = LUConv(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)),act)
    else:
        layer1 = LUConv(in_channel, 32*(2**depth),act)
        layer2 = LUConv(32*(2**depth), 32*(2**depth)*2,act)

    return nn.Sequential(layer1,layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channel,depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth,act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans+ outChans//2,depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out

class UnetEncoder(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, act='relu'):
        super(UnetEncoder, self).__init__()

        self.down_tr64 = DownTransition(1,0,act)
        self.down_tr128 = DownTransition(64,1,act)
        self.down_tr256 = DownTransition(128,2,act)
        self.down_tr512 = DownTransition(256,3,act)

    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128,self.skip_out128 = self.down_tr128(self.out64)
        self.out256,self.skip_out256 = self.down_tr256(self.out128)
        self.out512,self.skip_out512 = self.down_tr512(self.out256)

        return self.out512


class SingleAttention(nn.Module):
    def __init__(self, vis_dim=512) -> None:
        super(SingleAttention, self).__init__()

        self.vis_dim = vis_dim
        self.decoder_layer_global = TransformerDecoderLayer(d_model = self.vis_dim, nhead = 8, normalize_before=True)
        self.decoder_norm_global = nn.LayerNorm(self.vis_dim)
        self.transformer_decoder_global = TransformerDecoder(decoder_layer = self.decoder_layer_global, num_layers = 6, norm=self.decoder_norm_global)
    
    def global_query_local_key_value(self, local_feature, global_feature):

        # global as queries
        # global output [2, 512] = > [2, 1, 512]   local output [2, 512] => [2, 512, 1]
        
        B = global_feature.shape[0]
        global_feature = torch.reshape(global_feature, (B, -1, self.vis_dim))
        local_feature = torch.reshape(local_feature, (B, self.vis_dim, 1))
        
        pos_embedding = PositionalEncoding1D(self.vis_dim)(torch.zeros(1, 1, self.vis_dim)) # b h/p w/p d/p dim
        pos_embedding = rearrange(pos_embedding, 'b h c -> h b c')   # n b dim

        pos = pos_embedding.to(local_feature.device)    # (H/P W/P D/P) B Dim
        image_embedding = rearrange(local_feature, 'b dim h -> h b dim') # (H/P W/P D/P) B Dim
        queries = rearrange(global_feature, 'b n dim -> n b dim') # N B Dim

        global_fused,_ = self.transformer_decoder_global(queries, image_embedding, pos = pos) # N B Dim
        global_fused = rearrange(global_fused, 'n b dim -> (b n) dim') # (B N) Dim

        return global_fused
    
    
    def forward(self, local_feature, global_feature):

        global_fused = self.global_query_local_key_value(local_feature, global_feature)
        align_feature = global_feature.clone()
        return global_fused, align_feature

class DoubleAttention(nn.Module):
    def __init__(self, vis_dim=512) -> None:
        super(DoubleAttention, self).__init__()

        self.vis_dim = vis_dim

        self.decoder_layer_global = TransformerDecoderLayer(d_model = self.vis_dim, nhead = 8, normalize_before=True)
        self.decoder_norm_global = nn.LayerNorm(self.vis_dim)
        self.transformer_decoder_global = TransformerDecoder(decoder_layer = self.decoder_layer_global, num_layers = 6, norm=self.decoder_norm_global)

        self.decoder_layer_local = TransformerDecoderLayer(d_model = self.vis_dim, nhead = 8, normalize_before=True)
        self.decoder_norm_local = nn.LayerNorm(self.vis_dim)
        self.transformer_decoder_local = TransformerDecoder(decoder_layer = self.decoder_layer_local, num_layers = 6, norm=self.decoder_norm_local)
    
    def global_query_local_key_value(self, local_feature, global_feature):

        # global as queries
        # global output [2, 512] = > [2, 1, 512]   local output [2, 512] => [2, 512, 1]
        
        B = global_feature.shape[0]
        global_feature = torch.reshape(global_feature, (B, -1, self.vis_dim))
        local_feature = torch.reshape(local_feature, (B, self.vis_dim, 1))
        
        pos_embedding = PositionalEncoding1D(self.vis_dim)(torch.zeros(1, 1, self.vis_dim)) # b h/p w/p d/p dim
        pos_embedding = rearrange(pos_embedding, 'b h c -> h b c')   # n b dim

        pos = pos_embedding.to(local_feature.device)    # (H/P W/P D/P) B Dim
        image_embedding = rearrange(local_feature, 'b dim h -> h b dim') # (H/P W/P D/P) B Dim
        queries = rearrange(global_feature, 'b n dim -> n b dim') # N B Dim

        global_fused,_ = self.transformer_decoder_global(queries, image_embedding, pos = pos) # N B Dim
        global_fused = rearrange(global_fused, 'n b dim -> (b n) dim') # (B N) Dim

        return global_fused
    
    def local_query_global_key_value(self, local_feature, global_feature):

        ## local as queries
        ## local output [2, 512] => [2, 1, 512]   global output [2 512] => [2, 512, 1]

        B = local_feature.shape[0]
        local_feature = torch.reshape(local_feature, (B, -1, self.vis_dim))
        global_feature = torch.reshape(global_feature, (B, self.vis_dim, 1))

        pos_embedding = PositionalEncoding1D(self.vis_dim)(torch.zeros(1, 1, self.vis_dim)) # b h/p w/p d/p dim
        pos_embedding = rearrange(pos_embedding, 'b h c -> h b c')   # n b dim
        
        pos = pos_embedding.to(global_feature.device)    # (H/P W/P D/P) B Dim
        image_embedding = rearrange(global_feature, 'b dim h -> h b dim') # (H/P W/P D/P) B Dim
        queries = rearrange(local_feature, 'b n dim -> n b dim') # N B Dim

        local_fused,_ = self.transformer_decoder_local(queries, image_embedding, pos = pos) # N B Dim
        local_fused = rearrange(local_fused, 'n b dim -> (b n) dim') # (B N) Dim

        return local_fused
 
    def forward(self, local_feature, global_feature):

        global_fused = self.global_query_local_key_value(local_feature, global_feature)
        local_fused = self.local_query_global_key_value(local_feature, global_feature)

        fusion_feature = torch.cat((global_fused, local_fused), dim=1)
        align_feature = global_fused + local_fused

        return fusion_feature, align_feature

class Global_with_Local_noFusion(nn.Module):

    def __init__(self, out_channels = 3, local_prompt = True) -> None:
         super(Global_with_Local_noFusion, self).__init__()

         self.localbranch = Local_Branch_UnetClassification(out_channels, local_prompt)
         self.globalbranch = generate_model(model_depth=50, n_classes=512, input_W=128, input_H=128, input_D=128)
         self.cls_head = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=out_channels)
         )
    
    def load_params(self, model_dict):
        self.localbranch.load_params(model_dict)


    def forward(self, global_data, liver, spleen, left_kidney, right_kidney): 
        localfeature = self.localbranch(liver, spleen, left_kidney, right_kidney)
        globalfeature = self.globalbranch(global_data)
        catfeature = torch.cat((localfeature, globalfeature), dim=1)
        out = self.cls_head(catfeature)

        return out, catfeature


class Global_with_Local_UnetClassification(nn.Module):

    def __init__(self, out_channels = 3, local_prompt = True, CrossAttention = True) -> None:
         super(Global_with_Local_UnetClassification, self).__init__()

         self.CrossAttention = CrossAttention
         self.localbranch = Local_Branch_UnetClassification(out_channels, local_prompt)
         self.globalbranch = generate_model(model_depth=50, n_classes=512, input_W=128, input_H=128, input_D=128)
         self.Fusionmodule = DoubleAttention(vis_dim=512)
         self.SingleFusionModule = SingleAttention(vis_dim=512)
         
         self.cls_head_crossfusion = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=out_channels)
         )
         self.cls_head_singlefusion = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=out_channels)
         )
    
    def load_params(self, model_dict):
        self.localbranch.load_params(model_dict)


    def forward(self, global_data, liver, spleen, left_kidney, right_kidney): 
        localfeature = self.localbranch(liver, spleen, left_kidney, right_kidney)
        globalfeature = self.globalbranch(global_data)

        if self.CrossAttention:
            fusionfeature, alignfeature = self.Fusionmodule(localfeature, globalfeature)
            out = self.cls_head_crossfusion(fusionfeature)

            return out, alignfeature

        else:
            fusionfeature, alignfeature = self.SingleFusionModule(localfeature, globalfeature)
            out = self.cls_head_singlefusion(fusionfeature)

            return out, alignfeature

    

class Local_Branch_UnetClassification(nn.Module):
    def __init__(self, out_channels = 3, local_prompt = True) -> None:    ## change to 3
        super(Local_Branch_UnetClassification, self).__init__()

        self.local_prompt = local_prompt

        self.encoder = UnetEncoder()
        self.spleen_encoder = UnetEncoder()
        self.liver_encoder   = UnetEncoder()

        # self.precls_conv = nn.Sequential(
        #     nn.GroupNorm(16, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(64, 8, kernel_size=1)
        # )

        self.GAP = nn.Sequential(
                nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(512, 128, kernel_size=1, stride=1, padding=0),
                nn.Flatten()
        )
                
        # self.feature_fusion = 1

        self.cls_head = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=out_channels)
        )

        if self.local_prompt:
            self.register_buffer('organ_embedding', torch.randn(4, 512))
            self.text_to_vision = nn.Linear(512, 128)

        self.class_num = out_channels


    def load_params(self, model_dict):
        store_dict = self.encoder.state_dict()
        for key in model_dict.keys():
            if "down_tr" in key:
                store_dict[key.replace("module.backbone.", "")] = model_dict[key]
        self.encoder.load_state_dict(store_dict)
        self.spleen_encoder.load_state_dict(store_dict)
        self.liver_encoder.load_state_dict(store_dict)

        print('Use pretrained weights')


    def forward(self, liver, spleen, left_kidney, right_kidney):
        
        liver_feature  = self.liver_encoder(liver)    
        liver_feature  = self.GAP(liver_feature)
        
        spleen_feature = self.spleen_encoder(spleen)
        spleen_feature = self.GAP(spleen_feature)
        
        left_feature   = self.encoder(left_kidney)
        left_feature   = self.GAP(left_feature)

        right_feature = self.encoder(right_kidney)
        right_feature = self.GAP(right_feature)

        B = liver_feature.shape[0]

        liver_feature = liver_feature.unsqueeze(1)
        spleen_feature = spleen_feature.unsqueeze(1)
        left_feature = left_feature.unsqueeze(1)
        right_feature = right_feature.unsqueeze(1)

        all_feature = torch.cat([liver_feature, spleen_feature, left_feature, right_feature], 1)
        
        if self.local_prompt:
            task_encoding = F.relu(self.text_to_vision(self.organ_embedding))
            task_encoding = task_encoding.unsqueeze(0).repeat(B,1,1)
            feature = torch.mul(all_feature, task_encoding)
        else:
            feature = all_feature

        feature = feature.view(B,512)

        return feature

class Global_Prompt(nn.Module):
    def __init__(self, out_channels = 3) -> None:
        super(Global_Prompt, self).__init__()
        self.globalbranch = generate_model(model_depth=50, n_classes=512, input_W=128, input_H=128, input_D=128)
        self.cls_head = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=out_channels)
        )

    def forward(self, global_data): 
        globalfeature = self.globalbranch(global_data)
        align_feature = globalfeature.clone()

        out = self.cls_head(globalfeature)

        return out, align_feature


class Global_with_Local_Prompt_Fusion_UnetClassification(nn.Module):

    def __init__(self, out_channels = 3, local_prompt = True) -> None:
         super(Global_with_Local_Prompt_Fusion_UnetClassification, self).__init__()

         self.localbranch = Local_Branch_Prompt_Fused(out_channels, local_prompt)
         self.globalbranch = generate_model(model_depth=50, n_classes=512, input_W=128, input_H=128, input_D=128)
         self.Fusionmodule = DoubleAttention(vis_dim=512)
         self.cls_head = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=out_channels)
         )
    
    def load_params(self, model_dict):
        self.localbranch.load_params(model_dict)


    def forward(self, global_data, liver, spleen, left_kidney, right_kidney): 
        localfeature = self.localbranch(liver, spleen, left_kidney, right_kidney)
        globalfeature = self.globalbranch(global_data)
        fusionfeature, alignfeature = self.Fusionmodule(localfeature, globalfeature)
        out = self.cls_head(fusionfeature)

        return out, alignfeature

class OrganFusion(nn.Module):
    def __init__(self, vis_dim=128) -> None:
        super(OrganFusion, self).__init__()

        self.vis_dim = vis_dim

        self.decoder_layer_liver = TransformerDecoderLayer(d_model = self.vis_dim, nhead = 8, normalize_before=True)
        self.decoder_norm_liver = nn.LayerNorm(self.vis_dim)
        self.transformer_decoder_liver = TransformerDecoder(decoder_layer = self.decoder_layer_liver, num_layers = 6, norm=self.decoder_norm_liver)

        self.decoder_layer_spleen = TransformerDecoderLayer(d_model = self.vis_dim, nhead = 8, normalize_before=True)
        self.decoder_norm_spleen = nn.LayerNorm(self.vis_dim)
        self.transformer_decoder_spleen = TransformerDecoder(decoder_layer = self.decoder_layer_spleen, num_layers = 6, norm=self.decoder_norm_spleen)
        
        self.decoder_layer_lkidney = TransformerDecoderLayer(d_model = self.vis_dim, nhead = 8, normalize_before=True)
        self.decoder_norm_lkidney = nn.LayerNorm(self.vis_dim)
        self.transformer_decoder_lkidney = TransformerDecoder(decoder_layer = self.decoder_layer_spleen, num_layers = 6, norm=self.decoder_norm_spleen)

        self.decoder_layer_rkidney = TransformerDecoderLayer(d_model = self.vis_dim, nhead = 8, normalize_before=True)
        self.decoder_norm_rkidney = nn.LayerNorm(self.vis_dim)
        self.transformer_decoder_rkidney = TransformerDecoder(decoder_layer = self.decoder_layer_spleen, num_layers = 6, norm=self.decoder_norm_spleen)

    def Fusion(self, vision_feature, text_feature):

        B = text_feature.shape[0]
        text_feature = torch.reshape(text_feature, (B, -1, self.vis_dim))
        vision_feature = torch.reshape(vision_feature, (B, self.vis_dim, 1))
        
        pos_embedding = PositionalEncoding1D(self.vis_dim)(torch.zeros(1, 1, self.vis_dim)) # b h/p w/p d/p dim
        pos_embedding = rearrange(pos_embedding, 'b h c -> h b c')   # n b dim

        pos = pos_embedding.to(text_feature.device)    # (H/P W/P D/P) B Dim
        image_embedding = rearrange(vision_feature, 'b dim h -> h b dim') # (H/P W/P D/P) B Dim
        queries = rearrange(text_feature, 'b n dim -> n b dim') # N B Dim

        # global_fused,_ = self.transformer_decoder_global(queries, image_embedding, pos = pos) # N B Dim
        # global_fused = rearrange(global_fused, 'n b dim -> (b n) dim') # (B N) Dim

        return queries, image_embedding, pos
 
    def forward(self, vision_feature, text_feature):

        liver_queries, liver_image_embedding, liver_pos = self.Fusion(vision_feature[:,0,:], text_feature[:,0,:])
        spleen_queries, spleen_image_embedding, spleen_pos = self.Fusion(vision_feature[:,1,:], text_feature[:,1,:])
        lkidney_queries, lkidney_image_embedding, lkidney_pos = self.Fusion(vision_feature[:,2,:], text_feature[:,2,:])
        rkidney_queries, rkidney_image_embedding, rkidney_pos = self.Fusion(vision_feature[:,3,:], text_feature[:,3,:])


        liver_fused,_ = self.transformer_decoder_liver(liver_queries, liver_image_embedding, pos = liver_pos) # N B Dim
        liver_fused = rearrange(liver_fused, 'n b dim -> b (n dim)') # (B N) Dim

        spleen_fused,_ = self.transformer_decoder_spleen(spleen_queries, spleen_image_embedding, pos = spleen_pos) # N B Dim
        spleen_fused = rearrange(spleen_fused, 'n b dim -> b (n dim)') # (B N) Dim

        lkidney_fused,_ = self.transformer_decoder_lkidney(lkidney_queries, lkidney_image_embedding, pos = lkidney_pos) # N B Dim
        lkidney_fused = rearrange(lkidney_fused, 'n b dim -> b (n dim)') # (B N) Dim

        rkidney_fused,_ = self.transformer_decoder_rkidney(rkidney_queries, rkidney_image_embedding, pos = rkidney_pos) # N B Dim
        rkidney_fused = rearrange(rkidney_fused, 'n b dim -> b (n dim)') # (B N) Dim

        fusion_feature = torch.cat((liver_fused, spleen_fused, lkidney_fused, rkidney_fused), dim=1)

        return fusion_feature
    

class Local_Branch_Prompt_Fused(nn.Module):
    def __init__(self, out_channels = 3, local_prompt = True) -> None:    ## change to 3
        super(Local_Branch_Prompt_Fused, self).__init__()

        self.local_prompt = local_prompt

        self.encoder = UnetEncoder()
        self.spleen_encoder = UnetEncoder()
        self.liver_encoder   = UnetEncoder()
        self.organ_fusion = OrganFusion()
        ## liver_fusion = 

        # self.precls_conv = nn.Sequential(
        #     nn.GroupNorm(16, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(64, 8, kernel_size=1)
        # )

        self.GAP = nn.Sequential(
                nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(512, 128, kernel_size=1, stride=1, padding=0),
                nn.Flatten()
        )
                
        # self.feature_fusion = 1

        self.cls_head = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=out_channels)
        )

        if self.local_prompt:
            self.register_buffer('organ_embedding', torch.randn(4, 512))
            self.text_to_vision = nn.Linear(512, 128)

        self.class_num = out_channels


    def load_params(self, model_dict):
        store_dict = self.encoder.state_dict()
        for key in model_dict.keys():
            if "down_tr" in key:
                store_dict[key.replace("module.backbone.", "")] = model_dict[key]
        self.encoder.load_state_dict(store_dict)
        self.spleen_encoder.load_state_dict(store_dict)
        self.liver_encoder.load_state_dict(store_dict)

        print('Use pretrained weights')


    def forward(self, liver, spleen, left_kidney, right_kidney):
        
        liver_feature  = self.liver_encoder(liver)    
        liver_feature  = self.GAP(liver_feature)
        
        spleen_feature = self.spleen_encoder(spleen)
        spleen_feature = self.GAP(spleen_feature)
        
        left_feature   = self.encoder(left_kidney)
        left_feature   = self.GAP(left_feature)

        right_feature = self.encoder(right_kidney)
        right_feature = self.GAP(right_feature)

        B = liver_feature.shape[0]

        liver_feature = liver_feature.unsqueeze(1)
        spleen_feature = spleen_feature.unsqueeze(1)
        left_feature = left_feature.unsqueeze(1)
        right_feature = right_feature.unsqueeze(1)

        all_feature = torch.cat([liver_feature, spleen_feature, left_feature, right_feature], 1)
        
        if self.local_prompt:
            task_encoding = F.relu(self.text_to_vision(self.organ_embedding))
            task_encoding = task_encoding.unsqueeze(0).repeat(B,1,1)
            
            feature = self.organ_fusion(all_feature, task_encoding)

        else:
            feature = all_feature

        feature = feature.view(B,512)

        return feature

class OrganFusion2(nn.Module):
    def __init__(self, vis_dim=128) -> None:
        super(OrganFusion2, self).__init__()

        self.vis_dim = vis_dim
        self.decoder_layer = TransformerDecoderLayer(d_model = vis_dim, nhead = 8, normalize_before=True)
        self.decoder_norm = nn.LayerNorm(vis_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer = self.decoder_layer, num_layers = 1, norm=self.decoder_norm)

        

    def Fusion(self, vision_feature, text_feature):

        B = text_feature.shape[0]
        text_feature = torch.reshape(text_feature, (B, -1, self.vis_dim))
        vision_feature = torch.reshape(vision_feature, (B, self.vis_dim, -1))
        
        pos_embedding = PositionalEncoding1D(self.vis_dim)(torch.zeros(1, 1, self.vis_dim)) # b h/p w/p d/p dim
        pos_embedding = rearrange(pos_embedding, 'b h c -> h b c')   # n b dim

        pos = pos_embedding.to(text_feature.device)    # (H/P W/P D/P) B Dim
        image_embedding = rearrange(vision_feature, 'b dim h -> h b dim') # (H/P W/P D/P) B Dim
        queries = rearrange(text_feature, 'b n dim -> n b dim') # N B Dim

        global_fused,_ = self.transformer_decoder(queries, image_embedding, pos = pos) # N B Dim
        global_fused = rearrange(global_fused, 'n b dim -> b (n dim)') # (B N) Dim

        return global_fused
 
    def forward(self, vision_feature, text_feature):

        fusion_feature = self.Fusion(vision_feature, text_feature)

        return fusion_feature


class Local_Branch_Prompt_Fused2(nn.Module):
    def __init__(self, out_channels = 3, local_prompt = True) -> None:    ## change to 3
        super(Local_Branch_Prompt_Fused2, self).__init__()

        self.local_prompt = local_prompt

        self.encoder = UnetEncoder()
        self.spleen_encoder = UnetEncoder()
        self.liver_encoder   = UnetEncoder()
        self.organ_fusion = OrganFusion2()
        ## liver_fusion = 

        # self.precls_conv = nn.Sequential(
        #     nn.GroupNorm(16, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(64, 8, kernel_size=1)
        # )

        self.GAP = nn.Sequential(
                nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(512, 128, kernel_size=1, stride=1, padding=0),
                nn.Flatten()
        )
                
        # self.feature_fusion = 1

        self.cls_head = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=out_channels)
        )

        if self.local_prompt:
            self.register_buffer('organ_embedding', torch.randn(4, 512))
            self.text_to_vision = nn.Linear(512, 128)

        self.class_num = out_channels


    def load_params(self, model_dict):
        store_dict = self.encoder.state_dict()
        for key in model_dict.keys():
            if "down_tr" in key:
                store_dict[key.replace("module.backbone.", "")] = model_dict[key]
        self.encoder.load_state_dict(store_dict)
        self.spleen_encoder.load_state_dict(store_dict)
        self.liver_encoder.load_state_dict(store_dict)

        print('Use pretrained weights')


    def forward(self, liver, spleen, left_kidney, right_kidney):
        
        liver_feature  = self.liver_encoder(liver)    
        liver_feature  = self.GAP(liver_feature)
        
        spleen_feature = self.spleen_encoder(spleen)
        spleen_feature = self.GAP(spleen_feature)
        
        left_feature   = self.encoder(left_kidney)
        left_feature   = self.GAP(left_feature)

        right_feature = self.encoder(right_kidney)
        right_feature = self.GAP(right_feature)

        B = liver_feature.shape[0]

        liver_feature = liver_feature.unsqueeze(1)
        spleen_feature = spleen_feature.unsqueeze(1)
        left_feature = left_feature.unsqueeze(1)
        right_feature = right_feature.unsqueeze(1)

        all_feature = torch.cat([liver_feature, spleen_feature, left_feature, right_feature], 1)
        
        if self.local_prompt:
            task_encoding = F.relu(self.text_to_vision(self.organ_embedding))
            task_encoding = task_encoding.unsqueeze(0).repeat(B,1,1)
            
            feature = self.organ_fusion(all_feature, task_encoding)

        else:
            feature = all_feature

        feature = feature.view(B,512)

        pred = self.cls_head(feature)

        return pred


class Global_with_Local_UnetClassification_Global_Prompt_Embedding(nn.Module):


    def __init__(self, out_channels = 3, local_prompt = True, CrossAttention = True) -> None:
        super(Global_with_Local_UnetClassification_Global_Prompt_Embedding, self).__init__()

        self.CrossAttention = CrossAttention
        self.localbranch = Local_Branch_UnetClassification(out_channels, local_prompt)
        self.globalbranch = generate_model(model_depth=50, n_classes=512, input_W=128, input_H=128, input_D=128)
        self.Fusionmodule = DoubleAttention(vis_dim=512)
        self.SingleFusionModule = SingleAttention(vis_dim=512)
        
        self.cls_head_crossfusion = nn.Sequential(
        nn.Linear(in_features=1024, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=out_channels)
        )
        self.cls_head_singlefusion = nn.Sequential(
        nn.Linear(in_features=512, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=out_channels)
        )

        # B = 4 
        self.register_buffer('trauma_embedding', torch.randn(2, 3*512))
        self.text_to_vision = nn.Linear(3*512, 512)
    
    def load_params(self, model_dict):
        self.localbranch.load_params(model_dict)


    def forward(self, global_data, liver, spleen, left_kidney, right_kidney, isTraining=False): 
        localfeature = self.localbranch(liver, spleen, left_kidney, right_kidney)
        globalfeature = self.globalbranch(global_data)

        if self.CrossAttention:
            fusionfeature, alignfeature = self.Fusionmodule(localfeature, globalfeature)
            out = self.cls_head_crossfusion(fusionfeature)

            return out, alignfeature

        else:
            fusionfeature, alignfeature = self.SingleFusionModule(localfeature, globalfeature)
            if isTraining:
                out = self.cls_head_singlefusion(fusionfeature)
                task_encoding = F.relu(self.text_to_vision(self.trauma_embedding))
                weights_feature = torch.mul(fusionfeature, task_encoding)
                return out, alignfeature, weights_feature
                
            else:
                out = self.cls_head_singlefusion(fusionfeature)
                return out, alignfeature


if __name__ == "__main__":
    liver = torch.ones((2, 1, 96, 96, 96))
    spleen = torch.ones((2, 1, 96, 96, 96))
    left_kidney = torch.ones((2, 1, 96, 96, 96))
    right_kidney = torch.ones((2, 1, 96, 96, 96))
    global_data = torch.ones((2, 1, 128, 128, 128))
    model = Global_with_Local_UnetClassification(out_channels = 3, local_prompt=False)
    print(model)
    # load pretrain model
    pretrain = "../unet.pth"
    model.load_params(torch.load(pretrain, map_location='cpu')['net'])
    pred, align = model(global_data, liver, spleen, left_kidney, right_kidney)
    # pred = model(liver, spleen, left_kidney, right_kidney)
    # import numpy as np
    # pred = pred.detach().numpy()
    # pred = np.reshape(pred, (2, 1, 256, 288))
    print(pred.shape, align.shape)
    # print(pred)
    print(pred.shape)

    pred[pred >= 0] = 1
    pred[pred < 0] = 0

    print(pred)
