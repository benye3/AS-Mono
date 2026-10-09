import torch
import datasets
import networks

def get_monovit_pretrained(load_type):
    depth = networks.DeepNet('mpvitnet')

    if load_type == 1024:
        depth_dict = torch.load("pretrained_weight/MonoViT_M_1024x320/depth.pth")

        new_dict = {}
        for k, v in depth_dict.items():
            name = k[7:]
            new_dict[name] = v
        depth.load_state_dict({k: v for k, v in new_dict.items() if k in depth.state_dict()})

    if load_type == 640: 
        mpvit_encoder_dict = torch.load("pretrained_weight/MonoViT_M_640x192/encoder.pth")
        model_dict = depth.encoder.state_dict()
        depth.encoder.load_state_dict({k: v for k, v in mpvit_encoder_dict.items() if k in model_dict})
 
        mpvit_decoder_path = "pretrained_weight/MonoViT_M_640x192/depth.pth"
        depth.decoder.load_state_dict(torch.load(mpvit_decoder_path))


    return depth


def get_monovit_pretrained_with_feature(load_type):
    depth = networks.DeepNet_with_Feature('mpvitnet')
    if load_type == 1024:
        depth_dict = torch.load("pretrained_weight/MonoViT_M_1024x320/depth.pth")

        new_dict = {}
        for k, v in depth_dict.items():
            name = k[7:]
            new_dict[name] = v
        depth.load_state_dict({k: v for k, v in new_dict.items() if k in depth.state_dict()})

    if load_type == 640: 
        mpvit_encoder_dict = torch.load("pretrained_weight/MonoViT_M_640x192/encoder.pth")
        model_dict = depth.encoder.state_dict()
        depth.encoder.load_state_dict({k: v for k, v in mpvit_encoder_dict.items() if k in model_dict})
 
        mpvit_decoder_path = "pretrained_weight/MonoViT_M_640x192/depth.pth"
        depth.decoder.load_state_dict(torch.load(mpvit_decoder_path))
 

    return depth
