import os
from collections import OrderedDict

import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data.folder_dataset import FolderDataset
from models.networks import VGGLoss
from models.networks.quantizer import VectorQuantizer
from models.pix2pix_model import Pix2PixModel
from options.train_encoder_options import TrainEncoderOptions
from util.visualizer import Visualizer
from util import html
from models.unet.unet import UNet

opt = TrainEncoderOptions().parse()

transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])
dataset = FolderDataset(opt.path, transforms, data_rep=1)
dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads),
        drop_last=False
    )

load_disc = opt.disc_loss_weight > 0
model = Pix2PixModel(opt, load_disc=load_disc)
model.eval()

vector_quantizer = VectorQuantizer(opt.label_nc)

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))


n_classes = opt.semantic_nc
# encoder should get [B, 3, H, W] and output [B, 1, H, W]
# maybe I can output [B, label_nc, H, W] and skip the scatter inside SPADEGenerator - but this will require converting the discretes to one hot (by argmax)
encoder = UNet(3, n_classes)
if opt.optimizer == "adam":
    optimizer = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(0.0, 0.9))
elif opt.optimizer == "sgd":
    optimizer = torch.optim.SGD(encoder.parameters(), lr=opt.lr)
else:
    raise NotImplementedError(opt.optimizer)

if opt.loss == "l1":
    criterion = torch.nn.L1Loss()
elif opt.loss == "l2":
    criterion = torch.nn.MSELoss()
elif opt.loss == "vgg":
    criterion = VGGLoss(opt.gpu_ids)
else:
    raise NotImplementedError(opt.loss)

n_epochs = opt.niter

if len(opt.gpu_ids) > 0:
    encoder.cuda()

for epoch in range(n_epochs):
    for i, img in enumerate(dataloader):
        if len(opt.gpu_ids) > 0:
            img = img.cuda()

        label_tensor = encoder(img)
        label_tensor_one_hot, _ = vector_quantizer(label_tensor)

        data_i = {'label': label_tensor_one_hot,
                  'instance': 0,
                  'image': img,
                  'path': None,
                  }

        loss = 0

        if load_disc:
            d_loss, generated = model(data_i, mode="generator")
            loss += d_loss["GAN"] * opt.disc_loss_weight
        else:
            generated = model(data_i, mode="eval")

        loss += criterion(generated, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

encoder.eval()

for i, img in enumerate(dataloader):
    print('process image... %s' % i)
    with torch.no_grad():
        if len(opt.gpu_ids) > 0:
            img = img.cuda()
        label_tensor = encoder(img)
        label_tensor_one_hot, min_encoding_indices = vector_quantizer(label_tensor)
        data_i = {'label': label_tensor_one_hot,
                  'instance': 0,
                  'image': img,
                  'path': None,
                  }
        generated = model(data_i, mode='eval')

        for b in range(generated.shape[0]):
            print(f'process image {i} {b}')
            print(f"labels for image {i}_{b} are: {torch.unique(min_encoding_indices)}")
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('synthesized_image', generated[b])])
            visualizer.save_images(webpage, visuals, [f"{i}_{b}.png"])

webpage.save()
