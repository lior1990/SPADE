"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import numpy as np
from collections import OrderedDict

import torch

from options.train_options import TrainOptions
import data
from util.cutmix import rand_bbox
from util.iter_counter import IterationCounter
from util.mix_itertools import mix_dataloaders
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)
opt.phase = "test"
val_dataloader = data.create_dataloader(opt)
opt.phase = "train"

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader) + len(val_dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, (data_i, train_or_val) in enumerate(mix_dataloaders(dataloader, val_dataloader), start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        fake_only = train_or_val == "val"

        if not fake_only:
            for _ in range(opt.n_times_cutmix):
                lam = np.random.uniform()
                size = data_i["image"].size()
                bbx1, bby1, bbx2, bby2 = rand_bbox(size, lam)
                batch_index_permutation = torch.randperm(size[0])

                data_i["image"][:, :, bbx1:bbx2, bby1:bby2] = data_i["image"][batch_index_permutation, :, bbx1:bbx2, bby1:bby2]
                data_i["label"][:, :, bbx1:bbx2, bby1:bby2] = data_i["label"][batch_index_permutation, :, bbx1:bbx2, bby1:bby2]

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i, fake_only)

        # train discriminator
        trainer.run_discriminator_one_step(data_i, fake_only)

        if opt.random_labels:
            random_labels_fake_only = True
            new_label_tensor = data_i["label"].long()  # create a clone so the loop won't make any label disappear
            rand_perm = torch.randperm(opt.semantic_nc)
            # assumption: the number of unique labels should be quite small so it's ok to iterate over it
            for orig_label in torch.unique(data_i["label"]):
                new_value = rand_perm[orig_label.long()]
                new_label_tensor[data_i["label"] == orig_label] = new_value

            data_i["label"] = new_label_tensor

            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i, random_labels_fake_only)

            # train discriminator
            trainer.run_discriminator_one_step(data_i, random_labels_fake_only)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('real_image', data_i['image'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
