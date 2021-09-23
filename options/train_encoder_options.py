"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .test_options import TestOptions


class TrainEncoderOptions(TestOptions):
    def initialize(self, parser):
        TestOptions.initialize(self, parser)
        parser.add_argument('--path', type=str, required=True)
        parser.add_argument('--niter', type=int, default=100)
        parser.add_argument('--loss', type=str, default="l1", help="l1/l2/vgg")
        parser.add_argument('--optimizer', type=str, default="adam", help='adam/sgd')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for optimizer')

        # disc
        parser.add_argument('--disc_loss_weight', type=float, default=0)
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        parser.add_argument('--n_layers_D', type=int, default=4, help='# layers in each discriminator')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')


        parser.set_defaults(dataset_mode="ade_like")
        self.isTrain = False
        return parser
