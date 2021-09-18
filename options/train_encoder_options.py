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
        parser.add_argument('--loss', type=str, default="l1", help="l1/l2")

        parser.set_defaults(dataset_mode="ade_like")
        self.isTrain = False
        return parser
