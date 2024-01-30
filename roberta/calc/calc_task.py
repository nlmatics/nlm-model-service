# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
from collections import OrderedDict

import numpy as np
from fairseq import models
from fairseq.data import ConcatSentencesDataset
from fairseq.data import data_utils
from fairseq.data import Dictionary
from fairseq.data import IdDataset
from fairseq.data import NestedDictionaryDataset
from fairseq.data import NumelDataset
from fairseq.data import NumSamplesDataset
from fairseq.data import PrependTokenDataset
from fairseq.data import RawLabelDataset
from fairseq.data import RightPadDataset
from fairseq.data import RollDataset
from fairseq.data import SortDataset
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask
from fairseq.tasks import register_task

logger = logging.getLogger(__name__)


@register_task("calculation_task")
class CalculationTask(LegacyFairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            metavar="FILE",
            help="file prefix for data",
        )
        parser.add_argument(
            "--num-classes",
            type=int,
            default=-1,
            help="number of classes or regression targets",
        )
        parser.add_argument(
            "--init-token",
            type=int,
            default=None,
            help="add token at the beginning of each batch item",
        )
        parser.add_argument(
            "--separator-token",
            type=int,
            default=None,
            help="add separator token between inputs",
        )
        parser.add_argument("--regression-target", action="store_true", default=False)
        parser.add_argument("--no-shuffle", action="store_true", default=False)
        parser.add_argument(
            "--shorten-method",
            default="none",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )
        parser.add_argument(
            "--shorten-data-split-list",
            default="",
            help="comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)',
        )
        parser.add_argument(
            "--add-prev-output-tokens",
            action="store_true",
            default=False,
            help="add prev_output_tokens to sample, used for encoder-decoder arch",
        )

    def __init__(self, args, data_dictionary, label_dictionary):
        super().__init__(args)
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary
        if not hasattr(args, "max_positions"):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "input0", "dict.txt"),
            source=True,
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        label_dict = None
        #         if not args.regression_target:
        #             # load label dictionary
        #             label_dict = cls.load_dictionary(
        #                 args,
        #                 os.path.join(args.data, 'label1', 'dict.txt'),
        #                 source=False,
        #             )
        #             logger.info('[label] dictionary: {} types'.format(len(label_dict)))
        #         else:
        label_dict = data_dict
        return CalculationTask(args, data_dict, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(dataset_type, split):
            return os.path.join(self.args.data, dataset_type, split)

        def make_dataset(dataset_type, dictionary):
            split_path = get_path(dataset_type, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            return dataset

        input0 = make_dataset("input0", self.source_dictionary)
        assert input0 is not None, "could not find dataset: {}".format(
            get_path("input0", split),
        )
        if self.args.init_token is not None:
            input0 = PrependTokenDataset(input0, self.args.init_token)

        src_tokens = input0
        # input1 is not used for coref ########

        input1 = make_dataset("input1", self.source_dictionary)

        if input1 is None:
            src_tokens = input0
        else:
            if self.args.separator_token is not None:
                input1 = PrependTokenDataset(input1, self.args.separator_token)
            src_tokens = ConcatSentencesDataset(input0, input1)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        src_tokens = maybe_shorten_dataset(
            src_tokens,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.max_positions,
            self.args.seed,
        )

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

        if self.args.add_prev_output_tokens:
            prev_tokens_dataset = RightPadDataset(
                RollDataset(src_tokens, 1),
                pad_idx=self.dictionary.pad(),
            )
            dataset["net_input"].update(
                prev_output_tokens=prev_tokens_dataset,
            )

        def index_vector(start_line, end_line):
            vector = [0] * 7500
            start_indices = [int(i) for i in start_line.split()]
            end_indices = [int(i) for i in end_line.split()]
            for start, end in zip(start_indices, end_indices):
                for i in range(start, end):
                    vector[i] = 1
            return vector

        # change this, two for loops range 2
        start_path = "{}.label".format(get_path("span_start", split))
        end_path = "{}.label".format(get_path("span_end", split))
        dataset.update(
            {
                "span_indices": RawLabelDataset(
                    [
                        index_vector(start_line.strip(), end_line.strip())
                        for start_line, end_line in zip(
                            open(start_path).readlines(),
                            open(end_path).readlines(),
                        )
                    ],
                ),
            },
        )

        label_path = "{}.label".format(get_path("op", split))
        dataset.update(
            {
                "op_label": RawLabelDataset(
                    [
                        int(line.strip())
                        for i, line in enumerate(open(label_path).readlines())
                    ],
                ),
            },
        )

        def _flatten(dico, prefix=None):
            """Flatten a nested dictionary."""
            new_dico = OrderedDict()
            if isinstance(dico, dict):
                prefix = prefix + "." if prefix is not None else ""
                for k, v in dico.items():
                    if v is None:
                        continue
                    new_dico.update(_flatten(v, prefix + k))
            elif isinstance(dico, list):
                for i, v in enumerate(dico):
                    new_dico.update(_flatten(v, prefix + ".[" + str(i) + "]"))
            else:
                new_dico = OrderedDict({prefix: dico})
            return new_dico

        defn = _flatten(dataset)

        first = None
        for v in defn.values():

            #             if not isinstance(v, (FairseqDataset, torch.utils.data.Dataset, )):
            #                 raise ValueError('Expected Dataset but found: {}'.format(v.__class__))
            first = first or v
            if len(v) > 0:
                logger.error(f"len(v): {len(v)}, len(first): {len(first)}")
                assert len(v) == len(first), "dataset lengths must match"

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {} with #samples: {}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        model = models.build_model(args, self)
        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
