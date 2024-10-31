"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.retrieval_datasets import (
    VideoRetrievalDataset,
    VideoRetrievalEvalDataset,
)
from lavis.datasets.datasets.pac_retrieval_dataset import PACVideoRetrievalDataset

from lavis.common.registry import registry


def load_videos(file_path):
    if file_path.endswith(".jsonl"):
        data_dict = {}
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                data_dict[data["video_id"]] = data["outputs"]
        f.close()
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            data_dict = json.load(f)
        for k, v in data_dict.items():
            data_dict[k] = v["outputs"]
    else:
        raise NotImplementedError
    return data_dict


@registry.register_builder("pac_msrvtt_retrieval")
class PACMSRVTTRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = PACVideoRetrievalDataset
    eval_dataset_cls = VideoRetrievalEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/pac_ret.yaml",
    }

    def build(self):
        datasets = super().build()
        overall_caps = load_videos(self.config.get("overall_cap"))
        segment_caps = load_videos(self.config.get("segment_cap"))
        datasets["train"].set_video_captions(overall_caps, segment_caps)
        datasets["val"].set_video_captions(overall_caps, segment_caps)
        datasets["test"].set_video_captions(overall_caps, segment_caps)
        return datasets


@registry.register_builder("msrvtt_retrieval")
class MSRVTTRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoRetrievalDataset
    eval_dataset_cls = VideoRetrievalEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_ret.yaml",
        'pac': 'configs/datasets/msrvtt/pac_ret.yaml',
    }


@registry.register_builder("didemo_retrieval")
class DiDeMoRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoRetrievalDataset
    eval_dataset_cls = VideoRetrievalEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/didemo/defaults_ret.yaml"}


@registry.register_builder("msvd_retrieval")
class MSRVTTRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoRetrievalDataset
    eval_dataset_cls = VideoRetrievalEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_ret.yaml"}


@registry.register_builder("anet_retrieval")
class ANetRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoRetrievalDataset
    eval_dataset_cls = VideoRetrievalEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/anet/defaults_ret.yaml"}
