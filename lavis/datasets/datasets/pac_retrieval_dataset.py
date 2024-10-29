import io
import os
from collections import OrderedDict
from copy import deepcopy
from PIL import Image
from mmengine.fileio import FileClient
from lavis.datasets.datasets.retrieval_datasets import (
    VideoRetrievalDataset,
    VideoRetrievalEvalDataset,
)
from lavis.datasets.datasets.base_dataset import BaseDataset
import re


class PACVideoRetrievalDataset(VideoRetrievalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of videos.
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def set_video_captions(self, overall_caps, segment_caps):
        self.overall_caps = overall_caps
        self.segment_caps = segment_caps

    def __getitem__(self, index):
        ann = self.annotation[index]
        if self.file_client is not None:
            vid_obj = io.BytesIO(self.file_client.get(ann["video"]))
        else:
            vid_obj = io.BytesIO(
                open(self.video_dict[ann["video"]], 'rb').read())

        video = self.vis_processor(vid_obj)
        caption = self.text_processor(ann["caption"])
        overall_caption = self.text_processor(self.overall_caps[ann["video"]])
        segment_captions = [self.text_processor(
            cap) for cap in self.segment_caps[ann["video"]]]

        return {
            "video": video,
            "text_input": caption,
            "video_id": self.img_ids[ann["video"]],
            "overall_caption": overall_caption,
            "segment_captions": segment_captions
        }
