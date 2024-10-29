"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import io
import os
from collections import OrderedDict
from copy import deepcopy
from PIL import Image
from mmengine.fileio import FileClient 

from lavis.datasets.datasets.base_dataset import BaseDataset
import re

def pre_text(text, max_l=None):
    text = re.sub(r"([,.'!?\"()*#:;~])", '', text.lower())
    text = text.replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    text = re.sub(r"\s{2,}", ' ', text)
    text = text.rstrip('\n').strip(' ')

    if max_l:  # truncate
        words = text.split(' ')
        if len(words) > max_l:
            text = ' '.join(words[:max_l])
    return text

def reforge_annotations(old_annotation, is_train=True):
    annotation = []
    for anno in old_annotation:
        # Get video id
        if 'clip_name' in anno: # datasets from clipbert
            video_id = anno.pop('clip_name')
        elif 'video' in anno: # nextqa dataset
            video_id = anno.pop('video')
        else:
            raise NotImplementedError
        # remove ext
        video_id = os.path.basename(video_id)

        # Dealwith multi-caption per video (MSRVTT)
        if type(anno['caption']) != list:
            anno['caption'] = [anno['caption']]
        if is_train:
            captions = anno.pop('caption')
            for caption in captions:
                new_anno = dict(video=video_id, caption=caption)
                new_anno.update(anno)
                annotation.append(new_anno)
        else:
            new_anno = dict(video=video_id, caption=anno['caption'])
            new_anno.update(anno)
            annotation.append(new_anno)
    return annotation


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        visual_key = "image" if "image" in ann else "video"

        return OrderedDict(
            {
                "file": ann[visual_key],
                "caption": ann["caption"],
                visual_key: sample[visual_key],
            }
        )


class RetrievalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "instance_id": ann["instance_id"],
        }


class RetrievalEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                self.text.append(self.text_processor(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __getitem__(self, index):

        image_path = os.path.join(self.vis_root, self.annotation[index]["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {"image": image, "index": index}


class VideoRetrievalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of videos.
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # Forge annotations
        self.annotation = reforge_annotations(self.annotation)
        self.video_ids = set([ann["video"] for ann in self.annotation])

        # Get file client
        if os.path.exists(f'{vis_root}/lock.mdb'):
            self.file_client = FileClient(backend='lmdb', db_path=vis_root)
        else:
            self.file_client = None
            video_dict = {}
            for root, dub_dir, video_files in os.walk(vis_root):
                # print(video_files)
                for video_file in video_files:
                    video_id_ = ".".join(video_file.split(".")[:-1])
                    if video_id_ not in self.video_ids:
                        # print(f"Skip {video_id_}")
                        continue
                    file_path_ = os.path.join(root, video_file)
                    video_dict[video_id_] = file_path_
            self.video_dict = video_dict

        # Generate image ids
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["video"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):
        ann = self.annotation[index]
        if self.file_client is not None:
            vid_obj = io.BytesIO(self.file_client.get(ann["video"]))
        else:
            vid_obj = io.BytesIO(open(self.video_dict[ann["video"]], 'rb').read())

        video = self.vis_processor(vid_obj)
        caption = self.text_processor(ann["caption"])

        return {
            "video": video,
            "text_input": caption,
            "video_id": self.img_ids[ann["video"]],
        }


class VideoRetrievalEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of videos.
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # Forge annotations
        self.annotation = reforge_annotations(self.annotation, is_train=False)
        self.video_ids = set([ann["video"] for ann in self.annotation])

        # Get file client
        if os.path.exists(f'{vis_root}/lock.mdb'):
            self.file_client = FileClient(backend='lmdb', db_path=vis_root)
        else:
            self.file_client = None
            video_dict = {}
            for root, dub_dir, video_files in os.walk(vis_root):
                # print(video_files)
                for video_file in video_files:
                    video_id_ = ".".join(video_file.split(".")[:-1])
                    if video_id_ not in self.video_ids:
                        continue
                    file_path_ = os.path.join(root, video_file)
                    video_dict[video_id_] = file_path_
            self.video_dict = video_dict            

        self.has_multi_vision_gt = False
        self.build_data()

    def __len__(self):
        return len(self.annotation)
    
    def build_data(self):
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        if self.has_multi_vision_gt:
            self.build_data_multi_img_gt()
        else:
            self.build_data_multi_txt_gt()
        # self.anno_list = [dict(image=e) for e in self.image]        

    def build_data_multi_img_gt(self):
        """each text may have multiple ground_truth image, e.g., ssv2"""
        img_id = 0
        for txt_id, ann in enumerate(self.annotation):
            self.text.append(pre_text(ann["caption"]))
            self.txt2img[txt_id] = []
            _images = ann["video"] \
                if isinstance(ann["image"], list) else [ann["image"], ]
            for i, image in enumerate(_images):
                self.image.append(image)
                self.txt2img[txt_id].append(img_id)
                self.img2txt[img_id] = txt_id
                img_id += 1

    def build_data_multi_txt_gt(self):
        """each image may have multiple ground_truth text， e.g., COCO and Flickr30K"""
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["video"])
            self.img2txt[img_id] = []
            # print(ann,self.raw_annotation,flush=True)
            _captions = ann["caption"] \
                if isinstance(ann["caption"], list) else [ann["caption"], ]
            for i, caption in enumerate(_captions):
                self.text.append(pre_text(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                
    def __getitem__(self, index):
        ann = self.annotation[index]

        if self.file_client is not None:
            vid_obj = io.BytesIO(self.file_client.get(ann["video"]))
        else:
            vid_obj = io.BytesIO(open(self.video_dict[ann["video"]], 'rb').read())

        video = self.vis_processor(vid_obj)
        return {"video": video, "index": index}
