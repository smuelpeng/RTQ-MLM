import os

import torch
from lavis.common.registry import registry
from lavis.models.med import XBertEncoder
from lavis.models.blip_video_models.blip_video_retrieval import BlipVideoRetrieval

from .vit_post import VisionTransformerEncoderWithPostProcess
import os
from copy import deepcopy

import logging
import torch.nn.functional as F

from lavis.models.backbones.vit import interpolate_pos_embed
from lavis.common.registry import registry
from lavis.models.blip_video_models import compute_sim_matrix
from lavis.models.base_model import (
    MomentumDistilationMixin,
    SharedQueueMixin,
    all_gather_with_grad,
    concat_all_gather,
)
from lavis.models.blip_models.blip_retrieval import BlipRetrieval
from lavis.models.blip_models.blip_outputs import (
    BlipOutput,
    BlipSimilarity,
)
from lavis.models.blip_video_models.blip_video_outputs import BlipVideoIntermediateOutput
from lavis.models.backbones.co_attention_transformer_module import Co_attention_block
from einops import rearrange, reduce

from torch import nn


@registry.register_model("pac_blip_video_post_retrieval")
class PACBlipVideoPostRetrieval(BlipVideoRetrieval):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/blip_post/blip_base_retrieval.yaml",
    }

    def __init__(
        self,
        image_encoder,
        text_encoder,
        queue_size,
        alpha=0.4,
        embed_dim=256,
        momentum=0.995,
        negative_all_rank=False,
        max_txt_len=35,
        cfg=None,
    ):
        super().__init__(image_encoder=image_encoder,
                         text_encoder=text_encoder,
                         queue_size=queue_size,
                         alpha=alpha,
                         embed_dim=embed_dim,
                         momentum=momentum,
                         negative_all_rank=negative_all_rank,
                         max_txt_len=max_txt_len,
                         )
        self.pos_mix_type = cfg.get("pos_mix_type", "text_concat")

        # segment-aware text interaction
        self.segment_num = cfg.get("segment_num", 3)

        # co attention
        co_attention_cfg = cfg.get("co_attention")
        self.co_attention = Co_attention_block(co_attention_cfg.num_attention_heads,
                                               co_attention_cfghidden_size,                                               co_attention_cfg.dropout_rate)

    @ classmethod
    def from_config(cls, cfg=None):
        # set from_pretrained=True to load weights for 'bert-base-uncased'
        backbone_arch = cfg.get('backbone_arch', 'vit')
        if backbone_arch == 'vit':
            image_encoder = VisionTransformerEncoderWithPostProcess.from_config(
                cfg)
        else:
            image_encoder = registry.get_model_class(
                backbone_arch).from_config(cfg)

        text_encoder = XBertEncoder.from_config(cfg)

        if cfg.get("freeze_video", False):
            for param in image_encoder.parameters():
                param.requires_grad = False
        if cfg.get("freeze_text", False):
            for param in text_encoder.parameters():
                param.requires_grad = False

        embed_dim = cfg.get("embed_dim", 256)
        momentum = cfg.get("momentum", 0.995)
        alpha = cfg.get("alpha", 0.4)
        negative_all_rank = cfg.get("negative_all_rank", False)

        queue_size = cfg.get("queue_size", 0)
        max_txt_len = cfg.get("max_txt_len", 35)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            queue_size=queue_size,
            alpha=alpha,
            embed_dim=embed_dim,
            momentum=momentum,
            negative_all_rank=negative_all_rank,
            max_txt_len=max_txt_len,
            cfg=cfg,
        )

        model.load_checkpoint_from_config(cfg)
        model.reset_queue_ptr()

        return model

    def get_text_feat(self, caption, device):
        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        text_output = self.text_encoder.forward_text(text)
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        return text_feat

    def get_pac_mask(self, feat_t, feat_v, feat_p):
        import numpy as np
        sim_tv = torch.cosine_similarity(feat_t, feat_v, dim=-1)
        sim_pv = torch.cosine_similarity(feat_p, feat_v, dim=-1)
        # mask = np.zeros_like(v1, dtype=int)
        mask = torch.zeros_like(feat_t, dtype=torch.long)
        for i in range(len(sim_tv)):
            if sim_tv[i] > sim_pv[i]:
                mask[i] = 0
            else:
                prob = (sim_pv[i] - sim_tv[i]) / sim_pv[i]
                mask[i] = 1 if np.random.rand() < prob else 0

        return mask

    def get_pac_text(self, text, overall_caption, mask,
                     text_feat=None, overall_feat=None,
                     device="cpu"):
        if text_feat is None:
            text_feat = self.get_text_feat(text, device)
        if overall_feat is None:
            overall_feat = self.get_text_feat(overall_caption, device)

        if self.pos_mix_type == "text_concat":
            pac_text = f'{text} {overall_caption}'
            pac_feat = self.get_text_feat(pac_text, device)

        elif self.pos_mix_type == "embed_mix":
            pac_text = text
            pac_feat = text_feat + overall_feat / 2

        pac_feat = pac_feat * mask + text_feat * (1 - mask)
        return pac_text, pac_feat

    def segment_aware_text_interaction(self, video_outputs, video_atts, seg_caps):
        # get segment level text features
        seg_text = self.tokenizer(
            seg_caps,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(video_outputs.device)
        seg_text_output = self.text_encoder.forward_text(seg_text)
        seg_text_embeds = seg_text_output.last_hidden_state
        seg_text_feat = F.normalize(
            self.text_proj(seg_text_embeds[:, 0, :]), dim=-1)
        seg_text_feat = rearrange(
            seg_text_feat, '(b s) d -> b s d', s=self.segment_num)
        co_vision_layer_outputs = []
        for seg_i in range(self.segment_num):
            num_per_seg = video_outputs.size(1) // self.segment_num
            vision_layer_output_i, text_layer_output, co_attention_probs = self.co_attention(
                video_outputs[:, seg_i *
                              num_per_seg:(seg_i+1)*num_per_seg, :], video_atts,
                seg_text_embeds[:, seg_i, :],
                seg_text.attention_mask)
            co_vision_layer_outputs.append(vision_layer_output_i)
        video_outputs = torch.cat(co_vision_layer_outputs, dim=1)
        return video_outputs

    def forward(self, samples):
        video = samples["video"]
        caption = samples["text_input"]
        idx = samples["video_id"]

        alpha = self.alpha * self._rampup_factor(
            epoch=samples["epoch"],
            iters=samples["iters"],
            num_iters_per_epoch=samples["num_iters_per_epoch"],
        )

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        video_feat, video_outputs = self.visual_encoder.forward_features(video)
        video_atts = torch.ones(video_outputs.size()[:-1], dtype=torch.long).to(
            video.device
        )
        video_feat = F.normalize(self.vision_proj(video_feat), dim=-1)

        # get text features
        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(video.device)

        text_output = self.text_encoder.forward_text(text)
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # get pac-text features
        # pac_feats = self.get_text_feat(overall_caption)
        if "overall_caption" in samples:
            overall_caption = samples["overall_caption"]
            segment_captions = samples["segment_captions"]

            overall_feat = self.get_text_feat(overall_caption, video.device)
            pac_mask = self.get_pac_mask(text_feat, video_feat, overall_feat)
            pac_text, pac_feat = self.get_pac_text(
                caption, overall_caption, pac_mask, text_feat, overall_feat, video.device)

            text_feat = pac_feat

            video_outputs = self.segment_aware_text_interaction(
                video_outputs, video_atts, segment_captions)

            video_feat  = reduce(video_outputs[:,0], '(b to) h -> b h', 'mean', b=B)

        # video-text Contrastive Learning
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            video_feat_m = self.visual_encoder_m(video)
            video_feat_m = F.normalize(
                self.vision_proj_m(video_feat_m), dim=-1
            )
            video_feat_m_all = torch.cat(
                [video_feat_m.t(), self.image_queue.clone().detach()], dim=1
            )

            text_output_m = self.text_encoder_m.forward_text(text)
            text_embeds_m = text_output_m.last_hidden_state
            text_feat_m = F.normalize(
                self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
            text_feat_m_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )

            sim_i2t_m = video_feat_m @ text_feat_m_all / self.temp
            sim_t2i_m = text_feat_m @ video_feat_m_all / self.temp

            sim_i2t_targets = (
                alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            )
            sim_t2i_targets = (
                alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            )

        sim_i2t = video_feat @ text_feat_m_all / self.temp
        sim_t2i = text_feat @ video_feat_m_all / self.temp

        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1
        ).mean()

        loss_itc = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(video_feat_m, text_feat_m, idx)

        # video-text Matching
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        # forward the positve video-text pair
        bs = video.size(0)
        output_pos = self.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=video_outputs,
            encoder_attention_mask=video_atts,
            return_dict=True,
        )

        idxs = concat_all_gather(idx)
        if self.negative_all_rank:
            # compute sample similarity
            with torch.no_grad():
                mask = torch.eq(idx, idxs.t())

                video_feat_world = concat_all_gather(video_feat)
                text_feat_world = concat_all_gather(text_feat)

                sim_i2t = video_feat @ text_feat_world.t() / self.temp
                sim_t2i = text_feat @ video_feat_world.t() / self.temp

                weights_i2t = F.softmax(sim_i2t, dim=1)
                weights_i2t.masked_fill_(mask, 0)

                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_t2i.masked_fill_(mask, 0)

            video_outputs_world = all_gather_with_grad(video_outputs)

            # select a negative video (from all ranks) for each text
            video_outputs_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                video_outputs_neg.append(video_outputs_world[neg_idx])
            video_outputs_neg = torch.stack(video_outputs_neg, dim=0)

            # select a negative text (from all ranks) for each video
            input_ids_world = concat_all_gather(encoder_input_ids)
            att_mask_world = concat_all_gather(text.attention_mask)

            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(input_ids_world[neg_idx])
                text_atts_neg.append(att_mask_world[neg_idx])

        else:
            with torch.no_grad():
                mask = torch.eq(idx, idx.t())

                sim_i2t = video_feat @ text_feat.t() / self.temp
                sim_t2i = text_feat @ video_feat.t() / self.temp

                weights_i2t = F.softmax(sim_i2t, dim=1)
                weights_i2t.masked_fill_(mask, 0)

                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_t2i.masked_fill_(mask, 0)

            # select a negative video (from same rank) for each text
            video_outputs_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                video_outputs_neg.append(video_outputs[neg_idx])
            video_outputs_neg = torch.stack(video_outputs_neg, dim=0)

            # select a negative text (from same rank) for each video
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(encoder_input_ids[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        video_outputs_all = torch.cat(
            [video_outputs_neg, video_outputs], dim=0)
        video_atts_all = torch.cat([video_atts, video_atts], dim=0)

        output_neg = self.text_encoder(
            text_ids_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=video_outputs_all,
            encoder_attention_mask=video_atts_all,
            return_dict=True,
        )

        vl_embeddings = torch.cat(
            [
                output_pos.last_hidden_state[:, 0, :],
                output_neg.last_hidden_state[:, 0, :],
            ],
            dim=0,
        )
        itm_logits = self.itm_head(vl_embeddings)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(
                2 * bs, dtype=torch.long)],
            dim=0,
        ).to(self.device)
        loss_itm = F.cross_entropy(itm_logits, itm_labels)

        return BlipOutput(
            loss=loss_itc + loss_itm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            sims=BlipSimilarity(
                sim_i2t=sim_i2t,
                sim_t2i=sim_t2i,
                sim_i2t_m=sim_i2t_m,
                sim_t2i_m=sim_t2i_m,
                sim_i2t_targets=sim_i2t_targets,
                sim_t2i_targets=sim_t2i_targets,
            ),
            intermediate_output=BlipVideoIntermediateOutput(
                video_embeds=video_outputs,
                video_embeds_m=video_feat_m,
                text_embeds=text_embeds,
                text_embeds_m=text_embeds_m,
                encoder_output=output_pos,
                encoder_output_neg=output_neg,
                itm_logits=itm_logits,
                itm_labels=itm_labels,
            ),
        )
