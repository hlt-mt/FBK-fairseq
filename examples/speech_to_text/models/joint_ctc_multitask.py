# Copyright 2023 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import torch
from torch import nn
import torch.nn.functional as F


from examples.speech_to_text.models.multi_task import MultiTaskModel
from fairseq.models.speech_to_text import FairseqEncoderDecoderModel


class JointCtcMultiTaskModel(MultiTaskModel):
    """
    This model is an implementation of a multi-task model that predicts the output both
    with a CTC on the encoder output and with an autoregressive decoder.
    """
    encoder_parent_model: FairseqEncoderDecoderModel

    @classmethod
    def build_model(cls, args, task):
        task.target_dictionary.add_symbol("<ctc_blank>")
        base_model = cls.encoder_parent_model.build_model(args, task)
        auxiliary_decoder = CtcDecoder(args, task.target_dictionary)
        model = cls(base_model.encoder, base_model.decoder, auxiliary_decoder, args)
        model.ignore_prefix_size = getattr(args, 'ignore_prefix_size', 0)
        return model

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        lang_embeds = None
        if self.ignore_prefix_size > 0:
            assert self.ignore_prefix_size == 1, "ignore_prefix_size > 1 is not supported"
            # the prev_output_tokens contain <bos> <lang> and then the sentence
            # so the 2nd column corresponds to the language embeddings
            lang_embeds = self.decoder.embed_tokens(prev_output_tokens[:, 1:2])
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out)
        auxiliary_out = self.auxiliary_decoder(encoder_out, lang_embeds=lang_embeds)
        if self.encoder.ctc_flag:
            return (decoder_out, auxiliary_out), {
                "ctc_out": encoder_out["ctc_out"], "ctc_lengths": encoder_out["ctc_lengths"]}
        else:
            return decoder_out, auxiliary_out

    # At the moment, we support only target. If needed in the future, we can control the target
    # with a parameter
    def get_auxiliary_target(self, sample, auxiliary_output):
        return sample["target"]

    def get_auxiliary_token_lens(self, sample):
        return sample["target_lengths"] - 1

    def get_auxiliary_input_lens(self, sample, net_output):
        return (~net_output[1]["padding_mask"]).long().sum(-1)

    @property
    def auxiliary_dict(self):
        return self.auxiliary_decoder.dictionary


class CtcDecoder(nn.Module):
    def __init__(self, args, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.fc = nn.Linear(args.encoder_embed_dim, len(self.dictionary))
        if getattr(args, 'ignore_prefix_size', 0) > 0:
            self.lang_embed_layernorm = nn.LayerNorm(args.encoder_embed_dim)

    def get_normalized_probs(self, net_output, log_probs=False):
        if log_probs:
            probs = F.log_softmax(net_output[0], dim=-1)
        else:
            probs = F.softmax(net_output[0], dim=-1)
        probs.batch_first = True
        return probs

    def forward(self, encoder_out, lang_embeds=None):
        x = encoder_out["encoder_out"][0]  # T x B x C
        if lang_embeds is not None:
            x = x + lang_embeds.transpose(0, 1)  # this must not be an inplace operation (e.g. +=)
            x = self.lang_embed_layernorm(x)
        if len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]
        else:
            padding_mask = torch.zeros((x.shape[1], x.shape[0]), dtype=torch.bool).to(x.device)
        x = self.fc(x)
        x.masked_fill_(padding_mask.transpose(0, 1).unsqueeze(-1), 0.)
        return x, {"padding_mask": padding_mask}


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
