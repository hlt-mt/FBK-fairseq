from torch import nn
import torch

from fairseq.models import FairseqEncoderDecoderModel


class MultiTaskModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, auxiliary_decoder):
        super().__init__(encoder, decoder)
        self.auxiliary_decoder = auxiliary_decoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        auxiliary_out = self.auxiliary_decoder(encoder_out)
        return decoder_out, auxiliary_out

    def get_auxiliary_target(self, sample, auxiliary_output):
        return sample["auxiliary_target"]


class ClassifierDecoder(nn.Module):
    def get_normalized_probs(self, net_output, log_probs=False):
        if self.output_size == 1:
            if log_probs:
                return nn.functional.logsigmoid(net_output)
            else:
                return torch.sigmoid(net_output)
        else:
            if log_probs:
                return nn.functional.log_softmax(net_output, dim=-1)
            else:
                return nn.functional.softmax(net_output, dim=-1)

