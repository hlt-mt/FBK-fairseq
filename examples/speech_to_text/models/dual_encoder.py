# Copyright 2022 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from fairseq.models import FairseqEncoderDecoderModel, FairseqEncoder


class FairseqDualEncoderModel(FairseqEncoderDecoderModel):
    """
    Base class for encoder-decoder models accepting a text
    which is encoded with a proper encoder.
    Args:
        encoder (FairseqEncoder): speech encoder
        decoder (FairseqDecoder): decoder
        context_encoder (FairseqEncoder): context encoder
    """
    def __init__(self, encoder, decoder, context_encoder):
        super().__init__(encoder, decoder)
        assert isinstance(context_encoder, FairseqEncoder)
        self.context_encoder = context_encoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens, context_tokens, context_lengths, **kwargs):
        """
        Run the forward pass for a speech encoder-text decoder model with
        an additional context encoder.
        First, encode the context. Then, feed a batch of source speech tokens and the
        encoded context through the encoder. Lastly, feed the encoded speech, the
        encoded context and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::
            context_enc = self.context_encoder(context_tokens, context_lengths)
            encoder_out = self.encoder(src_tokens, src_lengths, context_enc)
            return self.decoder(prev_output_tokens, encoder_out, context_enc)
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            context_tokens (LongTensor): tokens representing the context of shape
                `(batch, context_len)`
            context_lengths (LongTensor): context lengths of shape `(batch)`
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        context_out = self.context_encoder(context_tokens, src_lengths=context_lengths)
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, context_encoder_out=context_out, **kwargs)
        if self.encoder.ctc_flag:
            return decoder_out, {"ctc_out": encoder_out["ctc_out"], "ctc_lengths": encoder_out["ctc_lengths"]}
        else:
            return decoder_out
