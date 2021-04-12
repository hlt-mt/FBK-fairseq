from torch import nn
import torch

from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerModel, base_architecture, \
    s2t_transformer_m, s2t_transformer_s, S2TTransformerEncoder
from examples.speech_to_text.models.multi_task import MultiTaskModel
from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.speech_to_text import TransformerDecoderScriptable


@register_model('s2t_transformer_2stage')
class S2TTransformer2Stage(MultiTaskModel):
    """
    This model is an implementation of a multi-task model that predicts both transcripts
    and translations, with the translation being generated from the output representation
    of the transcript decoder. It represents the 2Stage model of (Sperber et al. 2020).
    """
    # TODO: do we need different settings/configs for the two decoders?
    # For now, we assume NO.
    @staticmethod
    def add_args(parser):
        S2TTransformerModel.add_args(parser)
        parser.add_argument('--auxiliary-decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 100000
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 100000

        # This model requires a task that provides source dictionary and transcripts
        assert task.source_dictionary is not None and task.target_dictionary is not None

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        target_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)
        src_embed_tokens = build_embedding(src_dict, args.decoder_embed_dim)
        encoder = S2TTransformerEncoder(args, tgt_dict)
        decoder = TransformerDecoderScriptable(args, tgt_dict, target_embed_tokens)
        auxiliary_decoder = TransformerDecoderScriptable(args, src_dict, src_embed_tokens)
        return S2TTransformer2Stage(encoder, decoder, auxiliary_decoder)

    # In "speech_translation_with_transcription" the transcripts are read into
    # "transcript_target". Not the most elegant solution, but it allows
    # compatibility with existing code.
    def get_auxiliary_target(self, sample, auxiliary_output):
        return sample["transcript"]

    def get_auxiliary_token_lens(self, sample):
        return sample["transcript_lengths"]

    def forward(self, src_tokens, src_lengths, prev_output_tokens, prev_transcript_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        auxiliary_out = self.auxiliary_decoder(
            prev_transcript_tokens, encoder_out=encoder_out, features_only=True)
        auxiliary_padding_mask = prev_transcript_tokens.eq(
            self.auxiliary_decoder.padding_idx)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=EncoderOut(
                encoder_out=auxiliary_out[0].transpose(0, 1),  # T x B x C
                encoder_padding_mask=auxiliary_padding_mask,
                encoder_embedding=None,
                encoder_states=[],
                src_tokens=None,
                src_lengths=None),
            **kwargs
        )
        return decoder_out, (self.auxiliary_decoder.output_layer(auxiliary_out[0]), auxiliary_out[1])

    def forward_decoder(self, prev_output_tokens, auxiliary_out, auxiliary_tokens, **kwargs):
        if "encoder_out" in kwargs:
            del kwargs["encoder_out"]
        auxiliary_padding_mask = auxiliary_tokens.eq(
            self.auxiliary_decoder.padding_idx)
        return self.decoder(
            prev_output_tokens,
            encoder_out=EncoderOut(
                encoder_out=auxiliary_out.transpose(0, 1),
                encoder_padding_mask=auxiliary_padding_mask,
                encoder_embedding=None,
                encoder_states=[],
                src_tokens=None,
                src_lengths=None),
            **kwargs
        )


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


@register_model_architecture('s2t_transformer_2stage', 's2t_transformer_2stage')
def base_multilingual_architecture(args):
    base_architecture(args)


@register_model_architecture('s2t_transformer_2stage', 's2t_transformer_2stage_s')
def s2t_transformer_2stage_m(args):
    s2t_transformer_s(args)


@register_model_architecture('s2t_transformer_2stage', 's2t_transformer_2stage_m')
def s2t_transformer_2stage_m(args):
    s2t_transformer_m(args)


