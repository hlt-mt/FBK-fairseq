from torch import nn
import torch

from examples.speech_recognition.models.conv_transformer import ConvolutionalTransformerModel, base_architecture, \
    speechtransformer_big, speechtransformer_big2, ConvolutionalTransformerEncoder
from examples.speech_recognition.models.multi_task import MultiTaskModel, ClassifierDecoder
from examples.speech_recognition.tasks.speech_translation_ctc import SpeechTranslationCTCTask
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerDecoder


@register_model('conv_transformer_dualdecoder')
class ConvolutionalTransformerDualDecoder(MultiTaskModel):
    """
    This model is an implementation of a multi-task model that predicts both transcripts
    and translations, like in (Weiss et al. 2017). It represents the DirMul model of
    (Sperber et al. 2020).
    """
    # TODO: do we need different settings/configs for the two decoders?
    # For now, we assume NO.
    @staticmethod
    def add_args(parser):
        ConvolutionalTransformerModel.add_args(parser)
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
        assert isinstance(task, SpeechTranslationCTCTask)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        target_embed_tokens = build_embedding(
            tgt_dict, args.decoder_embed_dim, args.decoder_embed_path)
        src_embed_tokens = build_embedding(
            src_dict, args.decoder_embed_dim, args.auxiliary_decoder_embed_path)
        encoder = ConvolutionalTransformerEncoder(
            args, tgt_dict, audio_features=args.input_feat_per_channel)
        decoder = TransformerDecoder(args, tgt_dict, target_embed_tokens)
        auxiliary_decoder = TransformerDecoder(args, src_dict, src_embed_tokens)
        return ConvolutionalTransformerDualDecoder(encoder, decoder, auxiliary_decoder)

    # In "speech_translation_with_transcription" the transcripts are read into
    # "transcript_target". Not the most elegant solution, but it allows
    # compatibility with existing code.
    def get_auxiliary_target(self, sample, auxiliary_output):
        return sample["transcript_target"]

    def forward(self, src_tokens, src_lengths, prev_output_tokens, transcript_prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        auxiliary_out = self.auxiliary_decoder(
            transcript_prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out, auxiliary_out


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


@register_model_architecture('conv_transformer_dualdecoder', 'conv_transformer_dualdecoder')
def base_multilingual_architecture(args):
    base_architecture(args)
    args.auxiliary_decoder_embed_path = getattr(args, "auxiliary_decoder_embed_path", None)


@register_model_architecture('conv_transformer_dualdecoder', 'conv_transformer_dualdecoder_big')
def speechtransformer_multilingual_big(args):
    speechtransformer_big(args)
    args.auxiliary_decoder_embed_path = getattr(args, "auxiliary_decoder_embed_path", None)


@register_model_architecture('conv_transformer_dualdecoder', 'conv_transformer_dualdecoder_big2')
def speechtransformer_multilingual_big2(args):
    speechtransformer_big2(args)
    args.auxiliary_decoder_embed_path = getattr(args, "auxiliary_decoder_embed_path", None)
