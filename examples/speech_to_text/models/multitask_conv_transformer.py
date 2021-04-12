from torch import nn
import torch

from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerModel, base_architecture, \
    s2t_transformer_m, s2t_transformer_s
from examples.speech_to_text.models.multi_task import MultiTaskModel, ClassifierDecoder
from fairseq.models import register_model, register_model_architecture


@register_model('multitask_s2t_transformer')
class MultitaskConvolutionalTransformer(MultiTaskModel):
    @staticmethod
    def add_args(parser):
        S2TTransformerModel.add_args(parser)
        parser.add_argument("--additional-output-size", type=int, default=1,
                            help="number of outputs for the additional decoder")

    @classmethod
    def build_model(cls, args, task):
        base_model = S2TTransformerModel.build_model(args, task)
        auxiliary_decoder = FFNDecoderClassifier(args)
        return MultitaskConvolutionalTransformer(
            base_model.encoder, base_model.decoder, auxiliary_decoder)


class FFNDecoderClassifier(ClassifierDecoder):
    def __init__(self, args):
        super().__init__()
        self.output_size = args.additional_output_size
        self.input_size = args.encoder_embed_dim
        self.fc1 = nn.Linear(self.input_size, self.input_size)
        self.fc2 = nn.Linear(self.input_size, self.output_size)

    def forward(self, encoder_out):
        x = encoder_out.encoder_out  # T x B x C
        return torch.mean(self.fc2(nn.functional.relu(self.fc1(x))), dim=0)


@register_model_architecture('multitask_s2t_transformer', 'multitask_s2t_transformer')
def base_multilingual_architecture(args):
    base_architecture(args)


@register_model_architecture('multitask_s2t_transformer', 'multitask_s2t_transformer_s')
def speechtransformer_multilingual_big(args):
    s2t_transformer_s(args)


@register_model_architecture('multitask_s2t_transformer', 'multitask_s2t_transformer_m')
def speechtransformer_multilingual_big2(args):
    s2t_transformer_m(args)