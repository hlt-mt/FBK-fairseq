import logging
import os

import torch
import numpy as np
import sys

from fairseq import utils, options, tasks, progress_bar, checkpoint_utils
from examples.speech_to_text.data.knowledge_distillation import TeacherOutputDataset


logger = logging.getLogger("fairseq_cli.generate")



def gen_outputs(args):
    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    dataset = task.dataset(args.gen_subset)
    logger.info('{} {} {} examples'.format(args.data, args.gen_subset, len(dataset)))

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _ = checkpoint_utils.load_model_ensemble(
        args.path.split(':'), task=task, arg_overrides=eval(args.model_overrides))
    assert len(models) == 1
    model = models[0]
    # Optimize ensemble for generation
    model.make_generation_fast_(
        beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
        need_attn=args.print_alignment,
    )
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions()
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    ).next_epoch_itr(shuffle=False)

    outputs = [None for _ in range(len(dataset))]
    with progress_bar.build_progress_bar(args, itr) as t:
        for sample in t:
            s = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in s:
                continue
            # We assume the target is already present and known
            assert s['target'] is not None
            targets = s['target']
            with torch.no_grad():
                net_output = model(**s['net_input'])
                topk_outs, topk_idx = torch.topk(net_output[0], args.distill_topk, dim=-1)  # B, T, k
                non_padding_mask = targets.ne(task.target_dictionary.pad()).cpu().numpy().astype(bool)
            topk_idx = topk_idx.cpu().numpy()
            topk_outs = topk_outs.cpu().numpy()
            for i, id_s in enumerate(s['id'].data):
                outputs[id_s] = [
                    topk_idx[i, non_padding_mask[i]].tolist(),
                    topk_outs[i, non_padding_mask[i]].tolist()]
    return outputs


def save_expert_outputs(args, expert_outputs):
    logger.info("Start saving expert outputs..")
    src_lang = args.source_lang
    tgt_lang = args.target_lang
    file_prefix = '{}.{}-{}.{}'.format(args.gen_subset, src_lang, tgt_lang, tgt_lang)
    path = os.path.join(args.data, file_prefix + '.top{}_idx'.format(args.distill_topk))
    TeacherOutputDataset.save_bin(path, [o[0] for o in expert_outputs], np.int32)
    logger.info("Written {}".format(path))

    path = os.path.join(args.data, file_prefix + '.top{}_out'.format(args.distill_topk))
    TeacherOutputDataset.save_bin(path, [o[1] for o in expert_outputs], float)
    logger.info("Written {}".format(path))


if __name__ == '__main__':
    parser = options.get_generation_parser()
    parser.add_argument('--distill-topk', default=8, type=int)
    args = options.parse_args_and_arch(parser)
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 12000
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level="INFO",
        stream=sys.stdout,
    )
    logger.info(args)
    expert_outputs = gen_outputs(args)
    save_expert_outputs(args, expert_outputs)