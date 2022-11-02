#!/usr/bin/env python3 -u
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
"""
Simple server waiting on stdin for ST data for which a double output is produced.
"""
import json
import logging
import sys
import uuid
from argparse import Namespace

from omegaconf import DictConfig

from fairseq import options
from fairseq.dataclass.utils import convert_namespace_to_omegaconf


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"

    start_server(cfg)


def start_server(cfg: DictConfig):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )
    logger = logging.getLogger('fbk_fairseq.simple_io_server')
    logger.info("Starting server...")

    # Initialize generator
    if cfg.task.server_processor == "st_triangle":
        from api.st_triangle_processor import STTriangleProcessor
        processor = STTriangleProcessor(cfg)
    elif cfg.task.server_processor == "st":
        from api.st_processor import STProcessor
        processor = STProcessor(cfg)
    else:
        logger.exception(f"Invalid --server-processor: {cfg.server_processor}.")
        sys.exit(-1)

    while True:
        logger.info("Waiting for input...")
        input_json = sys.stdin.readline()
        request_id = uuid.uuid4()
        logger.info(f"Received Request ID[{request_id}]")
        try:
            input_request = json.loads(input_json)
            if "command" in input_request:
                if input_request["command"] == "shutdown":
                    logger.info("Shutting down...")
                    break
                else:
                    raise Exception(f"Unrecognized command {input_request['command']}")

            output = processor.process(request_id, input_request)
            output["status"] = "ok"
            output_json = json.dumps(output)
        except BaseException as e:
            logger.exception(f"Issue while processing ID[{request_id}]")
            output_json = json.dumps({"status": "error", "message": str(e)})
        logger.info(f"Answering Request ID[{request_id}]")
        sys.stdout.write(output_json + "\n")


def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument("--server-processor", type=str, choices=["st", "st_triangle"])
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
