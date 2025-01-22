import inspect
import logging
import re
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from typing import TypedDict, Unpack, cast
from unittest.mock import patch

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM, eval_logger
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, PreTrainedModel,
                          PreTrainedTokenizer)

logger = logging.getLogger("lm-eval")


class RegexFilter(logging.Filter):
    def __init__(self, regex):
        super().__init__()
        self.regex = re.compile(regex)

    def filter(self, record):
        return not self.regex.search(record.getMessage())


@contextmanager
def remove_logging_by_regex(
    logger: logging.Logger,
    regex: str,
):
    handlers = logger.handlers or logging.getLogger().handlers
    filter = RegexFilter(regex)
    for handler in handlers:
        handler.addFilter(filter)
    yield
    for handler in handlers:
        handler.removeFilter(filter)


@contextmanager
def override_tqdm_init(mininterval: float = 60.0, disable: bool = False):
    class tqdm_quieter(tqdm):
        def __init__(self, *args, **kwargs):
            all_kwargs = (
                inspect.signature(super().__init__).bind(*args, **kwargs).arguments
            )
            all_kwargs["mininterval"] = mininterval
            all_kwargs["maxinterval"] = max(10, mininterval * 2)
            all_kwargs["disable"] = disable
            super().__init__(**all_kwargs)

    with (
        # patch('lm_eval.base.tqdm', tqdm_quieter),
        patch("lm_eval.models.huggingface.tqdm", tqdm_quieter),
        # patch('bigcode_eval.utils.tqdm', tqdm_quieter),
    ):
        yield


@dataclass
class LMEvalLoggingConfig:
    remove_model_sha: bool = True
    tqdm_interval: float = 60.0
    disable_tqdm: bool = False
    remove_samples_in_result: bool = True
    disable_logging: bool = False


class _LMEvalLoggingConfig(TypedDict):
    remove_model_sha: bool
    tqdm_interval: float
    disable_tqdm: bool
    remove_samples_in_result: bool
    disable_logging: bool


@torch.inference_mode()
def run_lm_eval_hf(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | str,
    tasks: list[str],
    num_fewshot: int = None,
    limit: int = None,
    device: str | torch.device = "cuda",
    batch_size: str | int = None,
    # the max length for the inputs. If too long, will split into several forwards
    max_length: int = None,
    # model_dtype: str | torch.dtype = "auto", # unncessary, this is for model creation
    apply_chat_template: bool = False,
    **logging_kwargs: Unpack[_LMEvalLoggingConfig],
):
    device = str(torch.device(device).type)

    logging_config = HfArgumentParser(LMEvalLoggingConfig).parse_dict(
        logging_kwargs, allow_extra_keys=True
    )[0]
    logging_config = cast(LMEvalLoggingConfig, logging_config)
    logger.info("lm_eval patching: %s", logging_config)
    contexts = []
    if (
        logging_config.tqdm_interval is not None and logging_config.tqdm_interval > 0
    ) or logging_config.disable_tqdm:
        contexts.append(
            override_tqdm_init(
                mininterval=logging_config.tqdm_interval,
                disable=logging_config.disable_tqdm,
            )
        )
    if logging_config.remove_model_sha:
        contexts.append(remove_logging_by_regex(eval_logger, "Failed to get model SHA"))
    if logging_config.disable_logging:
        contexts.append(remove_logging_by_regex(eval_logger, ".*"))
    if model.training:
        logger.warning("Model is in training mode, setting to eval mode")
        model = model.eval()

    with ExitStack() as stack:
        for context in contexts:
            stack.enter_context(context)
        lm_obj = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            # device=device, # unnecessary, this is for model creation
            # dtype=model_dtype, # unnecessary, this is for model creation
            max_length=max_length,
        )
        results = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            use_cache=None,  # do not use cached evaluation results
            check_integrity=False,
            batch_size=batch_size,
            device=device,
            apply_chat_template=apply_chat_template,
        )
    if logging_config.remove_samples_in_result:
        results.pop("samples", None)
    return results


if __name__ == "__main__":
    import accelerate.hooks
    from transformers import LlamaForCausalLM

    torch.set_grad_enabled(False)

    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    )  # .eval()

    lengths = []

    class InputShapeHook(accelerate.hooks.ModelHook):
        def pre_forward(self, module, *args, **kwargs):
            lengths.append(args[0].shape[1])
            # print(
            #     "****",
            #     tokenizer.decode(
            #         args[0][0].cpu(), skip_special_tokens=False, clean_up_tokenization_spaces=False
            #     ).replace("\n", "\\n"),
            # )
            if "past_key_values" in kwargs:
                print(kwargs["past_key_values"])
            return super().pre_forward(module, *args, **kwargs)

    accelerate.hooks.add_hook_to_module(model, InputShapeHook())

    tasks = ["wikitext"]
    results = run_lm_eval_hf(
        model=model,
        tokenizer=model_id,
        tasks=tasks,
        num_fewshot=0,
        limit=2,
        max_length=2048,
        batch_size=1,
        device="cuda",
        disable_tqdm=True,
    )
    print(results["results"])
    print(lengths)
