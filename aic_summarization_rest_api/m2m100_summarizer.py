from transformers import AutoModelForSeq2SeqLM
from transformers import M2M100Tokenizer
from typing import List, Set, Dict, Tuple, Optional

from .abstract_summarizer import AbstractSummarizer

import logging
logger = logging.getLogger(__name__)


class M2M100Summarizer(AbstractSummarizer):
    def __init__(self, cfg: Dict, device: str, lang="cs"):
        assert lang is not None, 'Error: Language symbol is None, but it is expected!'

        self.cfg = cfg
        self.device = device
        model_name = cfg["model_name"]

        logger.info(f"loading tokenizer for: {model_name}")
        self.tokenizer = M2M100Tokenizer.from_pretrained(
            model_name, src_lang=lang, tgt_lang=lang)
        self.encoder_max_length = cfg["encoder_max_length"]
        self.lang_code = self.tokenizer.get_lang_id(lang)
        self.lang_token = self.lang_code
        logging.info(
            f"M2M100Tokenizer initialized with {lang} id and {self.lang_code} code")

        logger.info(f"loading model: {model_name}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name).to(device)

    def prepare_input_batch(self, batch: List[str]):
        inputs = self.tokenizer(batch, truncation=True, padding="max_length",
                                max_length=self.encoder_max_length, return_tensors="pt", add_special_tokens=True)
        return inputs

    def summarize_batch(self, batch: List[str]):
        inputs = self.prepare_input_batch(batch)
        Y = self.model.generate(inputs["input_ids"].to(self.device),
                                attention_mask=inputs["attention_mask"].to(
                                    self.device),
                                forced_bos_token_id=self.lang_code,
                                # decoder_start_token_id=self.tokenizer.convert_tokens_to_ids(
                                    # self.lang_token),
                                **self.cfg["inference"]
                                )
        summaries = self.tokenizer.batch_decode(Y, skip_special_tokens=True)
        return summaries
