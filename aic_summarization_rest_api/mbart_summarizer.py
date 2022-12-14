from sentence_splitter import SentenceSplitter
from transformers import MBartForConditionalGeneration
from transformers import MBartTokenizerFast
from typing import List, Set, Dict, Tuple, Optional

from .abstract_summarizer import AbstractSummarizer

import logging
logger = logging.getLogger(__name__)


class MBartSummarizer(AbstractSummarizer):
    def __init__(self, cfg: Dict, device: str, lang="cs"):
        assert lang is not None, 'Error: Language symbol is None, but it is expected!'
        self.lang_token = {'en': 'en_XX','de': 'de_DE', 'es': 'es_XX', 'fr': 'fr_XX', 'ru': 'ru_RU', 'tr': 'tr_TR', 'tu': 'tr_TR', 'cs': 'cs_CZ','zh': 'zh_CN', }.get(lang, lang)

        self.cfg = cfg
        self.device = device

        model_name = cfg["model_name"]
        logger.info(f"loading tokenizer for: {model_name}")
        self.tokenizer = MBartTokenizerFast.from_pretrained(model_name)
        self.encoder_max_length = cfg["encoder_max_length"]
        self.sentence_splitter = SentenceSplitter(language=lang)

        logger.info(f"loading model: {model_name}")
        self.model = MBartForConditionalGeneration.from_pretrained(
            model_name).to(device)

    def prepare_input_batch(self, batch: List[str]):
        # MBart needs to split sentences using </s> token
        batch = [self.tokenizer.eos_token.join(
            self.sentence_splitter.split(t)) for t in batch]
        inputs = self.tokenizer(batch, padding="max_length",
                                truncation=True, max_length=self.cfg["encoder_max_length"], return_tensors="pt")
        return inputs

    def summarize_batch(self, batch: List[str]):
        inputs = self.prepare_input_batch(batch)
        Y = self.model.generate(inputs["input_ids"].to(self.device),
                                attention_mask=inputs["attention_mask"].to(
                                    self.device),
                                decoder_start_token_id=self.tokenizer.convert_tokens_to_ids(
                                    self.lang_token),
                                **self.cfg["inference"]
                                )
        summaries = self.tokenizer.batch_decode(Y, skip_special_tokens=True)
        return summaries
