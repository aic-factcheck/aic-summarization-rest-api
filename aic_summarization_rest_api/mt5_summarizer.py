from transformers import AutoModelForSeq2SeqLM
from transformers import MT5TokenizerFast
from typing import List, Set, Dict, Tuple, Optional

from .abstract_summarizer import AbstractSummarizer

import logging
logger = logging.getLogger(__name__)


class MT5Summarizer(AbstractSummarizer):
    def __init__(self, cfg: Dict, device: str, lang="cs"):
        assert lang is not None, 'Error: Language symbol is None, but it is expected!'

        self.cfg = cfg
        self.device = device
        model_name = cfg["model_name"]

        self.encoder_max_length = cfg["encoder_max_length"]

        logger.info(f"loading tokenizer for: {model_name}")
        self.tokenizer = MT5TokenizerFast.from_pretrained(model_name)
        self.lang_token = { l: f"<extra_id_{i}>"  for l,i in zip(['cs', 'en', 'de', 'es', 'fr', 'ru', 'tu', 'zh'],range(0,8))}.get(lang, '<extra_id_8>')
        self.lang_code = self.tokenizer(self.lang_token)["input_ids"][0]
        logging.info(f"MT5Tokenizer initialized with {self.lang_token } token and {self.lang_code} code")    

        logger.info(f"loading model: {model_name}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name).to(device)

    def prepare_input_batch(self, batch: List[str]):
        # putting lang code before each document in the batch as a prefix - 
        # to ensure model recognizes appropriate language - do not change order of special ids for individual langs
        batch = [self.lang_token + txt for txt in batch]
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
        Y = Y[:, 2:] # remove <extra_id> tokens (language), this also removes start sequence token, which should not matter
        summaries = self.tokenizer.batch_decode(Y, skip_special_tokens=True)
        return summaries
