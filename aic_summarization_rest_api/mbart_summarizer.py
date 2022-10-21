from typing import List, Set, Dict, Tuple, Optional
from transformers import MBartTokenizerFast, MBartForConditionalGeneration

import logging
logger = logging.getLogger(__name__)

from aic_summarization_rest_api.tokenization import MorphoDiTaTokenizer

class MBartSummarizer():
    def __init__(self, model_name: str, device: str, lang="cs", encoder_max_length=512):
        if lang == "cs":
            self.lang= lang
            self.lang_token = "cs_CZ"
        else:
            assert f"unsupported language: {lang}"

        self.model_name = model_name
        self.device = device
        logger.info(f"loading tokenizer for: {model_name}")
        self.tokenizer = MBartTokenizerFast.from_pretrained(model_name)
        logger.info(f"loading model: {model_name}")
        self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
        self.encoder_max_length = encoder_max_length
        self.sentence_tokenizer = MorphoDiTaTokenizer(lang=lang)
    
    def preprocess_batch(self, batch: List[str]):
        # MBart needs to split sentences using </s> token
        batch = [self.tokenizer.eos_token.join(
            self.sentence_tokenizer.tokenizeSentences(t)) for t in batch]
        inputs = self.tokenizer(batch, padding="max_length",
                        truncation=True, max_length=self.encoder_max_length, return_tensors="pt")
        return inputs

    def summarize_batch(self, batch: List[str]):
        inputs = self.preprocess_batch(batch)
        Y = self.model.generate(inputs["input_ids"].to(self.device),
                        attention_mask=inputs["attention_mask"].to(self.device),
                        num_beams=4,
                        decoder_start_token_id=self.tokenizer.convert_tokens_to_ids(self.lang_token),
                        )
        summaries = self.tokenizer.batch_decode(Y, skip_special_tokens=True)
        return summaries