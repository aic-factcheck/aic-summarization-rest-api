from typing import List, Set, Dict, Tuple, Optional
from .m2m100_summarizer import M2M100Summarizer
from .mbart_summarizer import MBartSummarizer
from .mt5_summarizer import MT5Summarizer


def create_summarizer(cfg: Dict, device: str, lang: str = "cs"):
    model_name = cfg["model_name"]
    name = model_name.split('/')[-1]
    if name.startswith('mbart'):
        return MBartSummarizer(cfg, device, lang=lang)
    elif name.startswith('m2m100'):
        return M2M100Summarizer(cfg, device, lang=lang)
    elif name.startswith('mt5'):
        return MT5Summarizer(cfg, device, lang=lang)
    else:
        assert False, f"unknown model name: {name}"
