import torch
import transformers
from typing import List
from transformers import BertTokenizer, BertModel, BertConfig
from einops import rearrange

transformers.logging.set_verbosity_error()

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# config
MAX_LENGTH = 256
DEFAULT_BERT_NAME = 'bert-base-chinese'
BERT_CONFIGS = {}

def get_tokenizer(name):
    tokenizer = BertTokenizer.from_pretrained(name, model_max_length=MAX_LENGTH)
    return tokenizer

def get_model(name):
    model = BertModel.from_pretrained(name)
    return model

def get_model_and_tokenizer(name):
    global BERT_CONFIGS

    if name not in BERT_CONFIGS:
        BERT_CONFIGS[name] = dict()
    if "model" not in BERT_CONFIGS[name]:
        BERT_CONFIGS[name]["model"] = get_model(name)
    if "tokenizer" not in BERT_CONFIGS[name]:
        BERT_CONFIGS[name]["tokenizer"] = get_tokenizer(name)
    return BERT_CONFIGS[name]['model'], BERT_CONFIGS[name]['tokenizer']

def get_encoded_dim(name):
    if name not in BERT_CONFIGS:
        config = BertConfig.from_pretrained(name)
        BERT_CONFIGS[name] = dict(config=config)
    elif "config" in BERT_CONFIGS[name]:
        config = BERT_CONFIGS[name]["config"]
    elif "model" in BERT_CONFIGS[name]:
        config = BERT_CONFIGS[name]["model"].config
    else:
        assert False
    return config.hidden_size 

# encoding text

def bert_tokenize(
    texts: List[str],
    name = DEFAULT_BERT_NAME
):
    bert, tokenizer = get_model_and_tokenizer(name)

    if torch.cuda.is_available():
        bert = bert.cuda()
    device = next(bert.parameters()).device
    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = "pt",
        padding = 'longest',
        max_length = MAX_LENGTH,
        truncation = True
    )

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)
    return input_ids, attn_mask

def bert_encode_tokenized_text(
    token_ids,
    attn_mask = None,
    pad_id = None,
    name = DEFAULT_BERT_NAME
):
    assert exists(attn_mask) or exists(pad_id)
    bert, _ = get_model_and_tokenizer(name)
    attn_mask = default(attn_mask, lambda: (token_ids != pad_id).long())
    bert.eval()
    
    with torch.no_grad():
        output = bert(input_ids = token_ids, attention_mask = attn_mask)
        encoded_text = output.last_hidden_state.detach()
    attn_mask = attn_mask.bool()
    encoded_text = encoded_text.masked_fill(~rearrange(attn_mask, '... -> ... 1'), 0.) # just force all embeddings that is padding to be equal to 0.
    return encoded_text

def bert_encode_text(
    texts: List[str],
    name = DEFAULT_BERT_NAME,
    return_attn_mask = False
):

    token_ids, attn_mask = bert_tokenize(texts, name = name)
    encoded_text = bert_encode_tokenized_text(token_ids, attn_mask = attn_mask, name = name)

    if return_attn_mask:
        attn_mask = attn_mask.bool()
        return encoded_text, attn_mask

    return encoded_text
