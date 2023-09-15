from itertools import chain, repeat
import jax.numpy as jnp
from transformers import LlamaTokenizer

from ..data import TrainData, TestData

def gsm_collate_fn_train(tokenizer: LlamaTokenizer, max_len: int, data_batch: list[tuple[str, str]]):
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    seq_list = []
    seq_mask_list = []
    labels_list = []
    labels_mask_list = []

    for question, answer in data_batch:
        question = tokenizer(question, add_special_tokens=False, return_attention_mask=False).input_ids
        answer = tokenizer(answer, add_special_tokens=False, return_attention_mask=False).input_ids

        len_question = len(question)
        len_answer = len(answer)
        len_seq = len_question + len_answer + 2
        len_pad = max_len - len_seq

        assert len(question) + 1 < max_len, '`max_len` too small'

        seq = list(chain((bos_id,), question, answer, (eos_id,), repeat(eos_id, len_pad)))
        seq_mask = list(chain(repeat(True, 1 + len_question + len_answer + 1), repeat(False, len_pad)))

        labels = list(chain(question, answer, (eos_id,), repeat(eos_id, len_pad + 1)))
        labels_mask = list(chain(repeat(False, len_question), repeat(True, len_answer + 1), repeat(False, len_pad + 1)))

        seq = seq[:max_len]
        seq_mask = seq_mask[:max_len]
        labels = labels[:max_len]
        labels_mask = labels_mask[:max_len]

        seq_list.append(seq)
        seq_mask_list.append(seq_mask)
        labels_list.append(labels)
        labels_mask_list.append(labels_mask)

    seq_ = jnp.array(seq_list, dtype=jnp.uint16)
    seq_mask_ = jnp.array(seq_mask_list, dtype=jnp.bool_)
    labels_ = jnp.array(labels_list, dtype=jnp.uint16)
    labels_mask_ = jnp.array(labels_mask_list, dtype=jnp.bool_)

    return TrainData(seq_, seq_mask_, labels_, labels_mask_)

def gsm_collate_fn_test(tokenizer: LlamaTokenizer, max_len: int, data_batch: list[tuple[str, str]]):
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    seq_list = []
    seq_mask_list = []
    labels_list = []

    for question, answer in data_batch:
        question = tokenizer(question, add_special_tokens=False, return_attention_mask=False).input_ids

        len_question = len(question)
        len_seq = len_question + 1
        len_pad = max_len - len_seq

        assert len(question) + 1 < max_len, '`max_len` too small'

        seq = list(chain((bos_id,), question, repeat(eos_id, len_pad)))
        seq_mask = list(chain(repeat(True, 1 + len_question), repeat(False, len_pad)))

        seq = seq[:max_len]
        seq_mask = seq_mask[:max_len]

        seq_list.append(seq)
        seq_mask_list.append(seq_mask)
        labels_list.append(answer)

    seq_ = jnp.array(seq_list, dtype=jnp.uint16)
    seq_mask_ = jnp.array(seq_mask_list, dtype=jnp.bool_)

    return TestData(seq_, seq_mask_, labels_list)
