import torch
from functools import partial
from transformers import PreTrainedTokenizer
from transformers.trainer_pt_utils import LabelSmoother


def data_collator_with_video_labels(
    batch: list[list], *,
    tokenizer: PreTrainedTokenizer = None, image_processor = None,
    model_config=None, **kwargs
):  
    v_placeholder_id = model_config.v_placeholder_id
    frame_num_tokens = model_config.frame_num_tokens

    batch = list(zip(*batch))
    batch_text, batch_frames, batch_learn_ranges, src_batch_informative_labels, src_batch_relevance_labels, \
        batch_sample_idx = batch
    
    batch = tokenizer(batch_text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt", padding=True)

    batch_labels = torch.full_like(batch.input_ids, LabelSmoother.ignore_index, dtype=torch.long)
    batch_informative_labels = torch.full_like(batch.input_ids, LabelSmoother.ignore_index, dtype=torch.long)
    batch_relevance_labels = torch.full_like(batch.input_ids, LabelSmoother.ignore_index, dtype=torch.float)

    for text, labels, informative_labels, relevance_labels, src_informative_labels, src_relevance_labels, \
        input_ids, offset_mapping, learn_range in zip(
        batch_text, batch_labels, batch_informative_labels, batch_relevance_labels, src_batch_informative_labels, src_batch_relevance_labels,
        batch.input_ids, batch.offset_mapping, batch_learn_ranges
    ):
        for learn_r in learn_range:
            start = torch.nonzero(offset_mapping[:,0] == learn_r.start).item()
            if offset_mapping[:,0][-1] >= learn_r.stop:
                stop = torch.nonzero(offset_mapping[:,0] == learn_r.stop).item()
            else: # the last eos token
                stop = len(input_ids)
            labels[start-1:stop-1] = input_ids[start:stop]

        v_placeholder_indices = torch.nonzero(input_ids == v_placeholder_id).squeeze()
        indices_to_learn = v_placeholder_indices[frame_num_tokens-1::frame_num_tokens]
        if src_informative_labels is not None:
            informative_labels[indices_to_learn] = torch.tensor(src_informative_labels, dtype=torch.long)
        if src_relevance_labels is not None:
            relevance_labels[indices_to_learn] = torch.tensor(src_relevance_labels, dtype=torch.long)

    batch['labels'] = batch_labels
    batch['informative_labels'] = batch_informative_labels
    batch['relevance_labels'] = batch_relevance_labels
    batch.pop('offset_mapping')
    batch['frames'] = torch.cat(batch_frames)
    if image_processor is not None:
        batch['frames'] = image_processor.preprocess(batch['frames'], return_tensors="pt")['pixel_values']
    batch['sample_idxs'] = torch.tensor(batch_sample_idx)
    return batch

def get_data_collator(**kwargs):
    return partial(data_collator_with_video_labels, **kwargs)
