import torch
from torch.utils.data import DataLoader

from data.tokenizer import Tokenizer
from data.special_tokens import SpecialTokens
from data.vocabulary import Vocabulary
from data.special_token_ids import SpecialTokenIds
from data.sequence_rules import SequenceRules
from data.sequence_encoder import SequenceEncoder
from data.translation_dataset import TranslationDataset
from data.batch_collator import BatchCollator


def main():
    print("=== SANITY CHECK START ===")

    rows = [
        {"src": "I like apples", "tgt": "me gustan las manzanas"},
        {"src": "I eat", "tgt": "yo como"},
    ]

    train_texts = [r["src"] for r in rows] + [r["tgt"] for r in rows]

    tokenizer = Tokenizer()
    specials = SpecialTokens()
    vocab = Vocabulary(tokenizer, specials)
    vocab.build(train_texts)

    print("\n[VOCAB SIZE]", len(vocab))

    special_ids = SpecialTokenIds(vocab).as_dict()
    print("\n[SPECIAL IDS]")
    for k, v in special_ids.items():
        print(f"{k}: {v}")

    rules = SequenceRules(special_ids)
    encoder = SequenceEncoder(vocab, rules)

    dataset = TranslationDataset(rows, encoder)

    src_ids, tgt_ids = dataset[0]

    print("\n[DATASET ITEM 0]")
    print("src_ids:", src_ids.tolist())
    print("tgt_ids:", tgt_ids.tolist())

    assert src_ids[-1].item() == special_ids["eos"], "❌ source must end with EOS"
    assert tgt_ids[0].item() == special_ids["bos"], "❌ target must start with BOS"
    assert tgt_ids[-1].item() == special_ids["eos"], "❌ target must end with EOS"

    collator = BatchCollator(pad_id=special_ids["pad"])

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collator.collate,
    )

    batch = next(iter(loader))

    print("\n[BATCH]")
    print("src_ids:\n", batch.src_ids)
    print("src_mask:\n", batch.src_mask)
    print("tgt_ids:\n", batch.tgt_ids)
    print("tgt_mask:\n", batch.tgt_mask)

    pad_id = special_ids["pad"]

    for i in range(batch.src_ids.shape[0]):
        for j in range(batch.src_ids.shape[1]):
            if not batch.src_mask[i, j]:
                assert batch.src_ids[i, j].item() == pad_id, \
                    "src_mask False but token is not PAD"

    for i in range(batch.tgt_ids.shape[0]):
        for j in range(batch.tgt_ids.shape[1]):
            if not batch.tgt_mask[i, j]:
                assert batch.tgt_ids[i, j].item() == pad_id, \
                    "tgt_mask False but token is not PAD"

    print("\nALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()
