import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm


# ─────────────────────────────────────────
# 1.  Tokenizer Wrapper
# ─────────────────────────────────────────
class SPTokenizer:
    """
    Thin wrapper around a trained SentencePiece model.
    Provides encode / decode helpers used by the Dataset.
    """
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        # Special token IDs (must match what you set in spm training)
        self.pad_id = self.sp.piece_to_id('<pad>')   # 0
        self.unk_id = self.sp.piece_to_id('<unk>')   # 1
        self.bos_id = self.sp.piece_to_id('<s>')     # 2
        self.eos_id = self.sp.piece_to_id('</s>')    # 3

        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text: str, add_bos: bool = True,
               add_eos: bool = True) -> list[int]:
        """
        Convert a raw string to a list of token IDs.
        Optionally prepend BOS and append EOS.
        """
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Convert token IDs back to a string.
        Automatically strips BOS / EOS / PAD tokens.
        """
        ids = [i for i in ids
               if i not in (self.bos_id, self.eos_id, self.pad_id)]
        return self.sp.decode(ids)

    def __len__(self):
        return self.vocab_size


# ─────────────────────────────────────────
# 2.  Translation Dataset
# ─────────────────────────────────────────
class TranslationDataset(Dataset):
    """
    PyTorch Dataset for parallel Hindi–Marathi sentence pairs.

    Args:
        src_path  : path to source language file  (one sentence per line)
        tgt_path  : path to target language file  (one sentence per line)
        tokenizer : SPTokenizer instance
        max_len   : maximum token length; longer sentences are skipped
    """
    def __init__(self, src_path: str, tgt_path: str,
                 tokenizer: SPTokenizer, max_len: int = 128):

        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.pairs     = []          # list of (src_ids, tgt_ids)

        print(f"Loading dataset from:\n  src: {src_path}\n  tgt: {tgt_path}")

        with open(src_path, 'r', encoding='utf-8') as fs, \
             open(tgt_path, 'r', encoding='utf-8') as ft:
            src_lines = fs.readlines()
            tgt_lines = ft.readlines()

        assert len(src_lines) == len(tgt_lines), \
            "Source and target files must have the same number of lines!"

        skipped = 0
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()

            # Skip empty lines
            if not src_line or not tgt_line:
                skipped += 1
                continue

            src_ids = tokenizer.encode(src_line, add_bos=True,  add_eos=True)
            tgt_ids = tokenizer.encode(tgt_line, add_bos=True,  add_eos=True)

            # Skip pairs that exceed max_len
            if len(src_ids) > max_len or len(tgt_ids) > max_len:
                skipped += 1
                continue

            self.pairs.append((src_ids, tgt_ids))

        print(f"  Loaded : {len(self.pairs):,} pairs")
        print(f"  Skipped: {skipped:,} pairs (empty or too long)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.pairs[idx]
        return (torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long))


# ─────────────────────────────────────────
# 3.  Collate Function (handles padding)
# ─────────────────────────────────────────
def collate_fn(batch, pad_id: int = 0):
    """
    Called by DataLoader to assemble a batch.

    Each sentence in a batch may have a different length.
    We pad all of them to the length of the longest one
    so they can be stacked into a single tensor.

    Returns:
        src_tensor : (batch_size, max_src_len)  — padded source sequences
        tgt_tensor : (batch_size, max_tgt_len)  — padded target sequences
        src_lengths: (batch_size,)               — original source lengths
        tgt_lengths: (batch_size,)               — original target lengths
    """
    src_batch, tgt_batch = zip(*batch)

    src_lengths = torch.tensor([len(s) for s in src_batch], dtype=torch.long)
    tgt_lengths = torch.tensor([len(t) for t in tgt_batch], dtype=torch.long)

    # pad_sequence pads to the longest sequence in the batch
    src_padded = pad_sequence(src_batch, batch_first=True,
                              padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True,
                              padding_value=pad_id)

    return src_padded, tgt_padded, src_lengths, tgt_lengths


# ─────────────────────────────────────────
# 4.  DataLoader Factory
# ─────────────────────────────────────────
def get_dataloader(src_path: str, tgt_path: str,
                   tokenizer: SPTokenizer,
                   batch_size: int = 64,
                   max_len:    int = 128,
                   shuffle:    bool = True,
                   num_workers:int = 2) -> DataLoader:
    """
    Convenience function that builds a Dataset and wraps it
    in a DataLoader ready for training.

    Args:
        src_path    : source language file
        tgt_path    : target language file
        tokenizer   : SPTokenizer instance
        batch_size  : number of sentence pairs per batch
        max_len     : sentences longer than this are skipped
        shuffle     : shuffle data each epoch (True for train, False for val/test)
        num_workers : parallel workers for data loading

    Returns:
        PyTorch DataLoader
    """
    dataset = TranslationDataset(src_path, tgt_path, tokenizer, max_len)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
        pin_memory=True,       # faster GPU transfer
    )

    print(f"  Batches per epoch: {len(loader):,}  "
          f"(batch_size={batch_size})\n")
    return loader


# ─────────────────────────────────────────
# 5.  Quick Sanity Test
# ─────────────────────────────────────────
if __name__ == '__main__':
    # Run this cell directly to verify everything works
    # Update paths to match your Kaggle environment

    SPM_MODEL  = "/kaggle/working/spm_hi_mr.model"
    TRAIN_SRC  = "/kaggle/working/train.hi"
    TRAIN_TGT  = "/kaggle/working/train.mr"

    print("=" * 50)
    print("DATASET SANITY CHECK")
    print("=" * 50)

    # Load tokenizer
    tokenizer = SPTokenizer(SPM_MODEL)
    print(f"\nTokenizer vocab size: {len(tokenizer)}")
    print(f"PAD id : {tokenizer.pad_id}")
    print(f"BOS id : {tokenizer.bos_id}")
    print(f"EOS id : {tokenizer.eos_id}")

    # Build DataLoader
    loader = get_dataloader(
        TRAIN_SRC, TRAIN_TGT,
        tokenizer=tokenizer,
        batch_size=4,
        max_len=128,
        shuffle=False,
        num_workers=0,
    )

    # Inspect first batch
    src, tgt, src_lens, tgt_lens = next(iter(loader))

    print("\nFirst batch shapes:")
    print(f"  src : {src.shape}   (batch_size x max_src_len)")
    print(f"  tgt : {tgt.shape}   (batch_size x max_tgt_len)")
    print(f"  src_lengths: {src_lens.tolist()}")
    print(f"  tgt_lengths: {tgt_lens.tolist()}")

    print("\nDecoded source sentences:")
    for i in range(len(src)):
        print(f"  [{i}] {tokenizer.decode(src[i].tolist())}")

    print("\nDecoded target sentences:")
    for i in range(len(tgt)):
        print(f"  [{i}] {tokenizer.decode(tgt[i].tolist())}")

    print("\n✅ Dataset module working correctly!")
