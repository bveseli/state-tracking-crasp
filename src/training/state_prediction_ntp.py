import os
from transformers import GPT2LMHeadModel, GPT2Config, TrainingArguments, Trainer, TrainerCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from copy import deepcopy
import argparse
from datetime import datetime

BASE_SEED = 0  # shared default seed for reproducibility

class NoPE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return 0

class NoPEGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer.wpe = NoPE()

class customTokenizer():
    def __init__(self, vocab: list[str]):
        normal_tkn_num = len(vocab) # each element is a token

        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.and_token = "&"
        self.hashtag_token = "#"
        self.bos_token_id = normal_tkn_num
        self.eos_token_id = normal_tkn_num + 1
        self.pad_token_id = normal_tkn_num + 2
        self.and_id = normal_tkn_num + 3
        self.hashtag_id = normal_tkn_num + 4
        self.special_token_ids = [self.bos_token_id, self.eos_token_id, self.pad_token_id, self.and_id, self.hashtag_id]
        self.special_tokens = [self.bos_token,  self.eos_token, self.pad_token, self.and_token, self.hashtag_token]
        assert all(t not in vocab for t in self.special_tokens)
        
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.vocab[self.bos_token] = self.bos_token_id
        self.vocab[self.eos_token] = self.eos_token_id
        self.vocab[self.pad_token] = self.pad_token_id
        self.vocab[self.and_token] = self.and_id
        self.vocab[self.hashtag_token] = self.hashtag_id

        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.padding_side = "right"

    def __call__(self, strings: list[str] | str, **kwargs):

        if type(strings) == str:
            strings = [strings]
        ids = []
        strings = [list(s) for s in strings]
        max_len = max(map(lambda x: len(x), strings))
        for s in strings:
            ids.append( list(map(lambda x: self.vocab[x], s)) + [self.pad_token_id] * (max_len-len(s)) )

        return {"input_ids": torch.LongTensor(ids)}

    def convert_ids_to_tokens(self, ids: list[int], rm_special=False):
        if rm_special:
            return [self.vocab_inv[i] for i in ids if i not in self.special_token_ids]
        else:
            return list(map(lambda x: self.vocab_inv[x], ids))

    def __len__(self):
        return len(self.vocab)



def build_tokenizer_from_metadata(task_root: str) -> customTokenizer:
    meta_path = f"{task_root}/meta_data.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    symbols = meta["symbols"]
    states = meta["states"]

    vocab = list(set(symbols+states))

    return customTokenizer(vocab)


class FormalLanguageStateTrackingDataset(Dataset):
    def __init__(self, tokenizer: customTokenizer, dataset_path: str, length_range: tuple[int, int], max_test_length: int, split: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.range_min, self.range_max = length_range
        self.max_test_length = max_test_length
        
        self.processed_examples = []
        
        # Determine files to load
        prefix = "train_" if split == "train" else "test_"
        dataset_files = [f for f in os.listdir(dataset_path) if f.startswith(prefix) and f.endswith(".jsonl")]
        print("Dataset files:", dataset_files)
        
        if not dataset_files:
            raise ValueError(f"No JSONL files found in {dataset_path} for split {split}")

        # Load and Pre-process everything into memory
        for dataset_file in dataset_files:
            file_path = os.path.join(dataset_path, dataset_file)
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    seq = data["word"]
                    states = data["states"] 
                    
                    if self.range_min <= len(seq) <= self.range_max:
                        self.processed_examples.append(self._prepare_item(seq, states))

        
        print("Number of samples:", len(self.processed_examples))
        if not self.processed_examples:
            raise ValueError(f"No examples found in {dataset_path} within range {length_range}")

    def _split_states(self, s: str) -> list[str]:
        # All vocab tokens that look like states, ordered longest-first
        tokens = sorted(
            [t for t in self.tokenizer.vocab.keys() if t.startswith("q")],
            key=len,
            reverse=True,
        )

        res = []
        i = 0
        while i < len(s):
            matched = False
            for t in tokens:
                if s.startswith(t, i):
                    res.append(t)
                    i += len(t)
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Cannot match at position {i}: {s[i:]}")
        return res

    def _prepare_item(self, seq, target_seq):
        """Helper to format sequence, target, and position IDs."""

        # Format Input: <bos>&a&a&a&a&<eos><pad>
        instance = [self.tokenizer.bos_token_id]

        for char in seq:
            instance.append(self.tokenizer.and_id)
            instance.append(self.tokenizer.vocab[char])

        instance.append(self.tokenizer.and_id)
        instance.append(self.tokenizer.eos_token_id)
        instance.append(self.tokenizer.pad_token_id)

        # Format Target: <pad>#q1#q0#q1#q0#q1#
        state_tokens = self._split_states(target_seq)# target_seq is something like "q1q0q1q0q1"
        label_seq = [self.tokenizer.pad_token_id]  # <pad>
        for t in state_tokens:
            label_seq.append(self.tokenizer.hashtag_id)      # '#'
            label_seq.append(self.tokenizer.vocab[t])        # q*
        label_seq.append(self.tokenizer.hashtag_id)          # final '#'
        
        return instance, label_seq

    def __len__(self):
        return len(self.processed_examples)

    def __getitem__(self, idx):
        instance, label_seq = self.processed_examples[idx]
        
        pos_ids = list(range(len(instance)))
        if self.max_test_length != -1:
            offset = random.randint(0, max(0, self.max_test_length - len(instance)))
            pos_ids = [p + offset for p in pos_ids]
            
        return deepcopy(instance), deepcopy(pos_ids), deepcopy(label_seq)


class customCollator():
    def __init__(self, pad_id, tokenizer=None):
        self.pad_id = pad_id
        self.tokenizer = tokenizer

    def __call__(self, examples):
        input_ids, pos_ids, labels = tuple(zip(*examples))
        max_len = max(len(item) for item in input_ids)

        [item.extend([self.pad_id,] * (max_len - len(item))) for item in input_ids]
        input_ids = torch.LongTensor(input_ids)

        # Attention mask: mask out padding and the separator token `and_id`.
        attention_mask = (input_ids != self.pad_id)
        if self.tokenizer is not None and hasattr(self.tokenizer, "and_id"):
            attention_mask = attention_mask & (input_ids != self.tokenizer.and_id)
        attention_mask = attention_mask.long()

        [item.extend([self.pad_id,] * (max_len - len(item))) for item in labels]
        labels = torch.LongTensor(labels)
        labels[labels == self.pad_id] = -100
        labels[labels == self.tokenizer.hashtag_id] = -100


        [item.extend([item[-1],] * (max_len - len(item))) for item in pos_ids]
        pos_ids = torch.LongTensor(pos_ids)
        
        batch = {"input_ids": input_ids, "position_ids": pos_ids, "labels": labels, "attention_mask": attention_mask}
        return batch


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    predictions = np.argmax(shift_logits, axis=-1)

    correct = np.all((predictions == shift_labels) | (shift_labels == -100), axis=1)
    return {"acc": correct.sum() / len(correct)}


class myCallback(TrainerCallback):
    def on_evaluate(self, state, args, control, metrics=None, logs=None, eval_dataloader=None, **kwargs):
        assert metrics["epoch"] >= getattr(self, "current_epoch", 0)
        if metrics["epoch"] > getattr(self, "current_epoch", 0):
            self.latest_acc = {}
            self.current_epoch = metrics["epoch"]
        for key in metrics.keys():
            if key.endswith("acc"):
                self.latest_acc[key] = metrics[key]
        if len(self.latest_acc) == len(test_length_ranges):
            if (self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] == 1.0) or (self.current_epoch == 1.0):  
                if self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] == 1.0: 
                    control.should_training_stop = True
                    msg = f"early stop {self.current_epoch}\t\t"
                else:
                    msg = "reach max step\t\t"
                if self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] >= 0.99:
                    msg = ">> " + msg
                print(f"{n_layer}l{n_head}h{d_model}d\t\t", msg, "\t\t".join([f"{k}: {v}" for k, v in self.latest_acc.items()]), f"\t\tlr: {lr}", file=summary_f)
                summary_f.flush()

                # If longest bin reaches 100%, mark to stop the sweep
                if (self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] == 1.0) and (self.latest_acc.get(f"eval_len{test_length_ranges[-1][0]}-{test_length_ranges[-1][1]}_acc", 0.0) == 1.0):
                    global stop_sweep
                    stop_sweep = True


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--nope", action="store_true")
    parser.add_argument("--regularize", type=float, default=0.0)
    parser.add_argument("--big_model", action="store_true")
    parser.add_argument("--biggest_model", action="store_true")
    parser.add_argument("--train_length_range", type=str, default="(0,50)")
    parser.add_argument("--test_length_range", type=str, default="(51,100),(101,150),(151,200),(201,250),(251,300),(301,350),(351,400),(401,450),(451,500)")
    parser.add_argument("--dataset_root", type=str, default="../data/")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(BASE_SEED)
    random.seed(BASE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BASE_SEED)

    train_length_range=[(int(min_len), int(max_len)) for min_len, max_len in re.findall(r'\((min|\d+),\s*(\d+)\)', args.train_length_range)]
    test_length_ranges=[train_length_range]+ [(int(min_len), int(max_len)) for min_len, max_len in re.findall(r'\((min|\d+),\s*(\d+)\)', args.test_length_range)]
    max_test_length = test_length_ranges[-1][1]

    batch_size = 64
    per_device_bz = batch_size // torch.cuda.device_count() if torch.cuda.is_available() else batch_size 
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    dataset_path = f"{args.dataset_root}{args.task}"
    print("Dataset path:", dataset_path)

    
    if args.big_model:
        #configs = [(12, 12, 768, 1e-4)] # big model config
        # Valid combinations: d_model must be divisible by n_head
        configs = [(l, h, d, lr) for l in [6, 8, 12] for h in [4, 8] for d in [64, 256] for lr in [1e-3, 1e-4]]  # h=4,8 work with all d
        configs.append((12, 12, 768, 1e-4))
        configs.append((12, 12, 768, 1e-3))
    elif args.biggest_model:
        configs = [
            (16, 8, 768, 1e-4),
            (16, 8, 768, 1e-3),
            (16, 12, 768, 1e-4),
            (16, 12, 768, 1e-3),
            (18, 18, 1152, 1e-4),
            (18, 18, 1152, 1e-3),
            (24, 16, 1024, 1e-4),
            (24, 16, 1024, 1e-3),
        ]
    else:
        configs = [(l, h, d, lr) for l in [1, 2, 4] for h in [1, 2, 4] for d in [16, 64, 256] for lr in [1e-3, 1e-4]] # small model config


    tokenizer = build_tokenizer_from_metadata(dataset_path)
    print("Tokenizer:", tokenizer.vocab)


    # Create the train dataset
    train_dataset = FormalLanguageStateTrackingDataset(
        tokenizer, dataset_path, length_range=train_length_range, 
        max_test_length=max_test_length, split="train"
    )
    print("Train dataset length:", len(train_dataset))

    # Create the test datasets (no more EvalDataset wrapper needed)
    test_dataset = {}
    for test_range in test_length_ranges:
        ds_name = f"len{test_range[0]}-{test_range[1]}"
        test_dataset[ds_name] = FormalLanguageStateTrackingDataset(
            tokenizer, dataset_path, length_range=test_range, 
            max_test_length=-1, split="test"
        )
        print("Test dataset length:", len(test_dataset[ds_name]))
    
    task_path = f"../results/hyperparam_search/out-{args.task}"
    if not os.path.exists(task_path):
        os.mkdir(task_path)
    if args.nope:
        suffix = "-nope"
    elif args.regularize != 0:
        suffix = f"-reg{args.regularize}"
    else:
        suffix = ""
    

    max_n_layer = max(n_layer for n_layer, _, _, _ in configs)
    if max_n_layer >= 16:
        summary_prefix = "biggest"
    elif max_n_layer > 4:
        summary_prefix = "big"
    else:
        summary_prefix = "summary"
    summary_f = open(os.path.join(task_path, f"{summary_prefix}_{len(train_dataset)}trainsmpls{suffix}-{timestamp}.txt"), "w")

    for i in range(3):
        print("\ninput example:")
        print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[f"len{test_length_ranges[0][0]}-{test_length_ranges[0][1]}"][i][0])))
        print("label example:")
        print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[f"len{test_length_ranges[0][0]}-{test_length_ranges[0][1]}"][i][2])))

    # No global early-stop flags; run all configs.
    stop_sweep = False
    for n_layer, n_head, d_model, lr in configs: 
        print(f"Training {n_layer}l{n_head}h{d_model}d{'smalllr' if lr == 1e-4 else ''}")

        if n_layer > 4:
            max_steps = 70_000
            warmup_steps = 3000
        else:
            max_steps = 40_000
            warmup_steps = 0

        output_dir = f"{n_layer}l{n_head}h{d_model}d{'smalllr' if lr == 1e-4 else ''}{suffix}"
        output_dir = os.path.join(task_path, output_dir)

        n_positions =max_test_length*2 + 4
        cfg = GPT2Config(vocab_size=len(tokenizer), 
                    n_positions=n_positions*2,
                    n_embd=d_model,
                    n_layer=n_layer,
                    n_head=n_head,
                    bos_token_id=tokenizer.bos_token_id, 
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    attn_pdrop=0,
                    resid_pdrop=0,
                    embd_pdrop=0,
                    )

        if args.nope:
            model = NoPEGPT2LMHeadModel(cfg)
        else:
            model = GPT2LMHeadModel(cfg)

        training_args = TrainingArguments(
            output_dir=output_dir,    
            overwrite_output_dir=True,
            per_device_train_batch_size=per_device_bz,
            per_device_eval_batch_size=per_device_bz,
            max_steps=max_steps,
            evaluation_strategy="steps",
            eval_steps=3_000,
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=3_000,
            learning_rate=lr,
            weight_decay=0.01,
            optim='adamw_torch',
            lr_scheduler_type='linear',
            warmup_steps=warmup_steps,
            report_to="none",
        )

        data_collator = customCollator(tokenizer.pad_token_id, tokenizer=tokenizer)
        #_current_task[0] = args.task

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[myCallback],
        )

        trainer.train()

        # Always run all configs; no sweep-level early stop.
        if stop_sweep:
            break

    
    summary_f.close()