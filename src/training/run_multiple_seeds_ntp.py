#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, TrainerCallback, TrainingArguments, Trainer

from state_prediction_ntp import (
    NoPEGPT2LMHeadModel,
    FormalLanguageStateTrackingDataset,
    build_tokenizer_from_metadata,
    compute_metrics,
    customCollator,
    BASE_SEED,
)


class SeedEarlyStopCallback(TrainerCallback):
    """Track latest eval acc metrics and early-stop when in-distribution test data reaches 1.0."""

    def __init__(self, train_key: str):
        self.train_key = train_key
        self.latest_acc = {}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        for key, value in metrics.items():
            if key.endswith("acc"):
                self.latest_acc[key] = value

        key0 = f"eval_{self.train_key}_acc"
        if key0 in self.latest_acc and self.latest_acc[key0] == 1.0:
            control.should_training_stop = True


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Multi-seed runner like run_multiple_seeds_ntp.py, "
            "but using Trainer + built-in GPT2 training logic from state_prediction_ntp.py."
        )
    )
    p.add_argument("--num_run", type=int, default=10, help="Number of successful runs to collect per task")
    p.add_argument("--nope", action="store_true")
    p.add_argument("--regularize", type=float, default=0.0)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument(
        "--train_length_range",
        type=str,
        default="(0,50)",
        help="Train range as '(a,b)', e.g. '(0,50)'",
    )
    p.add_argument(
        "--test_length_range",
        type=str,
        default="(51,100),(101,150),(151,200),(201,250),(251,300),(301,350),(351,400),(401,450),(451,500)",
        help="Comma-separated test ranges, e.g. '(51,100),(101,150),(151,200)'",
    )
    p.add_argument(
        "--hparams_path",
        type=str,
        default="best_hparams.json",
        help="Optional override for best_hparams JSON path",
    )
    p.add_argument(
        "--dataset_root",
        type=str,
        default="../data",
        help="Base dataset root. Each task is read from {dataset_root}/{task}",
    )
    p.add_argument("--save_path", type=str, default="../results/multiseed_run")
    return p.parse_args()


def parse_arch(arch: str):
    if arch is None:
        return None

    lr = 1e-3 if "smalllr" not in arch else 1e-4

    lm = re.search(r"(\d+)l", arch)
    hm = re.search(r"l(\d+)h", arch)
    dm = re.search(r"h(\d+)d", arch)

    if lm and hm and dm:
        n_layer = int(lm.group(1))
        n_head = int(hm.group(1))
        d_model = int(dm.group(1))
    else:
        n_layer = 12
        n_head = 12
        d_model = 768

    if n_layer <= 4:
        max_steps = 40_000
        warmup_steps = 0
        threshold = 0.99
    else:
        max_steps = 70_000
        warmup_steps = 3000
        threshold = 0.0
    return n_layer, n_head, d_model, lr, max_steps, warmup_steps, threshold


if __name__ == "__main__":
    args = parse_args()

    random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BASE_SEED)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    hparams_path = args.hparams_path
    if args.nope and "nope" not in hparams_path:
        hparams_path = hparams_path.replace(".json", "_nope.json")
    with open(hparams_path, "r") as f:
        hparam_map = json.load(f)

    task_arch = {task: hparam_map.get(f"{task}") for task in args.tasks}

    train_length_range=[(int(min_len), int(max_len)) for min_len, max_len in re.findall(r'\((min|\d+),\s*(\d+)\)', args.train_length_range)]
    test_length_ranges=[train_length_range]+ [(int(min_len), int(max_len)) for min_len, max_len in re.findall(r'\((min|\d+),\s*(\d+)\)', args.test_length_range)]
    max_test_length = test_length_ranges[-1][1]

    batch_size = 64
    per_device_bz = batch_size // torch.cuda.device_count() if torch.cuda.is_available() else batch_size

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    if args.nope:
        suffix = "-nope"
    elif args.regularize != 0:
        suffix = f"-reg{args.regularize}"
    else:
        suffix = ""

    print("Start training...")
    for task in args.tasks:
        arch = task_arch.get(task)
        if arch is None:
            print(f"Skipping task {task}: no arch in hparams map.")
            continue

        n_layer, n_head, d_model, lr, max_steps, warmup_steps, threshold = parse_arch(arch)

        summary_path = os.path.join(save_path, f"{task}-{suffix}.txt")
        summary_f = open(summary_path, "w")
        print("\n\ntask: ", task, "\t", arch, "\n", file=summary_f)
        print(f"task: {task}  arch: {arch}")

        dataset_path = f"{args.dataset_root}/{task}"
        print("Dataset path:", dataset_path)

        tokenizer = build_tokenizer_from_metadata(dataset_path)
        print("Tokenizer:", tokenizer.vocab)

        train_dataset = FormalLanguageStateTrackingDataset(
            tokenizer,
            dataset_path,
            length_range=train_length_range,
            max_test_length=max_test_length,
            split="train",
        )

        test_dataset = {}
        for test_range in test_length_ranges:
            ds_name = f"len{test_range[0]}-{test_range[1]}"
            test_dataset[ds_name] = FormalLanguageStateTrackingDataset(
                tokenizer,
                dataset_path,
                length_range=test_range,
                max_test_length=-1,
                split="test",
            )

        results = {f"eval_len{r[0]}-{r[1]}_acc": [] for r in test_length_ranges}

        for seed in range(1000):
            print("Seed:", seed)

            n_positions = max_test_length*2 + 4
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


            # isolate RNG so only init uses this seed
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                if args.nope:
                    model = NoPEGPT2LMHeadModel(cfg)
                elif args.regularize != 0:
                    model = RegGPT2LMHeadModel(cfg, args.regularize)
                else:
                    model = GPT2LMHeadModel(cfg)

            training_args = TrainingArguments(
                output_dir=save_path,
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
                optim="adamw_torch",
                lr_scheduler_type="linear",
                warmup_steps=warmup_steps,
                report_to="none",
                dataloader_num_workers=0,
            )

            data_collator = customCollator(tokenizer.pad_token_id, tokenizer=tokenizer)
            callback = SeedEarlyStopCallback(train_key=f"len{test_length_ranges[1][0]}-{test_length_ranges[1][1]}")

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[callback],
            )

            trainer.train()
            metrics = trainer.evaluate(eval_dataset=test_dataset)

            used_steps = trainer.state.global_step
            key0 = f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"
            if metrics.get(key0, 0.0) >= threshold:
                for rkey in results:
                    results[rkey].append(metrics.get(rkey, 0.0))
                print(
                    f"{n_layer}l{n_head}h{d_model}d\t\t"
                    f"seed {seed}\tsteps {used_steps}\t\t"
                    + "\t\t".join([f"{k}: {metrics.get(k, 0.0)}" for k in results.keys()])
                    + f"\t\tlr: {lr}",
                    file=summary_f,
                )
                summary_f.flush()

            if len(results[key0]) == args.num_run:
                break

        n_ok = len(results[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"])
        if n_ok == 0:
            print("mean results\t\tno successful runs", file=summary_f)
        else:
            print(
                "mean results\t\t",
                "\t\t".join([f"{k}: {(sum(v)/n_ok):.4f}" for k, v in results.items()]),
                file=summary_f,
            )
        summary_f.flush()
        summary_f.close()
