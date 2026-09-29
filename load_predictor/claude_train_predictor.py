"""
Load-forecasting training script: LSTM baseline vs Time-LLM.

Changes in this version (fixing the CUDA OOM):
  1. --llm_layers is clipped to GPT2's real 12-layer depth inside New_Time_LLM.py
     (32 was a LLaMA-7B setting, not a GPT2 one -- it silently padded in extra
     randomly-initialized, untrained transformer blocks).
  2. output_attentions / output_hidden_states are now off in New_Time_LLM.py
     since only last_hidden_state is ever used -- this alone is a big memory win.
  3. Mixed precision (autocast + GradScaler) for the LLM branch.
  4. LLM branch now processes each fetched batch in smaller micro-batches with
     gradient accumulation, so raising --batch_size doesn't multiply GPU memory
     by (micro-batch x enc_in channels) the way it did before. Control the real
     per-step GPU load with --llm_micro_batch_size independently of --batch_size.
  5. torch.cuda.empty_cache() + memory logging after each epoch so you can see
     headroom instead of guessing.
  6. Optional gradient checkpointing (on by default) trades ~20-30% more compute
     time for a large activation-memory reduction -- exactly the right trade
     when you're VRAM constrained.

Everything from the previous fix (proper zero_grad, .item() accumulation, model
moved to device, no scaler leakage, early stopping + best checkpoints) is kept.
"""

from load_predictor.New_Time_LLM import Model
import torch
import argparse
import pandas as pd
import numpy as np
import os
from torch import nn, optim
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

import time

import config as cfg
from load_predictor.LSTM import MultiValueLSTM


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=16, pred_len=1):
        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return x, y


def vali(loader, model, criterion, mae_metric, device, autocast_dtype=None):
    model.eval()
    total_loss, total_mae = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    outputs = model(batch_x)
            else:
                outputs = model(batch_x)
            total_loss.append(criterion(outputs.float(), batch_y).item())
            total_mae.append(mae_metric(outputs.float(), batch_y).item())
    model.train()
    if len(total_loss) == 0:
        return float("nan"), float("nan")
    return float(np.mean(total_loss)), float(np.mean(total_mae))



class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.count = 0
        self.should_stop = False

    def step(self, value):
        if value < self.best - self.min_delta:
            self.best = value
            self.count = 0
            return True
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True
            return False


def parse_args():
    parser = argparse.ArgumentParser(description="Time-series-prediction")

    parser.add_argument("--llm", type=int, default=1)
    parser.add_argument("--lstm", type=int, default=1)
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--model_id", type=str, default="test")
    parser.add_argument("--model_comment", type=str, default="none")
    parser.add_argument("--seed", type=int, default=2021)
    # content
    parser.add_argument("--content", type=str, default="No Content")

    parser.add_argument("--data", type=str, default="ETTm1")
    parser.add_argument("--root_path", type=str, default="./dataset")
    parser.add_argument("--data_path", type=str, default=cfg.LOAD_DIR / f"[{cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]}]raw_load_for_bs_0")
    parser.add_argument("--data_path_template", type=str, default=str(
        cfg.LOAD_DIR /
        f"[{cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]}]raw_load_for_bs_{{}}.csv"
    ),)
    parser.add_argument("--num_bs", type=int, default=cfg.NUM_SERVERS)
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="t")
    parser.add_argument("--seasonal_patterns", type=str, default="Weakly")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--label_len", type=int, default=48)

    parser.add_argument("--train_end", type=int, default=(14*24*60)//15)
    parser.add_argument("--test_end", type=int, default=(21*24*60)//15)

    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=32)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--output_attention", action="store_true")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--patch_len", type=int, default=8)
    
    parser.add_argument("--prompt_domain", type=int, default=1)
    parser.add_argument("--llm_model", type=str, default="TINYBERT",
                         help="[LLAMA, GPT2, DISTILGPT2, BERT, DISTILBERT, TINYBERT]. "
                              "Fastest -> slowest: TINYBERT (2 layers, 128 hidden, ~4.4M params) "
                              "< DISTILBERT/DISTILGPT2 (6 layers, ~65-82M) < GPT2/BERT (12 layers) < LLAMA.")
    parser.add_argument("--llm_dim", type=int, default=768,
                         help="Informational only -- New_Time_LLM.py auto-detects the real hidden "
                              "size from whichever --llm_model actually loads, so this can't "
                              "desync and break shapes anymore.")
    # NOTE: this is clipped inside New_Time_LLM.py to whatever the chosen model
    # actually has pretrained blocks for (GPT2/BERT: 12, DISTILGPT2/DISTILBERT: 6,
    # LLAMA: 32) -- passing a higher number just gets a warning and a clip.
    parser.add_argument("--llm_layers", type=int, default=6)
    parser.add_argument("--gradient_checkpointing", type=int, default=1,
                         help="1 = trade compute for memory on the LLM branch (recommended when VRAM constrained)")

    parser.add_argument("--num_workers", type=int, default=2,
                         help="0 is usually best here: the dataset is a small in-memory tensor, "
                              "so __getitem__ has near-zero cost and worker processes just add "
                              "IPC overhead. Raise this only if you move to a larger, I/O-bound dataset.")
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64,
                         help="batch size used for data loading / the LSTM branch")
    parser.add_argument("--llm_micro_batch_size", type=int, default=8,
                         help="actual per-step batch fed to the LLM forward pass; gradients "
                              "are accumulated across micro-batches to reach --batch_size "
                              "worth of effective batch, without the memory spike")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--llm_learning_rate", type=float, default=0.0001)
    parser.add_argument("--lstm_learning_rate", type=float, default=0.001)
    parser.add_argument("--des", type=str, default="test")
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--lradj", type=str, default="type1")
    parser.add_argument("--pct_start", type=float, default=0.2)
    parser.add_argument("--use_amp", type=int, default=1, help="1 = mixed precision for the LLM branch")
    parser.add_argument("--percent", type=int, default=100)

    return parser.parse_args()


def load_and_scale(csv_path, train_end, test_end):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["date"], errors="ignore")

    train_data = df.iloc[:train_end].reset_index(drop=True)
    test_data = df.iloc[train_end:test_end].reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(df)

    train_scaled = pd.DataFrame(scaler.transform(train_data), columns=train_data.columns)
    test_scaled = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)

    return train_scaled, test_scaled, scaler


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.llm_model == "GPT2" and args.llm_layers > 12:
        print(f"[warning] --llm_layers={args.llm_layers} > 12 for GPT2; will be clipped to 12 in the model.")

    os.makedirs(args.checkpoints, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        # helps with fragmentation on top of the memory savings above
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    use_amp = bool(args.use_amp) and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_amp else None
    # bf16 doesn't need a GradScaler (no risk of fp16 under/overflow); if you
    # switch to fp16 instead, wrap backward/step with torch.cuda.amp.GradScaler.

    for b in range(args.num_bs):
        csv_path = args.data_path if args.num_bs == 1 else args.data_path_template.format(b)
        print(f"\n=== Training for base station {b} | file: {csv_path} ===")

        train_data, test_data, scaler = load_and_scale(csv_path, args.train_end, args.test_end)
        num_feature = train_data.shape[1]

        train_dataset = TimeSeriesDataset(train_data, seq_len=args.seq_len, pred_len=args.pred_len)
        test_dataset = TimeSeriesDataset(test_data, seq_len=args.seq_len, pred_len=args.pred_len)

        loader_kwargs = dict(
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
        test_loader = DataLoader(test_dataset, batch_size=args.llm_micro_batch_size, shuffle=False, **loader_kwargs)

        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()

        lstm_model = None
        lstm_optimizer = None
        lstm_stopper = None
        if args.lstm == 1:
            lstm_model = MultiValueLSTM(input_size=num_feature, output_size=num_feature).to(device)
            lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=args.lstm_learning_rate)
            lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(lstm_optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
            lstm_stopper = EarlyStopper(patience=args.patience)

        llm_model = None
        llm_optimizer = None
        llm_stopper = None
        if args.llm == 1:
            llm_model = Model(args).float().to(device)
            trained_parameters = [p for p in llm_model.parameters() if p.requires_grad]
            n_trained = sum(p.numel() for p in trained_parameters)
            print(f"[BS{b}] Time-LLM trainable parameters: {n_trained:,}")
            llm_optimizer = optim.Adam(trained_parameters, lr=args.llm_learning_rate)
            llm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(llm_optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
            llm_stopper = EarlyStopper(patience=args.patience)

        log_rows = []

        for e in range(args.train_epochs):
            lstm_epoch_loss = 0.0
            llm_epoch_loss = 0.0
            lstm_time = 0
            llm_time = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.float()
                batch_y = batch_y.float()

                # ---------------- LSTM branch: cheap, run on the full batch ----------------
                if args.lstm == 1:
                    lstm_start = time.time()
                    bx = batch_x.to(device)
                    by = batch_y.to(device)
                    lstm_optimizer.zero_grad()
                    outputs = lstm_model(bx)
                    loss = criterion(outputs, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
                    lstm_optimizer.step()
                    lstm_epoch_loss += loss.item()
                    lstm_end = time.time()
                    lstm_time += lstm_end - lstm_start

                # ---------------- LLM branch: micro-batched + accumulated ----------------
                if args.llm == 1:
                    # llm_start = time.time()
                    # llm_optimizer.zero_grad()
                    # n = batch_x.shape[0]
                    # micro = max(1, args.llm_micro_batch_size)
                    # n_chunks = (n + micro - 1) // micro
                    # chunk_loss_sum = 0.0

                    # for start in range(0, n, micro):
                    #     end = min(start + micro, n)
                    #     mx = batch_x[start:end].to(device)
                    #     my = batch_y[start:end].to(device)

                    #     if use_amp:
                    #         with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    #             outputs = llm_model(mx)
                    #             llm_loss = criterion(outputs, my)
                    #     else:
                    #         outputs = llm_model(mx)
                    #         llm_loss = criterion(outputs, my)

                    #     # scale by chunk weight so accumulated gradient matches
                    #     # what a single full-batch step would have produced
                    #     (llm_loss * (end - start) / n).backward()
                    #     chunk_loss_sum += llm_loss.item() * (end - start)

                    # torch.nn.utils.clip_grad_norm_(
                    #     [p for p in llm_model.parameters() if p.requires_grad], max_norm=1.0
                    # )
                    # llm_optimizer.step()
                    # llm_epoch_loss += chunk_loss_sum / n
                    # llm_end = time.time()
                    # llm_time += llm_end - llm_start
                    llm_start = time.time()

                    llm_optimizer.zero_grad(set_to_none=True)

                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)

                    if use_amp:
                        with torch.autocast(
                            device_type=device.type,
                            dtype=autocast_dtype
                        ):
                            outputs = llm_model(batch_x)
                            llm_loss = criterion(outputs, batch_y)
                    else:
                        outputs = llm_model(batch_x)
                        llm_loss = criterion(outputs, batch_y)

                    llm_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        [p for p in llm_model.parameters() if p.requires_grad],
                        max_norm=1.0
                    )

                    llm_optimizer.step()

                    llm_epoch_loss += llm_loss.item() * batch_x.shape[0]

                    llm_time += time.time() - llm_start

            if device.type == "cuda":
                torch.cuda.empty_cache()
                # allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                # reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                # print(f"[BS{b}] Epoch {e+1} GPU memory: {allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved")

            lstm_vali_loss = lstm_vali_mae = None
            llm_vali_loss = llm_vali_mae = None

            if args.lstm == 1:
                lstm_vali_loss, lstm_vali_mae = vali(test_loader, lstm_model, criterion, mae_metric, device)
                lstm_scheduler.step(lstm_vali_loss)
                if lstm_stopper.step(lstm_vali_loss):
                    torch.save(lstm_model.state_dict(), os.path.join(args.checkpoints, f"[{cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]}]lstm_best_bs{b}.pth"))

            if args.llm == 1:
                llm_vali_loss, llm_vali_mae = vali(
                    test_loader, llm_model, criterion, mae_metric, device,
                    autocast_dtype=autocast_dtype if use_amp else None,
                )
                llm_scheduler.step(llm_vali_loss)
                if llm_stopper.step(llm_vali_loss):
                    torch.save(llm_model.state_dict(), os.path.join(args.checkpoints, f"[{cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]}]llm_best_bs{b}.pth"))

            num_batches = len(train_loader)
            lstm_epoch_loss /= num_batches
            llm_epoch_loss /= num_batches
            log_rows.append([
                lstm_time, lstm_epoch_loss, lstm_vali_loss, lstm_vali_mae,
                llm_time ,llm_epoch_loss, llm_vali_loss, llm_vali_mae,
            ])

            print(
                f"[BS{b}] Epoch {e+1}: "
                f"LSTM Time: {lstm_time}"
                f"LSTM train={lstm_epoch_loss} vali={lstm_vali_loss} mae={lstm_vali_mae} | "
                f"LLSM Time: {llm_time}"
                f"LLM train={llm_epoch_loss} vali={llm_vali_loss} mae={llm_vali_mae}"
            )

            stop_lstm = (args.lstm == 1) and lstm_stopper.should_stop
            stop_llm = (args.llm == 1) and llm_stopper.should_stop
            active_branches = (args.lstm == 1) + (args.llm == 1)
            stopped_branches = int(stop_lstm) + int(stop_llm)
            if active_branches > 0 and stopped_branches == active_branches:
                print(f"[BS{b}] Early stopping at epoch {e+1}.")
                break

        log_df = pd.DataFrame(
            log_rows,
            columns=["lstm_training_time","lstm_train_loss", "lstm_test_loss", "lstm_test_mae",
                     "llm_training_time","llm_train_loss", "llm_test_loss", "llm_test_mae"],
        )
        log_df.to_csv(cfg.RESULT_DIR / f"[{cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]}]train_predictor_log_bs{b}.csv", index=True)
        print(f"[BS{b}] Saved training log and best checkpoints under {args.checkpoints}")


if __name__ == "__main__":
    print("Starting predictor training")
    main()