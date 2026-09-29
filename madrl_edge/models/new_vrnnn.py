"""
Variational RNN used by the UPF to compress a 16-step HISTORY of global load
vectors into a single embedding that represents the (compressed) PREDICTED
NEXT load state. This is a predictive/forecasting VRNN, not a plain
autoencoder.

-----------------------------------------------------------------------------
DESIGN (resolves the earlier server-vs-time sequence-axis ambiguity)
-----------------------------------------------------------------------------
Sequence axis = TIME. At every timestep, "load" is the GLOBAL load vector for
that slot -- i.e. the concatenation of every server's LOCAL_LOAD_DIM vector
into one flat vector of length NUM_SERVERS * LOCAL_LOAD_DIM. This matches how
FrozenVRNNEncoder.compress() was already building its rolling history, so
config.VRNN_INPUT_DIM should equal NUM_SERVERS * LOCAL_LOAD_DIM and
config.VRNN_SEQ_LEN should be 16 (the requested input history length).

TRAINING: each sample is 17 consecutive slots -- 16 known history steps plus
the true next slot appended as step 17. The model runs its normal VRNN
recurrence (prior -> encoder -> reparameterize -> decode) over all 17 steps,
because during training the true next-step value IS available and can be
fed to the encoder like any other step. The reconstruction loss at step 17
therefore directly supervises the model to predict the true next load --
no separate prediction loss is needed, it falls out of the standard ELBO.

INFERENCE (FrozenVRNNEncoder.compress -> uses model._step_prior_only): only
the 16 known history steps exist. The recurrence runs normally (with the
encoder) over those 16 steps, then ONE extra step is taken using ONLY the
prior network (no encoder, since the true next-step value is exactly what we
don't have yet) to sample z_17 and decode a predicted next load. z_17 is the
"compressed next load" representation broadcast to every server; the decoded
vector is the raw predicted next load itself, returned alongside it for
convenience (e.g. diagnostics, or reward shaping that wants the raw
forecast).

CONFUSION: I've assumed "load" here means the GLOBAL (all-servers-flattened)
vector, since that's what the encoder-side code already built historically.
If you actually want one VRNN per server (each predicting its own next local
load from its own 16-step history) rather than one shared VRNN over the
concatenated global vector, the fix is straightforward -- input_dim becomes
LOCAL_LOAD_DIM and you instantiate/run one model per server -- but please
confirm, since it changes both compute cost and what "compressed" means
(one global summary vs. per-server summaries).
-----------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

from madrl_edge import config

from pandas import DataFrame

from collections import deque
import config as cfg
# --- NaN-prevention constants (see prior fix) ----------------------------
# Hard bounds on any log-variance the network produces. Without this, the
# prior/encoder logvar heads can drift arbitrarily large (exp() overflow ->
# inf) or arbitrarily negative (exp() underflow -> divide-by-zero in the
# KLD term), which is the classic way this kind of VRNN goes NaN.
LOGVAR_MIN = -10.0
LOGVAR_MAX = 10.0
GRAD_CLIP_NORM = 5.0


class VRNN(nn.Module):
    def __init__(self, input_dim=cfg.VRNN_INPUT_DIM,
                 hidden_dim=cfg.VRNN_HIDDEN_DIM,
                 latent_dim=cfg.VRNN_LATENT_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # feature extractors
        self.phi_x = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.phi_z = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())

        # prior p(z_t | h_{t-1}) -- used every step during training, and
        # used ALONE (no encoder) for the one-step-ahead prediction at
        # inference time, since the true future value isn't available then.
        self.prior = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)

        # encoder q(z_t | x_t, h_{t-1}) -- only usable when x_t is known
        # (training, or any of the 16 known history steps at inference)
        self.enc = nn.Sequential(nn.Linear(hidden_dim + hidden_dim, hidden_dim), nn.ReLU())
        self.enc_mean = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # decoder p(x_t | z_t, h_{t-1})
        self.dec = nn.Sequential(nn.Linear(hidden_dim + hidden_dim, hidden_dim), nn.ReLU())
        self.dec_mean = nn.Linear(hidden_dim, input_dim)

        # recurrence
        self.rnn_cell = nn.GRUCell(hidden_dim + hidden_dim, hidden_dim)

    @staticmethod
    def _reparam(mean, logvar):
        logvar = torch.clamp(logvar, LOGVAR_MIN, LOGVAR_MAX)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def _step_with_encoder(self, x_t, h):
        """One VRNN step where x_t IS known (used for all training steps,
        and for the 16 known history steps at inference)."""
        phi_x_t = self.phi_x(x_t)

        prior_h = self.prior(h)
        prior_mean_t = self.prior_mean(prior_h)
        prior_logvar_t = torch.clamp(self.prior_logvar(prior_h), LOGVAR_MIN, LOGVAR_MAX)

        enc_h = self.enc(torch.cat([phi_x_t, h], dim=-1))
        enc_mean_t = self.enc_mean(enc_h)
        enc_logvar_t = torch.clamp(self.enc_logvar(enc_h), LOGVAR_MIN, LOGVAR_MAX)

        z_t = self._reparam(enc_mean_t, enc_logvar_t)
        phi_z_t = self.phi_z(z_t)

        dec_h = self.dec(torch.cat([phi_z_t, h], dim=-1))
        x_hat_t = self.dec_mean(dec_h)

        nll_t = F.mse_loss(x_hat_t, x_t, reduction="mean")
        kld_t = self._kld(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)

        h_next = self.rnn_cell(torch.cat([phi_x_t, phi_z_t], dim=-1), h)
        return h_next, z_t, x_hat_t, nll_t, kld_t

    def _step_prior_only(self, h):
        """One VRNN step where x_t is UNKNOWN (the real next-step prediction
        at inference time). No encoder is used -- z is sampled purely from
        the prior conditioned on the history seen so far, then decoded.
        There's no true x_t to compare against, so no loss is computed here.
        """
        prior_h = self.prior(h)
        prior_mean_t = self.prior_mean(prior_h)
        prior_logvar_t = torch.clamp(self.prior_logvar(prior_h), LOGVAR_MIN, LOGVAR_MAX)

        z_t = self._reparam(prior_mean_t, prior_logvar_t)
        phi_z_t = self.phi_z(z_t)

        dec_h = self.dec(torch.cat([phi_z_t, h], dim=-1))
        x_hat_t = self.dec_mean(dec_h)
        return z_t, x_hat_t

    def forward(self, x_seq):
        """
        TRAINING forward pass. x_seq: (batch, seq_len, input_dim) where
        seq_len is 17 -- 16 known history steps plus the true next step
        appended as the last element. Every step uses the encoder (the true
        value is known throughout training), so the reconstruction loss at
        the final step directly supervises next-load prediction.

        Returns: (nll, kld, z_final, x_hat_final)
          nll / kld   -- averaged over all seq_len steps (standard ELBO)
          z_final     -- the compressed latent for the LAST step, i.e. the
                         compressed "next load" representation
          x_hat_final -- the decoded raw prediction for the LAST step
        """
        batch, seq_len, _ = x_seq.shape
        h = torch.zeros(batch, self.hidden_dim, device=x_seq.device)

        kld_total = 0.0
        nll_total = 0.0
        z_final = None
        x_hat_final = None

        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            h, z_t, x_hat_t, nll_t, kld_t = self._step_with_encoder(x_t, h)
            nll_total = nll_total + nll_t
            kld_total = kld_total + kld_t
            z_final = z_t
            x_hat_final = x_hat_t

        return nll_total / seq_len, kld_total / seq_len, z_final, x_hat_final

    @staticmethod
    def _kld(mean1, logvar1, mean2, logvar2):
        # KL( N(mean1,var1) || N(mean2,var2) ), computed via exp(logvar1 -
        # logvar2) rather than exp(logvar1)/exp(logvar2) separately, which
        # is more numerically stable, plus an epsilon floor as a second
        # line of defense against divide-by-(near)zero.
        eps = 1e-6
        var_ratio = torch.exp(logvar1 - logvar2)
        sq_diff_term = (mean1 - mean2) ** 2 / (torch.exp(logvar2) + eps)
        return 0.5 * torch.mean(logvar2 - logvar1 + var_ratio + sq_diff_term - 1)


class FrozenVRNNEncoder:
    """
    Inference-only wrapper used during MADDPG training/execution once the
    VRNN has been pretrained. Maintains a rolling 16-step history of global
    load vectors and, on each call, produces a compressed representation of
    the PREDICTED NEXT load (not a reconstruction of the current slot).
    """

    def __init__(self, checkpoint_path: str, device="cpu"):
        self.device = device
        self.model = VRNN().to(device)
        state = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # Rolling history of the 16 most recent KNOWN global load vectors.
        self.history = deque(
            [[0.0] * cfg.VRNN_INPUT_DIM for _ in range(cfg.VRNN_SEQ_LEN)],
            maxlen=cfg.VRNN_SEQ_LEN
        )

    @torch.no_grad()
    def compress(self, all_server_loads, return_prediction=False):
        """
        all_server_loads: list of NUM_SERVERS lists, each config.LOCAL_LOAD_DIM
        long -- the raw per-server load vectors for the CURRENT slot.

        Appends the current slot to the rolling 16-step history, runs the
        encoder over those 16 known steps, then takes one prior-only step to
        produce a compressed representation of the PREDICTED NEXT load.

        Returns a GLOBAL_LOAD_DIM python list (the compressed next-load
        embedding) by default. If return_prediction=True, returns a tuple
        (compressed_embedding, predicted_next_raw_load) instead, where the
        second element is the decoded raw next-load vector (length
        config.VRNN_INPUT_DIM) -- useful for diagnostics or forecast-aware
        reward shaping.
        """
        loads = []
        for load in all_server_loads:
            if len(load) != cfg.LOCAL_LOAD_DIM:
                raise ValueError(f"Expected load vector of length {cfg.LOCAL_LOAD_DIM}, got {len(load)}")
            loads.extend(load)

        if any((v != v) or abs(v) == float("inf") for v in loads):  # v != v catches NaN
            raise ValueError(
                "FrozenVRNNEncoder.compress() received NaN/Inf in all_server_loads. "
                "Check the upstream scaler/normalization for zero-variance features "
                "before this point -- the VRNN itself did not produce this."
            )

        self.history.append(loads)

        x = torch.tensor([list(self.history)], dtype=torch.float32, device=self.device)  # (1, 16, input_dim)

        h = torch.zeros(1, self.model.hidden_dim, device=self.device)
        for t in range(x.shape[1]):
            h, _, _, _, _ = self.model._step_with_encoder(x[:, t, :], h)

        z_next, x_hat_next = self.model._step_prior_only(h)

        embedding = z_next.squeeze(0).cpu().tolist()
        if return_prediction:
            return embedding, x_hat_next.squeeze(0).cpu().tolist()
        return embedding

    def reset(self):
        self.history.clear()
        self.history.extend([[0.0] * cfg.VRNN_INPUT_DIM for _ in range(cfg.VRNN_SEQ_LEN)])


def train_vrnn(history_dataset, next_dataset, epochs=50, lr=1e-3, device="cpu", save_path="vrnn.pt"):
    """
    history_dataset: array-like of shape (N, 16, VRNN_INPUT_DIM) -- the 16
        known history steps for each training sample.
    next_dataset:    array-like of shape (N, VRNN_INPUT_DIM) -- the TRUE
        next-step global load vector for each sample (i.e. the label the
        model should learn to predict/reconstruct at step 17).

    Internally, each sample's history and its true next step are
    concatenated into one length-17 sequence and fed through VRNN.forward,
    so the reconstruction loss at the final step directly supervises
    next-load prediction -- this is what makes training match how
    FrozenVRNNEncoder.compress() is actually used at inference (16 known
    steps in, compressed prediction of the 17th out).

    CONFUSION: how you assemble (history_dataset, next_dataset) from raw
    simulated/logged traffic -- e.g. a sliding window over a longer trace --
    isn't specified here; plug your task-arrival generator's output into a
    16-step sliding window (with the 17th slot as the label) before calling
    this.
    """
    history_tensor = torch.tensor(history_dataset, dtype=torch.float32, device=device)  # (N, 16, D)
    next_tensor = torch.tensor(next_dataset, dtype=torch.float32, device=device)          # (N, D)

    if history_tensor.shape[0] != next_tensor.shape[0]:
        raise ValueError(
            f"history_dataset has {history_tensor.shape[0]} samples but next_dataset has "
            f"{next_tensor.shape[0]} -- they must be paired 1:1 (each 16-step history with "
            "its true next-step label)."
        )
    if history_tensor.shape[1] != cfg.VRNN_SEQ_LEN:
        raise ValueError(
            f"history_dataset's sequence length is {history_tensor.shape[1]}, expected "
            f"config.VRNN_SEQ_LEN={cfg.VRNN_SEQ_LEN}."
        )

    # FIX: check for NaN/Inf in the raw input data BEFORE training starts.
    # If loss goes NaN on epoch 1 / batch 1, it's very likely this check
    # that trips, not the VRNN math -- e.g. a scaler fit on a zero-variance
    # feature (std=0) produces NaN/Inf on transform.
    for name, tensor in (("history_dataset", history_tensor), ("next_dataset", next_tensor)):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            n_nan = torch.isnan(tensor).sum().item()
            n_inf = torch.isinf(tensor).sum().item()
            raise ValueError(
                f"train_vrnn: {name} contains {n_nan} NaN and {n_inf} Inf values before any "
                "training has happened. Fix the upstream normalization first (e.g. add a "
                "small epsilon to any std/scale denominator, or drop/replace zero-variance "
                "columns) -- this is a data problem, not a VRNN problem."
            )

    # Concatenate history + next-step label into one (N, 17, D) sequence.
    full_seq = torch.cat([history_tensor, next_tensor.unsqueeze(1)], dim=1)  # (N, 17, D)

    train_dataset = TensorDataset(full_seq)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print(f"[VRNN] training on {full_seq.shape[0]} samples "
          f"(16-step history -> next-step prediction), saving to {save_path}")
    model = VRNN(input_dim=cfg.VRNN_INPUT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    columns = ["Epoch", "Loss", "NLL", "KLD"]
    vrnn_dataframe = DataFrame(columns=columns)

    for epoch in range(epochs):
        total_loss = 0.0
        nlList = []
        klList = []
        print(f"[VRNN] epoch {epoch+1}/{epochs} starting...")
        for (batch,) in train_loader:
            x = batch.to(device)  # (b, 17, D)
            opt.zero_grad()
            nll, kld, _, _ = model(x)
            loss = nll + kld

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"[VRNN] non-finite loss at epoch {epoch+1} "
                    f"(nll={nll.item()}, kld={kld.item()}). If this fires on epoch 1, check "
                    "your input data first (see the dataset check above); if it fires after "
                    "several epochs of otherwise-decreasing loss, try a lower learning rate."
                )

            nlList.append(nll.item())
            klList.append(kld.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            total_loss += loss.item()

        row = [epoch + 1, total_loss / len(train_dataset), sum(nlList) / len(nlList), sum(klList) / len(klList)]
        vrnn_dataframe.loc[len(vrnn_dataframe)] = row
        print(f"[VRNN] epoch {epoch+1}/{epochs} loss={total_loss:.4f}")

    vrnn_dataframe.to_csv(cfg.RESULT_DIR / f"vrnn_{cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]}_training_log.csv", index=False)
    torch.save(model.state_dict(), save_path)
    print(f"[VRNN] saved frozen checkpoint to {save_path}")
    return save_path