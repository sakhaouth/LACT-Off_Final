"""
Variational RNN used by the UPF to compress the sequence of per-server load
vectors (one per server, per slot) into a single global load embedding that
is broadcast back to every server.

CONFUSION: "compressed global load" via VRNN could be modeled as:
  (a) a sequence over TIME (one server's load history) -> temporal VRNN, or
  (b) a sequence over SERVERS at a fixed time step (set-like aggregation), or
  (c) both (spatio-temporal).
I implemented (b) folded into (a): at every slot, the UPF feeds the ordered
list of the NUM_SERVERS load vectors through the VRNN as a sequence of length
NUM_SERVERS, and the final hidden state's latent z is treated as the global
summary for that slot. This means the RNN dimension is "which server", not
"which time step", so the model implicitly learns spatial correlations across
servers within a slot. If you actually intended a genuine temporal model
(load evolving slot-to-slot per server), the loop in `encode_global_load`
needs to run over TIME instead of over SERVERS, and you would want a
per-server hidden state carried across slots. This is a significant modeling
fork -- please confirm which one you meant.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

from madrl_edge import config

from pandas import DataFrame

from collections import deque



class VRNN(nn.Module):
    def __init__(self, input_dim=config.VRNN_INPUT_DIM,
                 hidden_dim=config.VRNN_HIDDEN_DIM,
                 latent_dim=config.VRNN_LATENT_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # feature extractors
        self.phi_x = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.phi_z = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())

        # prior p(z_t | h_{t-1})
        self.prior = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)

        # encoder q(z_t | x_t, h_{t-1})
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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x_seq):
        """
        x_seq: (batch, seq_len, input_dim) -- seq_len is NUM_SERVERS (see
        module docstring for the (a)/(b)/(c) ambiguity).
        Returns: reconstruction loss components + final latent summary z_final.
        """
        batch, seq_len, _ = x_seq.shape
        h = torch.zeros(batch, self.hidden_dim, device=x_seq.device)

        kld_total = 0.0
        nll_total = 0.0
        z_final = None

        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            phi_x_t = self.phi_x(x_t)

            prior_h = self.prior(h)
            prior_mean_t = self.prior_mean(prior_h)
            prior_logvar_t = self.prior_logvar(prior_h)

            enc_h = self.enc(torch.cat([phi_x_t, h], dim=-1))
            enc_mean_t = self.enc_mean(enc_h)
            enc_logvar_t = self.enc_logvar(enc_h)

            z_t = self._reparam(enc_mean_t, enc_logvar_t)
            phi_z_t = self.phi_z(z_t)

            dec_h = self.dec(torch.cat([phi_z_t, h], dim=-1))
            x_hat_t = self.dec_mean(dec_h)

            # losses
            nll_total = nll_total + F.mse_loss(x_hat_t, x_t, reduction="mean")
            kld_total = kld_total + self._kld(enc_mean_t, enc_logvar_t,
                                               prior_mean_t, prior_logvar_t)

            h = self.rnn_cell(torch.cat([phi_x_t, phi_z_t], dim=-1), h)
            z_final = z_t

        return nll_total / seq_len, kld_total / seq_len, z_final

    @staticmethod
    def _kld(mean1, logvar1, mean2, logvar2):
        # KL( N(mean1,var1) || N(mean2,var2) )
        return 0.5 * torch.mean(
            logvar2 - logvar1
            + (torch.exp(logvar1) + (mean1 - mean2) ** 2) / torch.exp(logvar2)
            - 1
        )


class FrozenVRNNEncoder:
    """
    Thin inference-only wrapper used during MADDPG training/execution once
    the VRNN has been pretrained. Produces the GLOBAL_LOAD_DIM summary vector
    given the current slot's per-server load vectors.
    """

    def __init__(self, checkpoint_path: str, device="cpu"):
        self.device = device
        self.model = VRNN().to(device)
        state = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.history = deque(
            [[0.0] * config.VRNN_INPUT_DIM for _ in range(config.VRNN_SEQ_LEN)],
            maxlen=config.VRNN_SEQ_LEN
        )

    @torch.no_grad()
    def compress(self, all_server_loads):
        """
        all_server_loads: list of NUM_SERVERS lists, each config.LOCAL_LOAD_DIM
        long (i.e. the raw per-server load vectors for the current slot).
        Returns: a single GLOBAL_LOAD_DIM python list, broadcast to every server.
        """
        loads = []
        for load in all_server_loads:
            if len(load) != config.LOCAL_LOAD_DIM:
                raise ValueError(f"Expected load vector of length {config.LOCAL_LOAD_DIM}, got {len(load)}")
            loads.extend(load)
        self.history.append(loads)
        x = torch.tensor([list(self.history)], dtype=torch.float32, device=self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # add batch dim
        
        # x = torch.tensor([all_server_loads], dtype=torch.float32, device=self.device)
        _, _, z_final = self.model(x)
        return z_final.squeeze(0).cpu().tolist()
    
    def reset(self):
        self.history.clear()
        self.history.extend([[0.0] * config.VRNN_INPUT_DIM for _ in range(config.VRNN_SEQ_LEN)])


def train_vrnn(dataset, epochs=50, lr=1e-3, device="cpu", save_path="vrnn.pt"):
    """
    dataset: iterable of (NUM_SERVERS, VRNN_INPUT_DIM) numpy/torch arrays,
    one entry per historical slot, collected from simulated or logged traffic
    BEFORE MADDPG training starts. This is the "train the VRNN first, freeze
    it during execution" step you described.

    CONFUSION: dataset generation itself (how realistic load traces are
    simulated) is not specified -- plug in your task-arrival generator here.
    """


    dataset_tensor = torch.tensor(dataset, dtype=torch.float32, device=device)
    train_dataset = TensorDataset(dataset_tensor)
    train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
    )
    print(f"[VRNN] training on {len(dataset)} slots of load data, saving to {save_path}")
    model = VRNN(input_dim=config.VRNN_INPUT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    columns = ["Epoch", "Loss", "NLL", "KLD"]
    vrnn_dataframe = DataFrame(columns=columns)
    for epoch in range(epochs):
        total_loss = 0.0
        nlList = []
        klList = []
        print(f"[VRNN] epoch {epoch+1}/{epochs} starting...")
        for (batch,) in train_loader:
            x = batch.to(device)
            if x.dim() == 2:
                x = x.unsqueeze(0)  # add batch dim
            opt.zero_grad()
            nll, kld, _ = model(x)
            nlList.append(nll.item())
            klList.append(kld.item())
            loss = nll + kld
            loss.backward()
            opt.step()
            total_loss += loss.item()
        row = [epoch + 1, total_loss / len(dataset), sum(nlList) / len(nlList), sum(klList) / len(klList)]
        vrnn_dataframe.loc[len(vrnn_dataframe)] = row
        print(f"[VRNN] epoch {epoch+1}/{epochs} loss={total_loss:.4f}")
    vrnn_dataframe.to_csv("vrnn_training_log.csv", index=False)
    torch.save(model.state_dict(), save_path)
    print(f"[VRNN] saved frozen checkpoint to {save_path}")
    return save_path
