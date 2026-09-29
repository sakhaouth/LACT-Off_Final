import torch
import pandas as pd
import numpy as np
from argparse import Namespace
from sklearn.preprocessing import StandardScaler

import pandas as pd

from load_predictor.New_Time_LLM import Model

##############################################
# Configuration (must match training)
##############################################

import config as cfg
print("Here I m starting")

def load_get_model(bs_id):
    args = Namespace(
        task_name= cfg.TASK_NAME,

        pred_len= cfg.PRED_LEN,
        seq_len= cfg.SEQ_LEN,
        label_len= cfg.LABEL_LEN,

        enc_in= cfg.ENC_IN,
        dec_in= cfg.DEC_IN,
        c_out= cfg.C_OUT,

        d_model= cfg.D_MODEL,
        d_ff= cfg.D_FF,
        n_heads= cfg.N_HEADS,

        patch_len= cfg.PATCh_LEN,
        stride= cfg.STRIDE,

        dropout= cfg.DROP_OUT,

        llm_model= cfg.LLM_MODEL,
        llm_layers= cfg.LLM_LAYERS,
        llm_dim= cfg.LLM_DIM,

        factor=cfg.FACTOR,

        prompt_domain= cfg.PROMT_DOMAIN,
        content=cfg.CONTENT,

        gradient_checkpointing=cfg.GRADIENT_CHECKPOINTING,
    )

    ##############################################
    # Device
    ##############################################

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################################
    # Build model
    ##############################################




    model = Model(args).float().to(cfg.DEVICE)
    model_dir = cfg.MODEL_DIR / f"[{cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]}]llm_best_bs{bs_id}.pth"
    model.load_state_dict(
        torch.load(
            model_dir,
            map_location=cfg.DEVICE
        )
    )

    model.eval()

##############################################
# Load data
##############################################
    load_dir = cfg.LOAD_DIR / f"[{cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]}]raw_load_for_bs_{bs_id}.csv"
    df = pd.read_csv(load_dir)

    df = df.drop(columns=["date"], errors="ignore")

    scaler = StandardScaler()
    scaler.fit(df)

    # scaled = scaler.transform(df)

    ##############################################
    # Last 16 samples
    ##############################################
    # SEQ_LEN = 16
    # start = 75
    # sequence = scaled[start:start+SEQ_LEN]

    # x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

    return model, scaler

##############################################
# Predict
##############################################

def llm_inference(sequence ,model, scaler):

    sequence = np.array(sequence)
    
    feature_names = [
    "low_cpu", "low_mem", "low_num", "low_ttl",
    "mid_cpu", "mid_mem", "mid_num", "mid_ttl",
    "high_cpu", "high_mem", "high_num", "high_ttl"
    ]

    x_train = pd.DataFrame(sequence, columns=feature_names)
    sequence = scaler.transform(x_train)

    
    x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(cfg.DEVICE)
    

    with torch.no_grad():
        pred = model(x)

    pred = pred.squeeze(0).cpu().numpy()

    # prediction = scaler.inverse_transform(pred)
    pred = scaler.inverse_transform(
        pred.reshape(1,-1)
    )[0]

    # print("Predicted next load:")

    # print(prediction)

    # true = df.iloc[start+SEQ_LEN].values

    # print("True load:")

    # print(true)
    return pred