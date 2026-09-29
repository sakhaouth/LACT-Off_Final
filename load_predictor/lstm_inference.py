import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from load_predictor.LSTM import MultiValueLSTM
import config as cfg
# =====================================================
# Configuration
# =====================================================


def lstm_inference(sequence,model, scaler):
    """
    sequence16 : numpy array (16,num_features)
    """
    # print("here is the sequence")
    # print(sequence)
    sequence = np.array(sequence)
    
    feature_names = [
    "low_cpu", "low_mem", "low_num", "low_ttl",
    "mid_cpu", "mid_mem", "mid_num", "mid_ttl",
    "high_cpu", "high_mem", "high_num", "high_ttl"
    ]

    x_train = pd.DataFrame(sequence, columns=feature_names)
    sequence = scaler.transform(x_train)
    # scaled = scaler.transform(sequence)

    x = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(cfg.DEVICE)

    with torch.no_grad():
        pred = model(x)

    pred = pred.squeeze().cpu().numpy()

    pred = scaler.inverse_transform(
        pred.reshape(1,-1)
    )[0]

    return pred

def lstm_load_get_model(bs_id):
    # print("Starting inference")
    SEQ_LEN = 16
    MODEL_PATH = cfg.MODEL_DIR / f"[{cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]}]lstm_best_bs{bs_id}.pth"
    # CSV_PATH = "./local_load_raw-4.csv"

    

    # =====================================================
    # Load data and fit scaler (same as training)
    # =====================================================
    load_dir = cfg.LOAD_DIR / f"[{cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]}]raw_load_for_bs_{bs_id}.csv"
    df = pd.read_csv(load_dir)
    df = df.drop(columns=["date"], errors="ignore")

    # TRAIN_END = 1000

    

    scaler = StandardScaler()
    scaler.fit(df)

    # scaled = scaler.transform(df)

    num_features = df.shape[1]

    # =====================================================
    # Load model
    # =====================================================

    model = MultiValueLSTM(
        input_size=num_features,
        output_size=num_features
    ).to(cfg.DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=cfg.DEVICE))
    model.eval()

    # =====================================================
    # Predict next sample
    # =====================================================

    # choose any starting position
    # start = 75

    # sequence = train_data[start:start+SEQ_LEN]

    # x = torch.tensor(
    #     sequence,
    #     dtype=torch.float32
    # ).unsqueeze(0).to(DEVICE)

    # pred = lstm_inference(sequence, scaler, model, DEVICE)

    # true = df.iloc[start+SEQ_LEN].values

    # print("\nPredicted next load")
    # print(pred)

    # print("\nActual next load")
    # print(true)

    return model, scaler