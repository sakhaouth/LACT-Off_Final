

import config as cfg

from load_predictor.lstm_inference import lstm_load_get_model, lstm_inference
from load_predictor.llm_inference import load_get_model, llm_inference
from collections import deque


class Predictor:
    def __init__(self, bs_id):
        self.bs_id = bs_id
        self.model = None
        self.predictor_inference = None
        self.scaler = None
        self.history = deque(
            [[0.0] * cfg.LOCAL_LOAD_DIM  for _ in range(cfg.SEQ_LEN)],
            maxlen=cfg.SEQ_LEN
        )
        # print(cfg.CURRENT_RUNNIG_MODE)
        if cfg.CURRENT_RUNNIG_MODE == cfg.PDMA or cfg.CURRENT_RUNNIG_MODE == cfg.LACT_Off_MINUS:
            self.model, self.scaler = lstm_load_get_model(bs_id)
            self.predictor_inference = lstm_inference
        elif cfg.CURRENT_RUNNIG_MODE == cfg.LACT_Off or cfg.CURRENT_RUNNIG_MODE == cfg.PDMA_PLUS:
            self.model, self.scaler = load_get_model(bs_id)
            self.predictor_inference = llm_inference

    def predict(self, current_load):

        if self.model == None:
            return None
        next_load = self.predictor_inference(self.history, self.model, self.scaler)
        self.history.append(current_load)
        return next_load
    def reset(self):
        self.history = deque(
            [[0.0] * cfg.LOCAL_LOAD_DIM  for _ in range(cfg.SEQ_LEN)],
            maxlen=cfg.SEQ_LEN
        )
    