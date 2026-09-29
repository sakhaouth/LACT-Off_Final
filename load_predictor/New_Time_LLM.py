from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizerFast, GPT2Config, GPT2Model, GPT2TokenizerFast, BertConfig, \
    BertModel, BertTokenizerFast, DistilBertConfig, DistilBertModel, DistilBertTokenizerFast
from load_predictor.layers.Embed import PatchEmbedding
import transformers
from load_predictor.layers.StandardNorm import Normalize


transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        # NOTE: self.d_llm is now set further below, auto-detected from whichever
        # backbone actually loads, instead of trusted blindly from configs.llm_dim.
        # A mismatch there (e.g. switching to a smaller-hidden-size model without
        # updating --llm_dim) used to cause a hard-to-read shape error deep in
        # the reprogramming layer.
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # ------------------------------------------------------------------
        # MEMORY FIX: GPT2 base only has 12 pretrained transformer blocks.
        # Requesting more (e.g. 32, a LLaMA-7B-sized setting) forces
        # transformers to pad in randomly-initialized extra blocks -- that
        # roughly doubles/triples memory and compute for layers that aren't
        # even pretrained. We clip it here and warn instead of silently
        # ballooning.
        # ------------------------------------------------------------------
        max_layers_by_model = {
            'GPT2': 12,
            'DISTILGPT2': 6,   # distilled GPT2: half the blocks, ~2x faster forward/backward
            'BERT': 12,
            'DISTILBERT': 6,   # distilled BERT: half the blocks, ~1.6x faster
            'TINYBERT': 2,     # prajjwal1/bert-tiny: 2 layers, 128 hidden, ~4.4M params -- by far the fastest option
            'LLAMA': 32,
        }
        requested_layers = configs.llm_layers
        max_layers = max_layers_by_model.get(configs.llm_model)
        if max_layers is not None and requested_layers > max_layers:
            print(
                f"[warning] --llm_layers={requested_layers} requested for {configs.llm_model}, "
                f"but it only has {max_layers} pretrained blocks. Clipping to {max_layers}."
            )
            requested_layers = max_layers

        # Optional: enable gradient checkpointing to trade compute for memory.
        # Big win when you need more layers/batch and can afford slower steps.
        self.use_gradient_checkpointing = getattr(configs, "gradient_checkpointing", True)

        print(f"Model Name is {configs.llm_model} ***************************888")
        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = requested_layers
            # MEMORY FIX: we only ever use last_hidden_state downstream, so
            # don't ask the model to materialize (and backprop through) every
            # layer's attention matrix and hidden state.
            self.llama_config.output_attentions = False
            self.llama_config.output_hidden_states = False
            self.llama_config.use_cache = False  # no autoregressive generation happening here
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )
            try:
                self.tokenizer = LlamaTokenizerFast.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizerFast.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'DISTILGPT2':
            # Distilled GPT2: same architecture/classes, half the blocks (6 vs 12),
            # ~82M params vs 124M -- meaningfully faster with a small accuracy tradeoff.
            self.gpt2_config = GPT2Config.from_pretrained('distilbert/distilgpt2')

            self.gpt2_config.num_hidden_layers = requested_layers
            self.gpt2_config.output_attentions = False
            self.gpt2_config.output_hidden_states = False
            self.gpt2_config.use_cache = False
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'distilbert/distilgpt2',
                    trust_remote_code=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'distilbert/distilgpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2TokenizerFast.from_pretrained(
                    'distilbert/distilgpt2',
                    trust_remote_code=True,
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2TokenizerFast.from_pretrained(
                    'distilbert/distilgpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = requested_layers
            # MEMORY FIX: see note above.
            self.gpt2_config.output_attentions = False
            self.gpt2_config.output_hidden_states = False
            self.gpt2_config.use_cache = False
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2TokenizerFast.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2TokenizerFast.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = requested_layers
            self.bert_config.output_attentions = False
            self.bert_config.output_hidden_states = False
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizerFast.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizerFast.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'DISTILBERT':
            # Distilled BERT: same hidden size (768) as bert-base, 6 layers vs 12,
            # ~66M params vs 110M -- roughly 1.6x faster.
            self.bert_config = DistilBertConfig.from_pretrained('distilbert/distilbert-base-uncased')

            self.bert_config.n_layers = requested_layers  # DistilBert names this n_layers, not num_hidden_layers
            self.bert_config.output_attentions = False
            self.bert_config.output_hidden_states = False
            try:
                self.llm_model = DistilBertModel.from_pretrained(
                    'distilbert/distilbert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = DistilBertModel.from_pretrained(
                    'distilbert/distilbert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = DistilBertTokenizerFast.from_pretrained(
                    'distilbert/distilbert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = DistilBertTokenizerFast.from_pretrained(
                    'distilbert/distilbert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'TINYBERT':
            print("Working wth TINYBERT")
            # prajjwal1/bert-tiny: 2 layers, 128 hidden size, ~4.4M params.
            # An order of magnitude smaller than DistilBERT/DistilGPT2 -- use
            # this if DistilGPT2 is still too slow for your setup.
            self.bert_config = BertConfig.from_pretrained('prajjwal1/bert-tiny')

            self.bert_config.num_hidden_layers = requested_layers
            self.bert_config.output_attentions = False
            self.bert_config.output_hidden_states = False
            try:
                self.llm_model = BertModel.from_pretrained(
                    'prajjwal1/bert-tiny',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'prajjwal1/bert-tiny',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizerFast.from_pretrained(
                    'prajjwal1/bert-tiny',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizerFast.from_pretrained(
                    'prajjwal1/bert-tiny',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        print("FREEZING PARAMS")
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # MEMORY FIX: gradient checkpointing trades ~20-30% extra compute time
        # for a large activation-memory reduction, which is what you want when
        # you're VRAM constrained rather than compute constrained.
        if self.use_gradient_checkpointing and hasattr(self.llm_model, "gradient_checkpointing_enable"):
            self.llm_model.gradient_checkpointing_enable()

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]

        # AUTO-DETECT FIX: derive d_llm from the model that actually loaded
        # rather than trusting --llm_dim, so swapping --llm_model can't silently
        # desync from the hidden size and blow up later in ReprogrammingLayer.
        detected_d_llm = self.word_embeddings.shape[-1]
        if detected_d_llm != configs.llm_dim:
            print(
                f"[info] --llm_dim was {configs.llm_dim} but {configs.llm_model} actually has "
                f"hidden size {detected_d_llm}; using {detected_d_llm}."
            )
        self.d_llm = detected_d_llm
        self.num_tokens = 250
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        # self.patch_nums = 2
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        # PERF FIX: the old version called .tolist() once per sample inside the
        # loop below, each one forcing a small GPU->CPU sync. Pull everything to
        # CPU/numpy in one bulk transfer up front instead -- same result, far
        # fewer round trips, especially noticeable once batch*enc_in gets into
        # the dozens/hundreds.
        min_values_np = min_values.squeeze(-1).detach().cpu().numpy()
        max_values_np = max_values.squeeze(-1).detach().cpu().numpy()
        medians_np = medians.squeeze(-1).detach().cpu().numpy()
        trends_np = trends.squeeze(-1).detach().cpu().numpy()
        lags_np = lags.detach().cpu().numpy()

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values_np[b])
            max_values_str = str(max_values_np[b])
            median_values_str = str(medians_np[b])
            lags_values_str = str(lags_np[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends_np[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding