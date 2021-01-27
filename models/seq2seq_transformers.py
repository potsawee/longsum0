import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers.modeling_bart import LearnedPositionalEmbedding, fill_with_neg_inf, invert_mask

class BartZero(nn.Module):
    def __init__(self, bart_config, torch_device):
        # Creating a model
        # Suppose to be the same model in Attention is All You Need
        # positional_encoder = PositionalEncoding(d_model=512)
        # model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
        #             dim_feedforward=2048, dropout=0.1,
        #             custom_encoder=None, custom_decoder=None)
        self.bart_config = BartConfig(
            activation_dropout=0.0, activation_function='gelu', vocab_size=50265,
            d_model=512, encoder_ffn_dim=2048, encoder_layers=6, encoder_attention_heads=8,
            decoder_ffn_dim=2048, decoder_layers=6, decoder_attention_heads=8,
            encoder_layerdrop=0.0, decoder_layerdrop=0.0, attention_dropout=0.0,
            dropout=0.1, max_position_embeddings=1024*2, init_std=0.02, classifier_dropout=0.0,
            num_labels=3, is_encoder_decoder=True, pad_token_id=1, bos_token_id=0, eos_token_id=2,
            normalize_before=False, add_final_layer_norm=False, scale_embedding=False, normalize_embedding=True,
            static_position_embeddings=False, add_bias_logits=False
        )

        self.bart = BartForConditionalGeneration(bart_config)

    def forward(self,input_ids,attention_mask,decoder_input_ids,decoder_attention_mask):
        x = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=target_ids,
            decoder_attention_mask=target_attention_mask,
        )
        return x

class BartAlpha(BartForConditionalGeneration):
    """
    BartForConditionalGeneration
    with a new LM head - note that bart-large-cnn shares the embedding three times:
        1) encoder
        2) decoder
        3) LM header
    but since fine-tuning this embedding would require backpropagation to first layer
    it's not memomry efficient, so we decided to add a new layer
    """
    # def __init__(self, model_name):
    #     super().__init__()
    #     self.bart = BartForConditionalGeneration.from_pretrained(model_name)
    #     self.decoder_lm_head = nn.Linear(self.bart.config.d_model, self.bart.config.vocab_size, bias=True)
    #     self.decoder_lm_head.weight.data = self.bart.model.shared.weight.data
    #     nn.init.zeros_(self.decoder_lm_head.bias)

    def __init__(self, config: BartConfig):
        super().__init__(config)

    def add_new_lm_head(self):
        self.decoder_lm_head = nn.Linear(self.model.config.d_model, self.model.config.vocab_size, bias=True)
        self.decoder_lm_head.weight.data = self.model.shared.weight.data
        nn.init.zeros_(self.decoder_lm_head.bias)

    # override
    def get_output_embeddings(self):
        if hasattr(self, 'decoder_lm_head'):
            return self.decoder_lm_head
        else:
            # self.add_new_lm_lead()
            # return self.decoder_lm_head
            print("decoder_lm_head has not been added")
            return None

    # override
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
        decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
        labels=None, use_cache=False, **unused):

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        lm_logits = self.decoder_lm_head(outputs[0])
        outputs = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
        return outputs

    def expand_learned_embed_positions(self, multiple=4, cut=0):
        if multiple != 2 and multiple != 4:
            raise ValueError("only multiple = 2,4 supported")

        new_embed_positions_size = 1026 * multiple - cut # original is 1024+2
        new_enc_embed_positions = LearnedPositionalEmbedding(new_embed_positions_size, self.model.config.hidden_size, self.model.config.pad_token_id)
        new_enc_embed_positions.weight.data[:1026] = self.model.encoder.embed_positions.weight.data
        new_enc_embed_positions.weight.data[1026:1026*2] = torch.flip(self.model.encoder.embed_positions.weight.data, dims=[0])
        if multiple == 4:
            new_enc_embed_positions.weight.data[1026*2:1026*3] = self.model.encoder.embed_positions.weight.data
            new_enc_embed_positions.weight.data[1026*3:1026*4-cut] = torch.flip(self.model.encoder.embed_positions.weight.data, dims=[0])[:-cut]
        self.model.encoder.embed_positions = new_enc_embed_positions

        new_dec_embed_positions = LearnedPositionalEmbedding(new_embed_positions_size, self.model.config.hidden_size, self.model.config.pad_token_id)
        new_dec_embed_positions.weight.data[:1026] = self.model.decoder.embed_positions.weight.data
        new_dec_embed_positions.weight.data[1026:1026*2] = torch.flip(self.model.decoder.embed_positions.weight.data, dims=[0])
        if multiple == 4:
            new_dec_embed_positions.weight.data[1026*2:1026*3] = self.model.decoder.embed_positions.weight.data
            new_dec_embed_positions.weight.data[1026*3:1026*4-cut] = torch.flip(self.model.decoder.embed_positions.weight.data, dims=[0])[:-cut]
        self.model.decoder.embed_positions = new_dec_embed_positions
        self.config.max_position_embeddings = new_embed_positions_size

        print("expanded learned_embed_positions to {} tokens".format(self.model.config.max_position_embeddings))

    def freeze_exclude_k_layers(self, k=1):
        for param in self.model.parameters(): param.requires_grad = False

        # for param in self.model.shared.parameters(): param.requires_grad = True
        # for param in self.model.encoder.embed_positions.parameters(): param.requires_grad = True
        # for param in self.model.encoder.layernorm_embedding.parameters(): param.requires_grad = True
        for _k in range(k):
            for param in self.model.encoder.layers[-(_k+1)].parameters(): param.requires_grad = True

        # for param in self.model.decoder.embed_positions.parameters(): param.requires_grad = True
        # for param in self.model.decoder.layernorm_embedding.parameters(): param.requires_grad = True
        for _k in range(k):
            for param in self.model.decoder.layers[-(_k+1)].parameters(): param.requires_grad = True

        print("freeze excluding top {} layer(s)".format(k))

class BartBeta(BartAlpha):
    def __init__(self, config: BartConfig):
        super().__init__(config)

    def pooling_layers(self):
        pass

    # override
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
        decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
        labels=None, use_cache=False, **unused):

        output_attentions    = False
        output_hidden_states = False

        # make masks if user doesn't supply
        decoder_input_ids, decoder_padding_mask, causal_mask = prepare_bart_decoder_inputs(
            self.config,
            input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_padding_mask=decoder_attention_mask,
            causal_mask_dtype=self.model.shared.weight.dtype,
        )

        if encoder_outputs is None:
            # encoder_outputs = self.model.encoder(
            encoder_outputs, reduced_attention_mask = self.forward_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        assert isinstance(encoder_outputs, tuple)

        if reduced_attention_mask is not None:
            attention_mask = (~reduced_attention_mask).long()

        decoder_outputs = self.model.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_cached_states=decoder_cached_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )

        # Attention and hidden_states will be [] or None if they aren't needed
        decoder_outputs: Tuple = _filter_out_falsey_values(decoder_outputs)
        assert isinstance(decoder_outputs[0], torch.Tensor)
        encoder_outputs: Tuple = _filter_out_falsey_values(encoder_outputs)
        # return decoder_outputs + encoder_outputs
        outputs = decoder_outputs + encoder_outputs

        # lm_logits = self.decoder_lm_head(outputs[0])
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)

        outputs = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
        return outputs

    def forward_encoder(self, input_ids, attention_mask=None,
                    output_attentions=False, output_hidden_states=False):

        # Why do we need to invert?? ANS: the EncoderLayer defined in huggingface is designed this way...
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self.model.encoder.embed_tokens(input_ids) * self.model.encoder.embed_scale
        embed_pos     = self.model.encoder.embed_positions(input_ids)

        x = inputs_embeds + embed_pos
        x = self.model.encoder.layernorm_embedding(x)
        x = F.dropout(x, p=self.model.encoder.dropout, training=self.model.encoder.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for layer_i, encoder_layer in enumerate(self.model.encoder.layers):
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.model.encoder.training and (dropout_probability < self.model.encoder.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask)

                if (layer_i+1) % 2 == 9999: # there are 12 layers
                    # ------------------------------- Pyramidal Style ------------------------------- #
                    stride = 2
                    len_0 = x.size(0)
                    len_1 = int(len_0/stride)

                    x_1 = torch.zeros((len_1, x.size(1), x.size(2)), dtype=x.dtype)
                    attention_mask_1 = torch.zeros((attention_mask.size(0),len_1), dtype=attention_mask.dtype)

                    for i in range(len_1):
                        x_1[i,:,:] = (x[i*2,:,:] + x[(i*2)+1,:,:]) / 2
                        attention_mask_1[:,i] = ~(~attention_mask[:,i*2] + ~attention_mask[:,(i*2)+1])

                    x = x_1.to(x.device)
                    attention_mask = attention_mask_1.to(attention_mask.device)
                    # -------------------------------------------------------------------------------- #
            if output_attentions:
                all_attentions.append(attn)

        if self.model.encoder.layer_norm:
            x = self.model.encoder.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        x = x.transpose(0, 1)

        return (x, encoder_states, all_attentions), attention_mask

def prepare_bart_decoder_inputs(
    config, input_ids, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32
):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
        dtype=causal_mask_dtype, device=decoder_input_ids.device
    )
    return decoder_input_ids, decoder_padding_mask, causal_mask

def _filter_out_falsey_values(tup):
    """Remove entries that are None or [] from an iterable."""
    return tuple(x for x in tup if isinstance(x, torch.Tensor) or x)

# def PyramidTransformer(nn.Module):
#     def __init__(self, bart_config, torch_device):
#         super(PyramidTransformer, self).__init__()
#         self.device = torch_device
