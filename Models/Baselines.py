import torch
from torch import nn
import torch.nn.functional as F

class Bert2Bert(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import EncoderDecoderModel
        from transformers import BertTokenizer
        self.seq2seq_model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert from pre-trained checkpoints      
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_model_inputs(self, inputs, targets, device):
        tokenizer_op = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
        encoder_input_ids = tokenizer_op['input_ids'].to(device)
        encoder_attention_mask = tokenizer_op['attention_mask'].to(device)

        tokenizer_op = self.tokenizer(targets, padding=True, truncation=True, return_tensors='pt')
        decoder_inputs = tokenizer_op['input_ids'].to(device)
        labels = decoder_inputs[:, 1:].clone()  
        labels[decoder_inputs[:, 1:] == self.tokenizer.pad_token_id] = -100 
        decoder_input_ids = decoder_inputs[:, :-1].clone()
        decoder_attention_mask = tokenizer_op['attention_mask'].to(device)
        decoder_attention_mask = decoder_attention_mask[:, :-1]
        model_inputs = {
            'input_ids': encoder_input_ids,
            'attention_mask': encoder_attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
            'decoder_attention_mask': decoder_attention_mask,
            'return_dict': True
        }
        return model_inputs

    def get_loss(self, outputs):
        loss, logits = outputs.loss, outputs.logits
        return loss

    def generate(self, inputs, device):
        tokenizer_op = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
        encoder_input_ids = tokenizer_op['input_ids'].to(device)
        encoder_attention_mask = tokenizer_op['attention_mask'].to(device)
        model_inputs = {
            'input_ids': encoder_input_ids,
            'attention_mask': encoder_attention_mask,
            'bos_token_id': self.tokenizer.cls_token_id,
            'eos_token_id': self.tokenizer.sep_token_id,
            'pad_token_id': self.tokenizer.sep_token_id,
            'num_beams': 2,
            'early_stopping': True,
            # 'no_repeat_ngram_size': 3,
            'repetition_penalty': 2.5,
            # 'length_penalty': 1.0,
            'max_length': 50

        }
        pred_ids = self.seq2seq_model.generate(**model_inputs)
        return self.tokenizer.batch_decode(pred_ids)

    def forward(self, **model_inputs):
        # print(model_inputs)
        return self.seq2seq_model(**model_inputs)

class T5(nn.Module):
    def __init__(self):
        super().__init__()
        PRETRAINED_MODEL_NAME = 't5-small'
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        self.t5_tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(PRETRAINED_MODEL_NAME, return_dict=True) 

    def get_model_inputs(self, inputs, targets, device):
        # Inputs
        tokenizer_op = self.t5_tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
        encoder_input_ids = tokenizer_op['input_ids'].to(device)
        encoder_attention_mask = tokenizer_op['attention_mask'].to(device)

        # Labels
        tokenizer_op = self.t5_tokenizer(targets, padding=True, truncation=True, return_tensors='pt')
        decoder_input_ids = tokenizer_op['input_ids'].to(device)

        # Model Inputs
        model_inputs = {
            'input_ids': encoder_input_ids,
            'attention_mask': encoder_attention_mask,
            'labels' : decoder_input_ids,
            'return_dict': True
        }

        return model_inputs

    def get_loss(self, outputs):
        return outputs.loss 

    def generate(self, inputs, device):
        tokenizer_op = self.t5_tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
        encoder_input_ids = tokenizer_op['input_ids'].to(device)
        encoder_attention_mask = tokenizer_op['attention_mask'].to(device)
        model_inputs = {
            'input_ids': encoder_input_ids,
            'attention_mask': encoder_attention_mask,
            'bos_token_id': self.t5_tokenizer.cls_token_id,
            'eos_token_id': self.t5_tokenizer.sep_token_id,
            'pad_token_id': self.t5_tokenizer.sep_token_id,
            'num_beams': 2,
            'early_stopping': True,
            # 'no_repeat_ngram_size': 3,
            'repetition_penalty': 2.5,
            # 'length_penalty': 1.0,
            'max_length': 50

        }
        pred_ids = self.t5_model.generate(**model_inputs)
        return self.t5_tokenizer.batch_decode(pred_ids)

    def forward(self, **model_inputs):
        return self.t5_model(**model_inputs)
