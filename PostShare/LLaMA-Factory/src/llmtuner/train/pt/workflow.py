# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py

import math
import torch
import numpy as np
from operator import itemgetter
from torch.nn import functional as F
from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForLanguageModeling, Trainer

from ...data import get_dataset, split_dataset
from ...extras.ploting import plot_loss
from ...model import load_model_and_tokenizer
from ...train.utils import create_modelcard_and_push


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments

def js_divergence(P, Q, reduction='sum'):
    P = F.softmax(P,dim=0)
    Q = F.softmax(Q,dim=0)
    M = 0.5 * (P + Q)
    return 0.5 * (F.kl_div(P.log(), M, reduction=reduction) + F.kl_div(Q.log(), M, reduction=reduction))

def kl_divergence(P, Q, reduction='sum'):
    # KL(P | Q) = sum(P * log(P / Q))
    # return torch.sum(P * (P.log() - Q.log()))
    P = F.softmax(P,dim=0)
    Q = F.softmax(Q,dim=0)
    kl_div = F.kl_div(P.log(), Q, reduction=reduction)
    return kl_div

def kl2_divergence(P, Q, reduction='sum'):
    # 0.5 * (KL(P|Q) + KL(Q|P))
    P = F.softmax(P,dim=0)
    Q = F.softmax(Q,dim=0)
    return 0.5*(F.kl_div(P.log(), Q, reduction=reduction)+F.kl_div(Q.log(), P, reduction=reduction))

class CustomTrainer(Trainer):
    def __init__(self, weight_loss, match_indexs, similarity_values, share_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Set which heads will share the weight
        self.weight_loss = weight_loss
        print("Setting which heads will share the weight...")
        match_indexs = np.array(np.load(match_indexs))
        similarity_values = np.array(np.load(similarity_values))
        
        dict_cor={}
        num_layer=len(similarity_values)+1 # layer 0 is not considered before
        num_head=len(similarity_values[0])
        shared_head=int(share_rate*num_layer*num_head)
        for l_index, layer in enumerate(similarity_values):
            for h_index, head in enumerate(layer):
                dict_cor['Layer '+str(num_layer-1-l_index) +' Head '+ str(h_index)]= head
        
        sorted_dict = dict(sorted(dict_cor.items(), key=itemgetter(1)))
        print('Sorting the similarity values...')
        # print(sorted_dict)
        replace_heads = list(sorted_dict.keys())[-shared_head:]
        print('Choosing which heads will share the weight...')
        print(replace_heads)
        
        self.replace_head_indexs=[[] for _ in range(num_layer)]
        self.cmp_head_indexs=[[] for _ in range(num_layer)]
        self.cmp_layer_indexs=[[] for _ in range(num_layer)]
        for head_name in replace_heads:
            layer_index=int(head_name.split('Layer ')[1].split(" Head ")[0])
            head_index=int(head_name.split(" Head ")[1])
            self.replace_head_indexs[layer_index].append(head_index)
        for index_i in range(num_layer):
            head_indexs = self.replace_head_indexs[index_i]
            for head_index in head_indexs:
                print('Layer '+ str(index_i) +' Head '+ str(head_index))
                match_index = match_indexs[num_layer-1-index_i][head_index]
                print(match_index) # match_head_index, match_layer_index
                self.cmp_head_indexs[index_i].append(match_index[0])
                self.cmp_layer_indexs[index_i].append(match_index[1])
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # Calling super 'compute_loss' method
        if return_outputs:
            super_loss, outputs = super().compute_loss(model, inputs, return_outputs=return_outputs)
        else:
            super_loss = super().compute_loss(model, inputs, return_outputs=return_outputs)
        
        JS_loss = 0
        KL_loss = 0
        norm_loss = 0
        mse_loss = 0
        total_head = 0
        weight1, weight2 = 0.5, 0.5
                
        for i, block in enumerate(model.model.layers):
            head_indexs = self.replace_head_indexs[i]
            if len(head_indexs)!=0:
                for j,head_index in enumerate(head_indexs):
                    total_head+=1
                    l_attention=block.self_attn
                    cmp_attention=model.model.layers[self.cmp_layer_indexs[i][j]].self_attn
                    split_size = l_attention.hidden_size 
                    head_dim = l_attention.head_dim
                    start_index = np.copy(head_index*head_dim)
                    cmp_start_index = np.copy(self.cmp_head_indexs[i][j]*head_dim)
                    
                    # print(start_index,cmp_start_index)
                    # print(start_index+head_dim)
                    # print(l_attention)
                    # print(l_attention.W_pack)
                    # print(l_attention.W_pack.weight)

                    # q_weight = l_attention.W_pack.weight[start_index:start_index+head_dim,:].flatten()
                    # q_weight_cmp = cmp_attention.W_pack.weight[cmp_start_index:cmp_start_index+head_dim,:].flatten()
                    # k_weight = l_attention.W_pack.weight[split_size+start_index:split_size+start_index+head_dim,:].flatten()
                    # k_weight_cmp = cmp_attention.W_pack.weight[split_size+cmp_start_index:split_size+cmp_start_index+head_dim,:].flatten()
                    # v_weight = l_attention.W_pack.weight[split_size*2+start_index:split_size*2+start_index+head_dim,:].flatten()
                    # v_weight_cmp = cmp_attention.W_pack.weight[split_size*2+cmp_start_index:split_size*2+cmp_start_index+head_dim,:].flatten()
                    
                    q_weight = l_attention.q_proj.weight[start_index:start_index+head_dim,:].flatten()
                    q_weight_cmp = cmp_attention.q_proj.weight[cmp_start_index:cmp_start_index+head_dim,:].flatten()
                    k_weight = l_attention.k_proj.weight[start_index:start_index+head_dim,:].flatten()
                    k_weight_cmp = cmp_attention.k_proj.weight[cmp_start_index:cmp_start_index+head_dim,:].flatten()
                    v_weight = l_attention.v_proj.weight[start_index:start_index+head_dim,:].flatten()
                    v_weight_cmp = cmp_attention.v_proj.weight[cmp_start_index:cmp_start_index+head_dim,:].flatten()
                    
                    # choice 1: norm 2
                    # choice 2: KL Divergence 
                    # choice 3: JS Divergence
                    # choice 4: MSE loss
                            
                    if self.weight_loss == "norm":
                        norm_loss+=torch.norm(q_weight-q_weight_cmp)
                        norm_loss+=torch.norm(k_weight-k_weight_cmp)
                        norm_loss+=torch.norm(v_weight-v_weight_cmp)
                    elif self.weight_loss == "MSE":
                        mse_loss += sum((q_weight - q_weight_cmp) ** 2) / len(q_weight)
                        mse_loss += sum((k_weight-k_weight_cmp) ** 2) / len(k_weight)
                    else:
                        # probability needs to be non negative
                        q_weight = q_weight - torch.min(q_weight) + torch.FloatTensor([1e-6]).to(self.args.device) # avoid zero
                        q_weight_cmp = q_weight_cmp - torch.min(q_weight_cmp) + torch.FloatTensor([1e-6]).to(self.args.device)
                        k_weight = k_weight - torch.min(k_weight) + torch.FloatTensor([1e-6]).to(self.args.device) # avoid zero
                        k_weight_cmp = k_weight_cmp - torch.min(k_weight_cmp) + torch.FloatTensor([1e-6]).to(self.args.device)
                        
                        if self.weight_loss == "JS":
                            JS_loss+=js_divergence(q_weight,q_weight_cmp, reduction='sum')
                            JS_loss+=js_divergence(k_weight,k_weight_cmp, reduction='sum')
                                    
                        elif self.weight_loss == "KL":
                            KL_loss+=kl_divergence(q_weight,q_weight_cmp, reduction='sum')
                            KL_loss+=kl_divergence(k_weight,k_weight_cmp, reduction='sum')
                                    
                        elif self.weight_loss == "KL2":
                            KL_loss+=kl2_divergence(q_weight,q_weight_cmp, reduction='sum')
                            KL_loss+=kl2_divergence(k_weight,k_weight_cmp, reduction='sum')
                                    
                        
        if self.weight_loss == "norm":   
            weight_loss = norm_loss / total_head
        elif self.weight_loss == "MSE":
            weight_loss = mse_loss / total_head
        elif self.weight_loss == "KL":
            weight_loss = KL_loss / total_head
        elif self.weight_loss == "JS":
            weight_loss = JS_loss / total_head
                    
        loss = weight1 * super_loss + weight2 * weight_loss
        
        from typing import Dict
        logs: Dict[str, float] = {}
        logs["model_loss"] = super_loss.detach().item()
        logs["weight_loss"] = weight_loss.detach().item()
        super().log(logs)
            
        return (loss, outputs) if return_outputs else loss
    
def run_pt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="pt")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize our Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     callbacks=callbacks,
    #     **split_dataset(dataset, data_args, training_args),
    # )
    trainer = CustomTrainer(
        weight_loss="norm",
        match_indexs='saved_npy/Llama-2-13b-hf-qk/match_qk_indexs_Llama-2-13b-hf_cos.npy',
        similarity_values='saved_npy/Llama-2-13b-hf-qk/match_qk_values_Llama-2-13b-hf_cos.npy',
        share_rate=0.3,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **split_dataset(dataset, data_args, training_args),
    )
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
