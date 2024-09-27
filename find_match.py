import os
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM


def getModelLayer(model):
    model_layer = None
    if hasattr(model, 'model'):
        model_component = model.model
        if hasattr(model_component, 'layers'): 
            model_layer = model_component.layers
    elif hasattr(model, 'transformer'):
        model_component = model.transformer
        if hasattr(model_component, 'h'): 
            model_layer = model_component.h
    else:
        print('please check the transformer structure name')
    return model_layer


def getAttention(model, index):
    model_layer = getModelLayer(model)[index]
    attention = None
    if hasattr(model_layer,"self_attn"):
        attention = model_layer.self_attn
    elif hasattr(model_layer, "attn"):
        attention = model_layer.attn
    elif hasattr(model_layer, "self_attention"):
        attention = model_layer.self_attention
    else:
        print('please check the attention structure name')
    return attention


def getAttentionWeight(model, layer_index):
    attention = getAttention(model, layer_index)
    
    q_weight, k_weight,v_weight = None, None, None
    if hasattr(attention,"W_pack"): # like baichuan
        q_weight = attention.W_pack.weight
        k_weight = attention.W_pack.weight
        v_weight = attention.W_pack.weight
    elif hasattr(attention,"c_attn"): # like gpt2-small
        q_weight = attention.c_attn.weight
        k_weight = attention.c_attn.weight
        v_weight = attention.c_attn.weight
    elif hasattr(attention, "k_proj"): # like llama
        q_weight = attention.q_proj.weight
        k_weight = attention.k_proj.weight
        v_weight = attention.v_proj.weight
    elif hasattr(model, "query_key_value"): # like Falcon, multi-query attention
        q_weight = attention.query_key_value.weight
        k_weight = attention.query_key_value.weight
        v_weight = attention.query_key_value.weight
    else:
        print('please check the attention weight name')
        
    return q_weight,k_weight,v_weight


def getAttentionHeadWeight(model, attn_q_weight, attn_k_weight, attn_v_weight, start_index):
    attention = getAttention(model, 0)
    head_dim = attention.head_dim
    split_size = attention.hidden_size 
    
    q_weight,k_weight,v_weight = None, None, None
    
    if hasattr(attention,"W_pack"): # like baichuan
        q_weight = attn_q_weight[start_index:start_index+head_dim,:]
        k_weight = attn_k_weight[split_size+start_index:split_size+start_index+head_dim,:]
        v_weight = attn_v_weight[split_size*2+start_index:split_size*2+start_index+head_dim,:]
    elif hasattr(attention,"c_attn"): # like gpt2-small
        q_weight = attn_q_weight[:,start_index:start_index+head_dim]
        k_weight = attn_k_weight[:,split_size+start_index:split_size+start_index+head_dim]
        v_weight = attn_v_weight[:,split_size*2+start_index:split_size*2+start_index+head_dim]
    elif hasattr(attention, "k_proj"): # like llama
        q_weight = attn_q_weight[start_index:start_index+head_dim,:]
        k_weight = attn_k_weight[start_index:start_index+head_dim,:]
        v_weight = attn_v_weight[start_index:start_index+head_dim,:]
    elif hasattr(model, "query_key_value"): # like Falcon, multi-query attention
        q_weight = attn_q_weight[start_index:start_index+head_dim,:]
        k_weight = attn_k_weight[split_size+start_index:split_size+start_index+head_dim,:]
        v_weight = attn_v_weight[split_size+start_index+head_dim:split_size+start_index+head_dim*2,:]
    else:
        print('please check the attention struction (self-attention, multi-query-attention, group-query-attention...')
        
    return q_weight,k_weight,v_weight
 
    
def findMatch(model, strategy):
    model_layer = getModelLayer(model)
    layer_num = len(model_layer)
    
    attention = getAttention(model, 0)
    head_num = attention.num_heads
    head_dim = attention.head_dim
    
    all_cofs_dict={}
    all_cofs=[]
    match_indexs=[]
    match_values=[]
    for layer_index in range(layer_num-1,0,-1):
        match_index=[]
        match_value=[]
        attn_q_weight, attn_k_weight, attn_v_weight = getAttentionWeight(model, layer_index)
        for head_index in range(head_num):
            start_index = head_index*head_dim
            q_to_cmp, k_to_cmp, v_to_cmp = getAttentionHeadWeight(model, attn_q_weight, attn_k_weight, attn_v_weight, start_index)
            
            if strategy == "q":
                attn_to_cmp = q_to_cmp
            elif strategy == "k":
                attn_to_cmp = k_to_cmp
            elif strategy == "v":
                attn_to_cmp = v_to_cmp
            elif strategy == "qk":    
                attn_to_cmp = torch.cat((q_to_cmp, k_to_cmp), dim=0)
            elif strategy == "qkv":
                attn_to_cmp = torch.cat((q_to_cmp, k_to_cmp, v_to_cmp), dim=0)
            elif strategy == "total":
                # softmax(Q*K/sqrt(d^k))*v
                attn_to_cmp = torch.matmul(q_to_cmp, k_to_cmp.transpose(-1, -2))
                attn_to_cmp = attn_to_cmp/ torch.full(
                        [], v_to_cmp.size(-1) ** 0.5, dtype=attn_to_cmp.dtype, device=attn_to_cmp.device
                    ) 
                attn_to_cmp = F.softmax(attn_to_cmp, dim=-1)
                attn_to_cmp = torch.matmul(attn_to_cmp,v_to_cmp)

            cofs=[]
            max_value = float('-inf')  
            max_layer_index = -1
            max_head_index = -1
            for layer_cmp in range(layer_index):
                cofs.append([])
                for head_cmp in range(head_num):
                    start_cmp_index = head_cmp*head_dim
                    attn_q_weight_ref, attn_k_weight_ref, attn_v_weight_ref = getAttentionWeight(model, layer_cmp)
                    q_ref, k_ref, v_ref = getAttentionHeadWeight(model, attn_q_weight_ref, attn_k_weight_ref, attn_v_weight_ref, start_cmp_index)
                    
                    if strategy == "q":
                        attn_ref = q_ref
                    elif strategy == "k":
                        attn_ref = k_ref
                    elif strategy == "v":
                        attn_ref = v_ref
                    elif strategy == "qk":    
                        attn_ref = torch.cat((q_ref, k_ref), dim=0)
                    elif strategy == "qkv":
                        attn_ref = torch.cat((q_ref, k_ref, v_ref), dim=0)
                    elif strategy == "total":
                        attn_ref = torch.matmul(q_ref, k_ref.transpose(-1, -2))
                        attn_ref = attn_ref / torch.full(
                                [], v_ref.size(-1) ** 0.5, dtype=attn_ref.dtype, device=attn_ref.device
                            ) 
                        attn_ref = F.softmax(attn_ref, dim=-1)
                        attn_ref = torch.matmul(attn_ref, v_ref)  
                    
                    A = attn_to_cmp.flatten().unsqueeze(0)
                    B = attn_ref.flatten().unsqueeze(0)
                    cof = F.cosine_similarity(A, B)[0].item()
                    cofs[-1].append(cof)
                    if cof > max_value:
                        max_value = cof
                        max_layer_index = layer_cmp
                        max_head_index = head_cmp 
                    
            index = 'Layer '+str(layer_index)+', Head '+str(head_index)
            all_cofs_dict[index]=cofs
            all_cofs.append(cofs)
            print(cofs)
            print('max match index: Layer ',max_layer_index,', Head ',max_head_index)
            print('max correlation coefficient value: ',max_value)
            match_value.append(max_value)
            match_index.append([max_head_index,max_layer_index])
            
        match_indexs.append(match_index)
        match_values.append(match_value)    
    
    return match_indexs, match_values


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="llama-2-7b-hf", help='name of model')
    parser.add_argument('--model_path', type=str, default="llama-2-7b-hf", help='path of model ckpt')
    parser.add_argument('--strategy', type=str, default="qk", choices=["qk", "q", "k", "v", "qkv","total"], help='the match function for shared weights')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    match_indexs, match_values = findMatch(model, args.strategy)
    save_path = 'saved_npy/'+args.model_name+"-"+args.strategy+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path+'match_'+args.strategy+'_indexs_'+args.model_name+'_cos.npy', match_indexs)
    np.save(save_path+'match_'+args.strategy+'_values_'+args.model_name+'_cos.npy', match_values)
    