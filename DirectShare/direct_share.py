import os
import copy
import torch
import argparse
import numpy as np
from torch import nn
from operator import itemgetter
from transformers import AutoTokenizer, AutoModelForCausalLM
from models_for_share.original.modeling_llama import LlamaForCausalLM as LlamaForCausalLM_O
from models_for_share.modeling_llama import LlamaForCausalLM

def getHeadIndex(share_rate,similarity_values,match_indexs):
    dict_cor={}
    num_head=len(similarity_values[0])
    num_layer=len(similarity_values)+1 # layer 0 is not considered before
    for l_index, layer in enumerate(similarity_values):
        for h_index, head in enumerate(layer):
            dict_cor['Layer '+ str(num_layer-1-l_index) +' Head '+ str(h_index)]= head
    
    print('Sorting the similarity values...')        
    sorted_dict = dict(sorted(dict_cor.items(), key=itemgetter(1)))
    
    print('Choosing which heads will share the weight...')
    shared_head=int(share_rate*num_layer*num_head)
    replace_heads = list(sorted_dict.keys())[-shared_head:]
    print(replace_heads)

    replace_head_indexs=[[] for _ in range(num_layer)]
    cmp_head_indexs=[[] for _ in range(num_layer)]
    cmp_layer_indexs=[[] for _ in range(num_layer)]
    for head_name in replace_heads:
        layer_index=int(head_name.split('Layer ')[1].split(" Head ")[0])
        head_index=int(head_name.split(" Head ")[1])
        replace_head_indexs[layer_index].append(head_index)
    for index_i in range(num_layer):
        head_indexs = replace_head_indexs[index_i]
        for head_index in head_indexs:
            # print('Layer '+ str(index_i) +' Head '+ str(head_index))
            match_index = match_indexs[num_layer-index_i-1][head_index]
            # print(match_index) # match_head_index, match_layer_index
            cmp_head_indexs[index_i].append(match_index[0])
            cmp_layer_indexs[index_i].append(match_index[1]) 
    
    return replace_head_indexs, cmp_head_indexs, cmp_layer_indexs

def DirectShare(model, share_head_indexs, cmp_head_indexs, cmp_layer_indexs):
    
    head_num = model.model.layers[0].self_attn.num_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    hidden_size = model.model.layers[0].self_attn.hidden_size
    layer_num = model.model.layers[0].self_attn.config.num_hidden_layers
    
    # step1: init the shared part
    for i, block in enumerate(model.model.layers):
        head_indexs = share_head_indexs[i]
        if len(head_indexs)!=0:
            for j, head_index in enumerate(head_indexs):
                # add one head to share part
                model.model.share_head_indexs[i].append(head_index)
                
                cmp_layer_index = cmp_layer_indexs[i][j]
                cmp_head_index = cmp_head_indexs[i][j]
                cmp_attention=model.model.layers[cmp_layer_index].self_attn
                cmp_start_index = cmp_head_index*head_dim
                # add one head to share part
                model.model.share_head_indexs[cmp_layer_index].append(cmp_head_index)
                model.model.share_hash[head_index+i*head_num]=len(model.model.shared_heads_q_proj)
                model.model.share_hash[cmp_head_index+cmp_layer_index*head_num]=len(model.model.shared_heads_q_proj)
                
                # query
                q_tmp = nn.Linear(hidden_size,head_dim, bias=False)
                q_tmp.weight.data.copy_(cmp_attention.q_proj.weight.data[cmp_start_index:cmp_start_index+head_dim,:])#copy.deepcopy()
                model.model.shared_heads_q_proj.append(q_tmp)
                # key
                k_tmp = nn.Linear(hidden_size, head_dim, bias=False)
                k_tmp.weight.data.copy_(cmp_attention.k_proj.weight.data[cmp_start_index:cmp_start_index+head_dim,:])#copy.deepcopy()
                model.model.shared_heads_k_proj.append(k_tmp)
                # value
                v_tmp = nn.Linear(hidden_size, head_dim, bias=False)
                v_tmp.weight.data.copy_(cmp_attention.v_proj.weight.data[cmp_start_index:cmp_start_index+head_dim,:])#copy.deepcopy()
                model.model.shared_heads_v_proj.append(v_tmp)
                # output projection
                o_tmp = nn.Linear(head_dim, hidden_size, bias=False)
                o_tmp.weight.data.copy_(cmp_attention.o_proj.weight.data[:,cmp_start_index:cmp_start_index+head_dim])#copy.deepcopy()
                model.model.shared_heads_o_proj.append(o_tmp)

    for i in range(layer_num):
        model.model.share_head_indexs[i]=list(set(model.model.share_head_indexs[i])) # ignore the overlapping   
              
    # step 2: pass the shared part to each attention head
    for layer_index in range(layer_num):
        model.model.layers[layer_index].self_attn.q_proj_shared = model.model.shared_heads_q_proj
        model.model.layers[layer_index].self_attn.k_proj_shared = model.model.shared_heads_k_proj
        model.model.layers[layer_index].self_attn.v_proj_shared = model.model.shared_heads_v_proj
        model.model.layers[layer_index].self_attn.o_proj_shared = model.model.shared_heads_o_proj
        model.model.layers[layer_index].self_attn.share_hash = model.model.share_hash
        model.model.layers[layer_index].self_attn.layer_index = layer_index
        model.model.layers[layer_index].self_attn.share_head_indexs = sorted(model.model.share_head_indexs[layer_index]) # index from small to big
        model.model.layers[layer_index].self_attn.init_proj_unshared()
        # free the space for original weight matrix
        # model.model.layers[layer_index].self_attn.q_proj=None
        # model.model.layers[layer_index].self_attn.k_proj=None
        # model.model.layers[layer_index].self_attn.v_proj=None
        # model.model.layers[layer_index].self_attn.o_proj=None
        del model.model.layers[layer_index].self_attn.q_proj
        del model.model.layers[layer_index].self_attn.k_proj
        del model.model.layers[layer_index].self_attn.v_proj
        del model.model.layers[layer_index].self_attn.o_proj
    
    return model

def generate(text, tokenizer, model, length=64, penalty=1.1):
    inputs = tokenizer(text, return_tensors='pt', return_token_type_ids=False)
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs, max_new_tokens=length, repetition_penalty=penalty)
    print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="llama-2-7b-hf", help='name of model')
    parser.add_argument('--model_path', type=str, default="llama-2-7b-hf", help='path of model ckpt')
    parser.add_argument('--share_rate', type=str, default="0.3", help='the ratio for weight sharing')
    parser.add_argument('--match_index_path', type=str, default="../saved_npy/llama-2-7b-hf-qk/match_qk_indexs_llama-2-7b-hf_cos.npy", help='about candidate attention head pairs')
    parser.add_argument('--match_value_path', type=str, default="../saved_npy/llama-2-7b-hf-qk/match_qk_values_llama-2-7b-hf_cos.npy", help='about matching function values (cosine similarity)')
    parser.add_argument('--output_folder', type=str, default="shared_models/", help='path of saved model')
    args = parser.parse_args()
    
    if "llama" in args.model_name.lower():
        model_to_share = LlamaForCausalLM.from_pretrained(args.model_path, trust_remote_code = True) 
        model_to_share.to('cuda')
        model_to_share.to(torch.bfloat16)
    else:
        print('not support yet')

    tokenizer =  AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    match_indexs = np.array(np.load(args.match_index_path))
    similarity_values = np.array(np.load(args.match_value_path))
    share_head_indexs, cmp_head_indexs, cmp_layer_indexs = getHeadIndex(args.share_rate,similarity_values,match_indexs)
    
    model_shared = DirectShare(model_to_share, share_head_indexs, cmp_head_indexs, cmp_layer_indexs)
    model_shared.to('cuda')
    model_shared.to(torch.bfloat16)

    text = 'The future is going to be one that presents many challenges.'
    tokens = tokenizer(text, return_tensors='pt', return_token_type_ids=False, return_attention_mask=False).to(model_shared.device)
    out1 = model_shared(**tokens, output_attentions=True)

    
    if not os.path.exists(args.output_folder): 
        os.makedirs(args.output_folder)
    torch.save({'model': model_shared, 'tokenizer': tokenizer,}, args.output_folder+'/model.bin')
    
    # for loading 
    # shared_dict = torch.load(path, map_location='cpu')
    # tokenizer, model = shared_dict['tokenizer'], shared_dict['model']
    