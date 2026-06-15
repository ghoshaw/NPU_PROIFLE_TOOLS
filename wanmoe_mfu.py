### 模型MFU/HFU计算脚本

# 用户提供训练配置
SP=32
CP=32
EP = 8
# 用户提供序列长度
total_video_seq_len= 540672  # 129480 # #32760
text_seq_len=512
# 用户提供迭代耗时，单位是us
latency_us = 9901000 #22363492.06 #23738100 #34751924.75 #
fa_latency_us = 5660350.565
fag_latency_us = 7602064.756
# 用户提供标称算力
peak_tflops_power= 432  #   A3:353 A5:432
# 模型参数
num_heads=32
hidden_size=4096
head_dim=128

# MoE
expert_num = 384
topk = 20
ffn_intern_size= 1536

num_layers=5



def calc_gemm_flops():
    video_seq_len_per_device=total_video_seq_len/SP
    video_attn_seq_len_per_device=total_video_seq_len
    forward_gemm_flops=2*video_seq_len_per_device * hidden_size*hidden_size*(4+2)*num_layers+2*text_seq_len * hidden_size*hidden_size*2*num_layers+2*hidden_size*video_seq_len_per_device*expert_num*num_layers+2*video_seq_len_per_device*hidden_size*ffn_intern_size*3*topk*num_layers
    backward_gemm_flops=forward_gemm_flops*2
    # print('forward_fa_flops : ',forward_fa_flops*2/ fa_latency_us/1e6/peak_tflops_power)
    # print('backward_fa_flops : ',backward_fa_flops/ fag_latency_us/1e6/peak_tflops_power)
    total_gemm_flops_w_mfu = forward_gemm_flops + backward_gemm_flops
    total_gemm_flops_w_hfu = forward_gemm_flops*2 + backward_gemm_flops
    return total_gemm_flops_w_mfu, total_gemm_flops_w_hfu

def calc_fa_flops():
    video_attn_seq_len_per_device=total_video_seq_len
    attn_heads_per_device=num_heads/CP
    forward_fa_flops=2*video_attn_seq_len_per_device * (video_attn_seq_len_per_device+text_seq_len)*attn_heads_per_device*head_dim*2*num_layers
    backward_fa_flops=forward_fa_flops*(5/2)
    print('forward_fa_flops : ',forward_fa_flops*2/ fa_latency_us/1e6/peak_tflops_power)
    print('backward_fa_flops : ',backward_fa_flops/ fag_latency_us/1e6/peak_tflops_power)
    total_fa_flops_w_mfu = forward_fa_flops + backward_fa_flops
    total_fa_flops_w_hfu = forward_fa_flops*2 + backward_fa_flops
    return total_fa_flops_w_mfu, total_fa_flops_w_hfu

total_gemm_flops_w_mfu, total_gemm_flops_w_hfu = calc_gemm_flops()
total_fa_flops_w_mfu, total_fa_flops_w_hfu = calc_fa_flops()
model_flops_w_mfu=total_gemm_flops_w_mfu + total_fa_flops_w_mfu
model_flops_w_hfu=total_gemm_flops_w_hfu + total_fa_flops_w_hfu
# 计算mfu
mfu=model_flops_w_mfu/ latency_us/1e6/peak_tflops_power
# 计算hfu，开启重计算训练的模型用例
hfu=model_flops_w_hfu/ latency_us/1e6/peak_tflops_power

# 0.4795653388122369 0.6202864674764578
print(model_flops_w_mfu/1e16, mfu, hfu)