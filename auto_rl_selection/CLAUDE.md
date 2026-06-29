轨迹在本地这个压缩包里：/shared_workspace_mfs/fenglin/miles_dump_details/260401_231502_cd7fb81c.tar.zst 大小约 43G。里面是按 SWE task/run 分开的 .txt 轨迹 dump，

例如：
  miles_qwen3_32b_multi_r2e_gym_slime_deepswe_epoch0_260401_231502_cd7fb81c/...-swe-pillow-b3604167-260405143350-R104062.txt  
       
我抽样看过，内容包含 Status、token 统计、session_id、num_records、prompt、工具调用/响应等完整 trajectory 信息。
  
ckpt 在这里：

/shared_workspace_mfs/fenglin/miles_checkpoints/260401_231502_cd7fb81c/

大小约 428G，当前本地保留的是：

/shared_workspace_mfs/fenglin/miles_checkpoints/260401_231502_cd7fb81c/iter_0000179 

latest_checkpointed_iteration.txt 里是 179。格式是 Megatron/Torch distributed checkpoint，含 __*_*.distcp、common.pt，以及 rollout state：    

/shared_workspace_mfs/fenglin/miles_checkpoints/260401_231502_cd7fb81c/rollout/global_dataset_state_dict_*.pt  

另外，HF 导出也能访问：        

https://huggingface.co/PGCodeLLM/AgenticRL-blackbox-260401_231502_cd7fb81c      

远端有 global_step_9, 19, ..., 179。本地 /shared_workspace_mfs/fenglin/eval_exports/260401_231502_cd7fb81c/ 现在主要是 README 和上传/转换日志，不是完整权重。W&B API 这边我直接查 kirillv-org/miles-swe/
cd7fb81c 返回 project not found，看起来当前环境没有权限或项目不是公开的。