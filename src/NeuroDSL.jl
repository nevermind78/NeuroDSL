module NeuroDSL
using Printf, Statistics, Random, LinearAlgebra, Dates,JSON

include("backend.jl")
using .Backend


include("types.jl")
include("graph_api.jl")
include("kernels.jl")
include("dispatch.jl")
include("backward.jl")
include("graph_data.jl")
include("liveness.jl")
include("checkpoint.jl")
include("compiler_config.jl")
include("compiler_rules.jl")
include("compiler.jl")
include("flash_attention.jl")
include("mixed_precision.jl")
include("macros.jl")
include("layers.jl")
include("viz.jl")
include("dsl_macros.jl")
include("backward_sparse.jl")


export NeuroGraph, GraphNode, GraphRule, CtxStore
export NeurAtom, Datom, Quantom, is_backpropable
export set!, node, addrule!, demand!, params
export activate!, namespaces, graph_summary
export topo_order!, zero_grads!, invalidate_all!
export backward_graph!, accum_grad!, GRAD_RULES, register_op!
export rmsnorm_fwd!, rmsnorm_bwd!, swiglu_fwd!, swiglu_bwd!
export softmax_fwd!, softmax_bwd!, cross_entropy_loss, cross_entropy_grad
export mse_loss_fwd, mse_loss_bwd, adamw_step!
export GraphData, CPUTrainData, CUDATrainData, CheckpointData, MixedPrecData
export auto_graphdata, graphdata_from_backend, get_device, fwd_precision, bwd_precision, supports_checkpointing, supports_mixed_precision, checkpoint_every
export LivenessInterval, BufferPool, MemoryPlan, plan_memory!, demand_planned!, compute_liveness, pool_stats
export CheckpointSchedule, forward_with_checkpointing!, backward_with_checkpointing!
export flash_attn_fwd!, flash_attn_bwd!, MultiHeadFlashAttention, flash_attn_fwd_cpu!, flash_attn_fwd_cpu_simple!, flash_attn_bwd_cpu!
export cast_fp16, cast_fp32, LossScaleTracker, update!, backward_with_loss_scaling!, mixed_precision_step!
export LayerNorm, Linear, MultiHeadAttention, LlamaBlock, LlamaModel
export @addrules, Backend, debug!
export _watch!, _fuse!, _invalidate_upstream!, _invalidate_downstream! 
export ExecutionLog, log_event!
export save_interactive_graph, graph_to_json
export TrainingSnapshot, TrainingRecorder, should_capture
export save_interactive_graph_animated
export @neuro, @rule, @node, @snapshot,@defop, GraphBuilder, call_rule, record_snapshot!
export backward_graph_sparse!
export RewriteRule, CompilerConfig, CompiledPlan, compile, scan_summary
export FULL_LLAMA_RULES, FULL_GPT_RULES, MEMORY_RULES
export is_dirty, recompile!
end