module NeuroDSL
using Printf, Statistics, Random, LinearAlgebra, Dates,JSON

include("backend.jl")
using .Backend


include("types.jl")
include("graph_api.jl")
include("kernels.jl")
include("dispatch.jl")
include("kv_cache.jl")
include("demand_release.jl")
include("grad_pool.jl")
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
include("patching.jl")
include("serialization.jl")
include("synthetic_circuits.jl")
include("graph_surgery.jl")


export NeuroGraph, GraphNode, GraphRule, CtxStore
export NeurAtom, Datom, Quantom, is_backpropable
export set!, node, addrule!, demand!, params
export activate!, namespaces, graph_summary
export topo_order!, zero_grads!, invalidate_all!, copy_params_to_namespace!, alias_tied_param!
export backward_graph!, accum_grad!, GRAD_RULES, register_op!, CUSTOM_SHAPE_RULES
export rmsnorm_fwd!, rmsnorm_bwd!, swiglu_fwd!, swiglu_bwd!
export softmax_fwd!, softmax_bwd!, cross_entropy_loss, cross_entropy_grad
export mse_loss_fwd, mse_loss_bwd, adamw_step!, adamw_step_batched!
export GraphData, CPUTrainData, CUDATrainData, CheckpointData, MixedPrecData
export auto_graphdata, graphdata_from_backend, get_device, fwd_precision, bwd_precision, supports_checkpointing, supports_mixed_precision, checkpoint_every
export LivenessInterval, BufferPool, MemoryPlan, plan_memory!, demand_planned!, compute_liveness, pool_stats
export GradPool, grad_acquire!, grad_release!, grad_owns, empty_grad_pool!
export demand_release!, release_intermediates!
export CheckpointSchedule, forward_with_checkpointing!, backward_with_checkpointing!
export flash_attn_fwd!, flash_attn_bwd!, MultiHeadFlashAttention, flash_attn_fwd_cpu!, flash_attn_fwd_cpu_simple!, flash_attn_bwd_cpu!
export cast_fp16, cast_fp32, LossScaleTracker, update!, backward_with_loss_scaling!, mixed_precision_step!
export LayerNorm, Linear, Embedding, MultiHeadAttention, LlamaBlock, LlamaModel
export CachedMultiHeadAttention, CachedLlamaBlock, CachedLlamaModel, build_cached_decode_graph!
export prime_kv_cache_from_prefix!
export @addrules, Backend, debug!
export _watch!, _fuse!, _invalidate_upstream!, _invalidate_downstream! 
export ExecutionLog, log_event!
export save_interactive_graph, graph_to_json
export TrainingSnapshot, TrainingRecorder, should_capture, capture_snapshot
export save_interactive_graph_animated
export @neuro, @rule, @node, @snapshot,@defop, GraphBuilder, call_rule, record_snapshot!
export backward_graph_sparse!
export RewriteRule, CompilerConfig, CompiledPlan, compile, scan_summary
export FULL_LLAMA_RULES, FULL_GPT_RULES, MEMORY_RULES
export is_dirty, recompile!
export capture_activations, patch_node!, recovery_metric, patch_and_measure!
export restore_from_cache!, restore_from_cache_batched!, sweep_patch_sites!
export patch_nodes!, restore_nodes_from_cache!
export greedy_patch_search!
export backward_prune!
export AdamWState, save_graph!, load_graph!, save_all_graph!, load_all_graph!, extend_adamw_state!
export insert_block!
export graft_shadow_block!
export position_patch_cache, set_params!
export selection_circuit_weights, build_selection_circuit
export build_induction_graph, sample_induction_sequence, train_induction!, evaluate_induction
export build_multihop_graph, sample_multihop_sequence, train_multihop!, evaluate_multihop
end