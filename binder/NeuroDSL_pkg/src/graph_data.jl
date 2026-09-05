# ════════════════════════════════════════════════════════════════════════════════
# NeuroDSL — graph_data.jl
# Abstraction GraphData : sépare l'architecture (DAG) du runtime
# Compatible descendante : l'API existante (Backend.CPUDevice / CUDADevice)
# continue de fonctionner. GraphData est un opt-in pour les usages avancés.
# ════════════════════════════════════════════════════════════════════════════════

"""
    GraphData
Type abstrait racine. Tout comportement runtime (device, précision, checkpointing)
est encodé dans le type via multiple dispatch — zéro if/switch dans les Atoms.
"""
abstract type GraphData end

# ── 1. CPUTrainData ──────────────────────────────────────────────────────────
"""
    CPUTrainData()
Runtime CPU pur, Float32. Fallback sans CUDA.
"""
struct CPUTrainData <: GraphData
    device :: Backend.CPUDevice
end
CPUTrainData() = CPUTrainData(Backend.CPUDevice())

# ── 2. CUDATrainData ─────────────────────────────────────────────────────────
"""
    CUDATrainData()
Runtime CUDA, Float32, 1 GPU. Runtime par défaut si CUDA disponible.
"""
struct CUDATrainData <: GraphData
    device :: Backend.CUDADevice
end
CUDATrainData() = CUDATrainData(Backend.CUDADevice())

# ── 3. CheckpointData ────────────────────────────────────────────────────────
"""
    CheckpointData(inner; every=4)
Wrapper : stocke 1 activation sur `every` pendant le forward,
recompute les autres pendant le backward → −(1 − 1/every) × VRAM activations.
"""
struct CheckpointData <: GraphData
    inner :: GraphData
    every :: Int
end
CheckpointData(inner::GraphData = auto_graphdata(); every::Int = 4) =
    CheckpointData(inner, every)

# ── 4. MixedPrecData ─────────────────────────────────────────────────────────
"""
    MixedPrecData(inner; loss_scale)
Loss scaling dynamique : scale les gradients pour éviter l'underflow Float16.
Forward reste Float32 dans l'implémentation actuelle (les kernels BLAS
n'acceptent que Float32). La précision Float16 peut être activée nœud par nœud
via cast_fp16! sur les activations de sortie.
"""
mutable struct MixedPrecData <: GraphData
    inner      :: GraphData
    loss_scale :: Float32
    _ok        :: Bool       # false si overflow détecté au dernier step
end
MixedPrecData(inner::GraphData = auto_graphdata(); loss_scale::Float32 = Float32(2^15)) =
    MixedPrecData(inner, loss_scale, true)

# ── Protocole de capacités (dispatch compile-time, zéro overhead) ─────────────

"""Retourne le device physique (CPUDevice ou CUDADevice) d'un GraphData."""
get_device(d::CPUTrainData)   = d.device
get_device(d::CUDATrainData)  = d.device
get_device(d::CheckpointData) = get_device(d.inner)
get_device(d::MixedPrecData)  = get_device(d.inner)

"""Précision du forward pass."""
fwd_precision(::CPUTrainData)   = Float32
fwd_precision(::CUDATrainData)  = Float32
fwd_precision(::CheckpointData) = Float32
fwd_precision(::MixedPrecData)  = Float32   # Float16 opt-in nœud par nœud

"""Précision du backward — toujours Float32 pour la stabilité numérique."""
bwd_precision(::GraphData) = Float32

"""Gradient checkpointing activé ?"""
supports_checkpointing(::CheckpointData) = true
supports_checkpointing(::GraphData)      = false

"""Mixed precision avec loss scaling activée ?"""
supports_mixed_precision(::MixedPrecData) = true
supports_mixed_precision(::GraphData)     = false

"""Cadence des checkpoints."""
checkpoint_every(d::CheckpointData) = d.every
checkpoint_every(::GraphData)       = typemax(Int)

# ── Constructeur automatique selon CUDA disponible ───────────────────────────
"""
    auto_graphdata() → GraphData
Retourne CUDATrainData si CUDA est disponible, CPUTrainData sinon.
Miroir de Backend.active_device() pour le nouveau système.
"""
auto_graphdata() = Backend.CUDA_AVAILABLE ? CUDATrainData() : CPUTrainData()

# ── Compatibilité descendante ─────────────────────────────────────────────────
"""
    graphdata_from_backend(dev) → GraphData
Convertit l'ancien type de device en GraphData.
Permet au code existant de ne rien changer.
"""
graphdata_from_backend(::Backend.CPUDevice)  = CPUTrainData()
graphdata_from_backend(::Backend.CUDADevice) = CUDATrainData()

# ── Affichage ─────────────────────────────────────────────────────────────────
Base.show(io::IO, ::CPUTrainData)  = print(io, "CPUTrainData(Float32)")
Base.show(io::IO, ::CUDATrainData) = print(io, "CUDATrainData(Float32)")
Base.show(io::IO, d::CheckpointData) =
    print(io, "CheckpointData(every=", d.every, " | ", d.inner, ")")
Base.show(io::IO, d::MixedPrecData) =
    print(io, "MixedPrecData(scale=", d.loss_scale, " | ", d.inner, ")")
