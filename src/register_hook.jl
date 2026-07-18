# ── Registration hook (zero-dependency, host-agnostic) ──────────────────────────────────────────────
#
# NeuroDSL registers PROCESS-GLOBAL compute state — custom ops (`CUSTOM_OPS` via `register_op!`), gradient
# rules (`GRAD_RULES` via `register_grad!`). These are pure side effects, invisible to a host's dataflow
# analysis. A host that runs ONE graph across several execution contexts (e.g. distributed workers) must
# re-establish those registrations in EVERY context, or compute there fails ("Opérateur inconnu" /
# "pas de règle backward pour :op").
#
# NeuroDSL names NO host and takes NO host dependency. It simply offers a NOTIFICATION callback a host may
# install; NeuroDSL fires it after each registration with the registered names, and the host does whatever
# it needs (typically: replay the registering code on its other contexts). Unset — the standalone case —
# it is a no-op.

const _REGISTER_HOOK = Ref{Any}(nothing)

"""
    set_register_hook!(f) -> nothing

Install a callback `f(names::Vector{Symbol})` invoked right after NeuroDSL registers process-global compute
state (`register_op!`, `register_grad!`). A HOST orchestrating one graph across multiple execution contexts
uses it to re-establish those registrations elsewhere — their defining code is a side effect its dataflow
can't see. Pass `nothing` to clear. NeuroDSL never sets it itself and is a no-op standalone.
"""
set_register_hook!(f) = (_REGISTER_HOOK[] = f; nothing)

# Fire the hook after a registration — no-op when unset; errors are swallowed so a misbehaving host
# integration can never fail a registration.
function _notify_register(names::Symbol...)
    h = _REGISTER_HOOK[]
    h === nothing && return nothing
    try
        h(collect(names))
    catch
    end
    return nothing
end
