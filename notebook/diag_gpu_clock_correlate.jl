# Corrèle diag_gpu_clock_during_decode_trace.csv (nvidia-smi, échantillonné
# toutes les ~25ms) avec diag_gpu_clock_during_decode_steps.csv (horodatages
# epoch de chaque pas de décodage Julia) -- sans dépendance externe (pas de
# CSV.jl/DataFrames.jl, interdiction d'ajouter des deps à Project.toml).
using Dates

const DIR = @__DIR__
clock_lines = filter(l -> count(==(','), l) >= 3, readlines(joinpath(DIR, "diag_gpu_clock_during_decode_trace.csv"))[2:end])
step_lines  = filter(l -> count(==(','), l) >= 4, readlines(joinpath(DIR, "diag_gpu_clock_during_decode_steps.csv"))[2:end])

# nvidia-smi timestamp local, format "2026/08/31 16:34:37.565" -- pas de
# fuseau explicite ; on suppose la même horloge système que `time()` de
# Julia (heure locale), donc on convertit en "secondes depuis minuit local"
# et on aligne par SOUSTRACTION D'UN DÉCALAGE CONSTANT calé sur le PREMIER
# timestamp du bloc nvidia-smi vs le PREMIER `t_before_wall` du bloc Julia
# (les deux processus tournent sur la même machine, même horloge murale).
function parse_clock_row(line)
    parts = split(line, ",")
    ts_str = strip(parts[1])
    sm_str = strip(parts[2])
    sm = parse(Int, split(sm_str)[1])
    dt = DateTime(ts_str, dateformat"yyyy/mm/dd HH:MM:SS.sss")
    return (dt=dt, sm=sm)
end
clock_rows = [parse_clock_row(l) for l in clock_lines]

function parse_step_row(line)
    parts = split(line, ",")
    return (step=parse(Int,parts[1]), cur_step=parse(Int,parts[2]),
            t_before=parse(Float64,parts[3]), t_after=parse(Float64,parts[4]), dur=parse(Float64,parts[5]))
end
step_rows = [parse_step_row(l) for l in step_lines]

# Décalage horloge : temps epoch Unix (Julia `time()`) vs DateTime local
# nvidia-smi -- on aligne sur le PREMIER échantillon nvidia-smi et le DÉBUT
# du script Julia (proche en pratique, le tout tourne sur quelques dizaines
# de secondes) en supposant que le PREMIER timestamp clock correspond au
# DÉMARRAGE du script (avant TRACE_START). Plus robuste : on cherche le
# décalage qui aligne le milieu de la fenêtre de génération. On procède par
# recherche du décalage `offset` tel que epoch_clock = datetime2unix(dt) +
# offset colle avec step_rows' t_before/t_after (déjà en epoch Unix) --
# comme les deux processus partagent la même horloge système (heure locale
# vs UTC), `datetime2unix` suppose déjà UTC alors que nvidia-smi imprime
# l'heure LOCALE -- on doit donc soustraire l'écart fuseau. On le déduit
# directement : offset = t_first_step_before - datetime2unix(dt_first_clock_row)
# (calé sur le début, la dérive d'horloge sur <1 min est négligeable).
offset = step_rows[1].t_before - datetime2unix(clock_rows[1].dt)
_log(msg) = println(msg)
_log("Décalage horloge estimé (s) : $offset")

clock_epoch = [(t=datetime2unix(r.dt)+offset, sm=r.sm) for r in clock_rows]

# Pour chaque pas de décodage, min/max/moyenne du clock SM pendant SA fenêtre
# [t_before, t_after], ET pendant le "trou CPU" [t_after(step-1), t_before(step)].
println("\n","="^100)
println("Corrélation horloge GPU (sm clock, MHz) vs pas de décodage")
println("="^100)
println(rpad("step",5), rpad("cur_step",9), rpad("dur(ms)",9), rpad("clk_min",9), rpad("clk_max",9), rpad("clk_mean",10), "n_samples")
for (i, sr) in enumerate(step_rows)
    samples_in_step = [c.sm for c in clock_epoch if sr.t_before <= c.t <= sr.t_after]
    if isempty(samples_in_step)
        println(rpad(sr.step,5), rpad(sr.cur_step,9), rpad(round(1000*sr.dur,digits=1),9), "  (aucun échantillon dans la fenêtre)")
        continue
    end
    println(rpad(sr.step,5), rpad(sr.cur_step,9), rpad(round(1000*sr.dur,digits=1),9),
            rpad(minimum(samples_in_step),9), rpad(maximum(samples_in_step),9),
            rpad(round(sum(samples_in_step)/length(samples_in_step),digits=1),10), length(samples_in_step))
end

# Trou CPU entre pas consécutifs (step i-1 -> step i) : clock pendant ce trou.
println("\n","-"^100)
println("Horloge pendant les TROUS CPU-seuls entre pas consécutifs (argmax+set!+invalidate_all!, PAS de travail GPU) :")
println(rpad("between",14), rpad("gap(ms)",9), rpad("clk_min",9), rpad("clk_max",9), "clk_mean")
for i in 2:length(step_rows)
    gap_lo = step_rows[i-1].t_after
    gap_hi = step_rows[i].t_before
    gap_ms = 1000*(gap_hi - gap_lo)
    samples_in_gap = [c.sm for c in clock_epoch if gap_lo <= c.t <= gap_hi]
    if isempty(samples_in_gap)
        println(rpad("$(i-1)->$i",14), rpad(round(gap_ms,digits=2),9), "  (aucun échantillon, trou trop court)")
    else
        println(rpad("$(i-1)->$i",14), rpad(round(gap_ms,digits=2),9),
                rpad(minimum(samples_in_gap),9), rpad(maximum(samples_in_gap),9), round(sum(samples_in_gap)/length(samples_in_gap),digits=1))
    end
end

overall_min = minimum(c.sm for c in clock_epoch if step_rows[1].t_before <= c.t <= step_rows[end].t_after)
overall_max = maximum(c.sm for c in clock_epoch if step_rows[1].t_before <= c.t <= step_rows[end].t_after)
n_low = count(c -> step_rows[1].t_before <= c.t <= step_rows[end].t_after && c.sm < 1000, clock_epoch)
n_tot = count(c -> step_rows[1].t_before <= c.t <= step_rows[end].t_after, clock_epoch)
println("\n","="^100)
println("Résumé fenêtre de génération complète ($(length(step_rows)) pas) :")
println("  clock SM min=$(overall_min)MHz max=$(overall_max)MHz")
println("  échantillons < 1000MHz : $n_low / $n_tot ($(round(100*n_low/n_tot,digits=1))%)")
