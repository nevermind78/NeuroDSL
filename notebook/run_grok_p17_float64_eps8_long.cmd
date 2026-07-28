@echo off
set GROK_DEVICE=cpu
set GROK_P=17
set GROK_STEPS=20000
set GROK_SEED=999
set GROK_CLIP=1e6
set GROK_EPS=1e-8
set GROK_LR=1e-3
set GROK_WD=1.0
julia --project=C:\Users\Nevermind\Desktop\NeuroDSL C:\Users\Nevermind\Desktop\NeuroDSL\notebook\grokking_demo_neurodsl.jl > C:\Users\Nevermind\Desktop\NeuroDSL\notebook\grokking_p17_float64_eps8_long_run.log 2>&1
