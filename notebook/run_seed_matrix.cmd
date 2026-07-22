@echo off
rem Matrice de graines -- 4 graines sequentielles (11 22 33 44), ~1h15-1h45 chacune.
rem Log unique : notebook\marker_seed_matrix.log (marqueurs DEBUT/FIN par graine).
cd /d C:\Users\Nevermind\Desktop\NeuroDSL
echo ===== MATRICE DE GRAINES : demarrage %DATE% %TIME% ===== > notebook\marker_seed_matrix.log
for %%S in (11 22 33 44) do (
  echo ===== SEED %%S DEBUT %DATE% %TIME% ===== >> notebook\marker_seed_matrix.log
  set MARKER_MATRIX_SEED=%%S
  julia --project=. notebook\marker_seed_matrix.jl >> notebook\marker_seed_matrix.log 2>&1
  echo ===== SEED %%S FIN ===== >> notebook\marker_seed_matrix.log
)
echo ===== MATRICE TERMINEE ===== >> notebook\marker_seed_matrix.log
