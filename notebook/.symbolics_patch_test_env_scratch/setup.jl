import Pkg
Pkg.activate(@__DIR__; io=devnull)
Pkg.develop(path=raw"C:\Users\Nevermind\.julia\packages\Symbolics\mLvup"; io=devnull)
Pkg.develop(path=raw"C:\Users\Nevermind\.julia\packages\SymbolicUtils\c9cTZ"; io=devnull)
Pkg.add(["Test"]; io=devnull)
println("scratch test env ready")
