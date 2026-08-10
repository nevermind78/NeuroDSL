# Raccourci vers le depot LOCAL des sources d'articles (artilce/*.tex, *.md).
#
# Pourquoi ce depot existe separement : `artilce/*.tex` est dans le .gitignore du
# depot principal, et ces sources ne doivent PAS partir sur GitHub. Mais elles
# avaient besoin d'un historique -- son absence a rendu la v1 de NeuroDSL.tex
# irrecuperable le 2026-08-08, au moment precis ou il fallait savoir ce qui avait
# ete soumis a arXiv.
#
# Garanties : aucun remote configure, et un hook pre-push qui refuse tout envoi
# meme si un remote etait ajoute un jour.
#
# USAGE (depuis n'importe ou) :
#   .\papers-git.ps1 status
#   .\papers-git.ps1 log --oneline
#   .\papers-git.ps1 diff
#   .\papers-git.ps1 add -A
#   .\papers-git.ps1 commit -m "message"
#
# Astuce : pour en faire une commande courte, ajoutez a votre profil PowerShell
#   function pgit { & "C:\Users\Nevermind\Desktop\NeuroDSL\papers-git.ps1" @args }

$GitDir   = "C:\Users\Nevermind\.neurodsl_papers.git"
$WorkTree = "C:\Users\Nevermind\Desktop\NeuroDSL\artilce"

if (-not (Test-Path $GitDir)) {
    Write-Error "Depot absent : $GitDir"
    exit 1
}

& git --git-dir=$GitDir --work-tree=$WorkTree @args
exit $LASTEXITCODE
