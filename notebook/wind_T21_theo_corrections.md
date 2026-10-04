# WIND-T21 — Texte corrigé pour `theo.tex` (proposition de remplacement)

*Mode d'emploi.* Chaque section ci-dessous remplace les lignes indiquées de `theo.tex`. Le texte LaTeX est en anglais,
comme l'original, et prêt à coller.

Consigne de l'auteur respectée :
- le texte proposé ne contient ni aveu de faiblesse ni autocritique ;
- les énoncés sont exacts, les hypothèses explicites et les revendications calibrées sur ce qui est démontré ou
  mesuré.

La justification de chaque changement est dans `wind_T21_theo_audit.md`. Les chiffres viennent des fichiers WIND-T8
à T21.

Macros à conserver ; macros à ajouter :
```latex
\newcommand{\JT}{J^{\mathrm{T}}}
\newcommand{\JA}{J^{\mathrm{A}}}
\newcommand{\JAN}{J^{\mathrm{AN}}}
\newcommand{\JAH}{J^{\mathrm{AH}}}
\newcommand{\hatx}{\hat{x}}
```

---

## Titre et résumé (remplacent les l. 86–104)

```latex
\title{\Huge \textbf{Foundations of Geometric Interpretability}\\
\vspace{1em}
\LARGE \textit{Frozen Normalization, Scale Feedback and Conservation in Linearized Language Models}}
\author{\textbf{Abdallah Khemais}\\ \textit{Independent AI Researcher / ISITCom}}

\begin{abstract}
\noindent Linear attribution methods for large language models replace each layer by a linear map and pull the
differential of a readout back through these maps. We study this dual (cotangent) transport for the last-token
residual stream of Qwen2.5-1.5B, GPT-2 small and Gemma-2-2B, using exact per-block Jacobians.

Normalization layers make every branch input scale-invariant. Freezing their denominators, as done by LRP-style
conservation rules and attribution graphs, removes this invariance. We prove that the resulting correction to a
block Jacobian has rank at most the number of normalization sites in the block: two for Qwen2.5 and GPT-2, four for
Gemma-2. The full-depth effect is captured exactly by a lifted feedback loop between normalization ``sensors'' and
``injections''. Freezing turns the radial contraction of the true network into a positive scale feedback. A
second-order time-varying realization of this loop predicts the critical feedback gain $\theta_c$ of 50 fresh
prompts, with predictions locked by SHA-256 before evaluation: Spearman $0.971$, within $\pm15\%$ for $98\%$ of
prompts. $24\%$ of prompts are supercritical.

We then prove a conservation--fidelity theorem for gated MLPs. No neuron-wise linearization rule (gradient, half
rule, uniform rule) is both complete and exact on the directions transverse to its input. The unique linearization
with both properties is the gradient corrected by a rank-one radial term, $L^\star = J - E\,h^\top/\|h\|^2$, where
$E$ is the Euler excess. On real weights, every complete neuron-wise rule distorts the transverse derivative by at
least $15$--$23\%$ (Qwen2.5, Gemma-2), and the half rule by $51$--$64\%$. The corrected linearization removes the
collapse ($0/49$ prompts), restores the effective rank of the true Jacobian ($17.3$ vs $17.0$) and the depth profile
of attributions, within $0.04$ of the fidelity of the best non-conservative linearization. Across models, the stability margin is
governed by the long-range memory of the normalization feedback loop.
\end{abstract}
```

---

## Chapitre 1 (remplace les l. 109–154)

```latex
\chapter{Mathematical Background}

\section{Primal and dual spaces}
Let $\X = \R^D$ be the residual-stream space at a fixed token position, with $D = 1536$ (Qwen2.5-1.5B),
$D = 768$ (GPT-2 small) and $D = 2304$ (Gemma-2-2B).

\begin{definition}[Tangent and cotangent vectors]
A tangent vector $\Delta x \in T_x\X \cong \R^D$ is an infinitesimal intervention on the state $x$. A cotangent
vector (1-form) $\omega \in T_x^*\X$ is a linear functional on interventions. The differential of a scalar readout
$\mathcal{L}$ is the 1-form $\omega = d\mathcal{L}_x$, with $\Delta\mathcal{L} = \omega(\Delta x) + o(\|\Delta x\|)$.
\end{definition}

\begin{definition}[Radial gain and one-step stretch]
For a linear map $J_k$ between consecutive states $x_k, x_{k+1}$, with $\hatx = x/\|x\|$, define the radial gain
$\lambda_k = \langle \hatx_{k+1}, J_k\hatx_k\rangle$ and the one-step stretch $\|J_k\hatx_k\|$. A product of $L$
maps is \emph{amplifying along the state} if $\prod_k \lambda_k > 1$.
\end{definition}

\begin{theorem}[Sylvester's determinant identity]\label{thm:sylvester}
For $X \in \R^{m\times n}$ and $Y \in \R^{n\times m}$, $\det(I_m - XY) = \det(I_n - YX)$, and $XY$, $YX$ have the
same non-zero eigenvalues with multiplicities.
\end{theorem}
```

*Changements :*
- D de Gemma : 2048 devient 2304.
- La définition FTLE et la dichotomie « dissipatif/chaotique » sont remplacées par des quantités mesurables.
- La phrase sur Sylvester et Ho–Kalman est supprimée.
- Le « flot géométrique de difféomorphismes » est supprimé.

---

## Chapitre 2 (remplace les l. 157–207)

```latex
\chapter{Forward Flow and Dual Transport}

\section{The residual block}
Qwen2.5 and GPT-2 use sequential pre-normalized blocks:
\begin{equation}
h_k = x_k + a_k\big(n_{1,k}(x_k)\big), \qquad x_{k+1} = h_k + m_k\big(n_{2,k}(h_k)\big),
\end{equation}
with RMSNorm (Qwen2.5) or LayerNorm (GPT-2). Gemma-2 adds post-normalizations $p_{1,k}, p_{2,k}$ on both branch
outputs:
\begin{equation}
h_k = x_k + p_{1,k}\big(a_k(n_{1,k}(x_k))\big), \qquad x_{k+1} = h_k + p_{2,k}\big(m_k(n_{2,k}(h_k))\big).
\end{equation}
We study the last token $n$, with the states of the other positions fixed at their clean values. The block then
defines a map $x_k[n] \mapsto x_{k+1}[n]$ of $\R^D$.

\section{Four linearizations}
At the clean point we consider:
\begin{itemize}
\item $\JT_k = \partial x_{k+1}[n]/\partial x_k[n]$, the true Jacobian;
\item $\JA_k$, the Jacobian with attention probabilities pinned to their clean values;
\item $\JAN_k$, as $\JA_k$ with all normalization denominators frozen;
\item $\JAH_k$, as $\JAN_k$ with the gated MLP linearized by the half rule (gate nonlinearity $\sigma$ frozen,
gradient halved through the product).
\end{itemize}
The linearization $\JAN$ underlies LRP with the LayerNorm and attention-head rules; $\JAH$ underlies the
conservation rules used for gated MLPs.

\section{Dual transport}
\begin{definition}[Pullback of attributions]
Let $\omega_L = d\mathcal{L}_{x_L}$ be the differential of a readout. Its pullback to layer $k$ through a
linearization $J$ is $\omega_k = J_k^\top\,\omega_{k+1}$, i.e.\ reverse-mode differentiation through the linearized
network. The attribution of an intervention $\Delta x_k$ is $\omega_k(\Delta x_k)$.
\end{definition}
```

*Changements :*
- Le bloc séquentiel remplace le bloc parallèle, et le cas Gemma-2 est ajouté.
- La restriction au dernier token est explicite.
- Les quatre linéarisations sont nommées.
- « Transport parallèle » devient « pullback » (rétropropagation).

---

## Chapitre 3 (remplace les l. 210–256)

```latex
\chapter{Frozen Normalization as a Low-Rank Perturbation}

Frozen denominators make every normalization linear, which is what conservation-based attribution requires. We
compute exactly what this changes.

\section{The Jacobian of RMSNorm}
\begin{lemma}[RMSNorm Jacobian]\label{lem:rmsnorm}
Let $n(x) = \gamma\odot x/\rho(x)$ with $\rho(x) = (\|x\|^2/D + \varepsilon)^{1/2}$ and $\Gamma = \diag(\gamma)$.
Then
\begin{equation}
Dn(x) = \frac{\Gamma}{\rho}\big(I - s\,\hatx\hatx^\top\big), \qquad s = \frac{\|x\|^2}{\|x\|^2 + D\varepsilon}\in(0,1),
\qquad Dn(x)\,x = (1-s)\,n(x).
\end{equation}
The frozen-denominator map $F = \Gamma/\rho$ satisfies $F - Dn(x) = s\,n(x)\,x^\top/\|x\|^2$.
\end{lemma}
\begin{proof}
$\partial\rho/\partial x = x/(D\rho)$. Hence
$\partial(x_i/\rho)/\partial x_j = \rho^{-1}\big(\delta_{ij} - x_ix_j/(D\rho^2)\big)$ with $D\rho^2 = \|x\|^2 + D\varepsilon$.
\end{proof}
For $\varepsilon = 0$, each normalized branch input is invariant under $x \mapsto cx$, $c>0$. Freezing the
denominator removes exactly this invariance, through the rank-one term $s\,n(x)x^\top/\|x\|^2$.

\section{Rank of the freezing correction}
\begin{proposition}[Rank $\le q$]\label{prop:rank}
Consider a block whose Jacobian, for a fixed linearization of all other operations, is a composition involving $q$
normalization sites with inputs $z_1,\dots,z_q$, listed in topological order. Freezing the $q$ denominators changes
the block Jacobian by a matrix of rank at most $q$. Its row space is spanned by the live sensors
$c_i = (\partial z_i/\partial x)^\top \hat z_i$.
\end{proposition}
\begin{proof}
By Lemma~\ref{lem:rmsnorm}, freezing site $i$ replaces $Dn_i$ by $Dn_i + E_i$ with
$E_i = s_i\, n_i(z_i)\,z_i^\top/\|z_i\|^2$. Expanding the composition, the difference is a sum over non-empty sets
$S$ of sites of chains carrying $E_i$ at $i\in S$ and $Dn_j$ elsewhere. Let $i$ be the first site of $S$. All sites
before $i$ are live, so the chain ends with $z_i^\top(\partial z_i/\partial x)/\|z_i\|^2$, whose row is $c_i^\top$
up to a scalar, independently of $S$. Grouping by $i$ writes the difference as $\sum_{i=1}^q w_i c_i^\top$.
\end{proof}

\begin{corollary}[Sequential pre-norm block]
With pinned attention, $A_1 = Da\,F_1$, $M = Dm\,F_2$ and $Q_j = I - s_j\hat z_j\hat z_j^\top$:
\begin{equation}
\JAN_k - \JA_k = (I+M)(I+A_1) - (I+MQ_2)(I+A_1Q_1)
= s_2\,(M\hat h)\,\hat h^\top(I + A_1) \;+\; s_1\,(I + MQ_2)A_1\hatx\,\hatx^\top .
\end{equation}
\end{corollary}

\begin{remark}
Measured singular-value tails beyond rank $q$ are below $10^{-3}$: rank $2$ for Qwen2.5 and GPT-2, rank $4$ for
Gemma-2 (two pre- and two post-normalizations). The statement compares linearizations that treat attention
identically: if attention is live on one side only, the difference also contains the query--key terms.
\end{remark}
```

---

## Chapitre 4 (remplace les l. 259–331)

```latex
\chapter{Scale Feedback and the Critical Gain}

\section{Radial gains}
\begin{proposition}[Measured radial gains]
Across 50 prompts (Qwen2.5, GPT-2), the median radial gain is $\lambda = 0.846$ (true) vs.\ $1.130$ (frozen
normalization) for Qwen2.5, and $0.938$ vs.\ $1.411$ for GPT-2. The median one-step stretch $\|J_k\hatx_k\|$ is
$0.96$ vs.\ $1.22$ (Qwen2.5), $0.99$ vs.\ $1.55$ (GPT-2) and $0.99$ vs.\ $1.22$ (Gemma-2, live side with pinned
attention), over 10, 10 and 5 prompts respectively.
\end{proposition}
Freezing therefore turns a contracting radial direction into an amplifying one. The per-layer gain alone does not
determine the end-to-end behaviour: GPT-2 has the largest per-layer amplification and no rank collapse. The
end-to-end effect is governed by the feedback structure below.

\section{Exact lifting}
For a stack of $K$ blocks, write $J_k(\theta) = \JA_k + \theta\,U_kV_k^\top$, where $U_kV_k^\top = \JAN_k - \JA_k$ (Proposition~\ref{prop:rank}),
so that $\theta = 0$ is the live network and $\theta = 1$ the frozen one. Let $N$ be the true Jacobian of the final
normalization, and $\Phi_0(j,k) = \JA_{j-1}\cdots\JA_k$.
\begin{lemma}[Exact lifting]\label{lem:lift}
With $\mathcal{U}_k = N\Phi_0(K+1,k+1)U_k$, $\mathcal{V}_k = V_k^\top\Phi_0(k,1)$ and
$W_{jk} = V_j^\top\Phi_0(j,k+1)U_k$ for $j>k$ (zero otherwise),
\begin{equation}
N\,J_{K}(\theta)\cdots J_1(\theta) = N\Phi_0(K+1,1) + \theta\,\mathcal{U}(I-\theta W)^{-1}\mathcal{V}.
\end{equation}
Since $W$ is strictly block lower triangular, $(I-\theta W)^{-1} = \sum_{m<K}\theta^mW^m$ is a polynomial in
$\theta$.
\end{lemma}
\begin{proof}
Expand the product. A term is indexed by the increasing set $k_1<\dots<k_m$ of factors $\theta U V^\top$ and equals
$\theta^m\,\mathcal{U}_{k_m}W_{k_mk_{m-1}}\cdots W_{k_2k_1}\mathcal{V}_{k_1}$. Summing over sets of size $m$ gives
$\theta^m\mathcal{U}W^{m-1}\mathcal{V}$; summing over $m$ gives the claim.
\end{proof}
$W$ is the response of the normalization sensors of the live network to the injections of the frozen
normalizations. Equivalently, freezing a denominator multiplies the normalized signal by the gain
$\rho(z)/\rho(z_0)$, since $n_{\mathrm{fr}}(z) = (\rho(z)/\rho_0)\,n(z)$ exactly. The parameter $\theta$ is the gain
of this scale feedback.

\section{Stationary idealization}
\begin{proposition}[Threshold for a time-invariant loop]\label{prop:stationary}
Assume $\JA_k \equiv \Phi$, $U_k \equiv B$ and $V_k^\top \equiv C$. A resonant mode appears when the spectral radius
$\rho(\Phi + \theta BC)$ crosses $1$. If the crossing eigenvalue is $+1$ and $I - \Phi$ is invertible, the threshold
solves
\begin{equation}
\det\!\big(I_2 - \theta\, C(I-\Phi)^{-1}B\big) = 0 .
\end{equation}
\end{proposition}
\begin{proof}
$\det(I - \Phi - \theta BC) = \det(I-\Phi)\det(I - \theta (I-\Phi)^{-1}BC)$, and Theorem~\ref{thm:sylvester}
reduces the second factor to the $2\times2$ determinant. The growth statement follows from the eigen-decomposition
of $\Phi + \theta BC$.
\end{proof}
Real networks are not time-invariant ($\Phi$, $B$, $C$ change with depth), so we determine $\theta_c$ from the lifted
loop itself.

\section{Pre-registered prediction of the critical gain}
Define $\theta_c$ as the smallest gain at which the effective rank
$\exp H(\sigma_i^2/\textstyle\sum\sigma^2)$ of $N J_L(\theta)\cdots J_1(\theta)$ drops to $3$. A time-varying
Ho--Kalman realization of order $r$ is fitted on the exact loop $W$. It predicts $\theta_c$ without computing any
true product at $\theta\neq0$. Predictions were locked by SHA-256 before the ground truth was computed.

\begin{table}[h]\centering
\begin{tabular}{@{}llc@{}}\toprule
Pre-registered criterion & Threshold & Result (50 fresh prompts) \\ \midrule
C1: Spearman$(\theta_c^{\mathrm{model}}, \theta_c^{\mathrm{true}})$, $r=2$ & $\ge 0.70$ & $0.971$ \\
C2: fraction within $\pm15\%$, $r=2$ & $\ge 0.70$ & $0.98$ ($49/50$) \\
C3: balanced accuracy of collapse ($\theta_c<1$) & $\ge 0.80$ & $0.974$ (TP 12, FN 0, FP 2, TN 36) \\
C4: order 2 needed & see text & $r=1$: $0.866$; $r=2$: $0.971$; $r=3$: $0.977$ \\
D1: frequency of collapse & descriptive & $24\%$ ($12/50$; 95\% CI $14$--$37\%$) \\ \bottomrule
\end{tabular}
\caption{Critical gain of the frozen-normalization loop, Qwen2.5-1.5B, prompts disjoint from the discovery set.}
\end{table}

\begin{proposition}[Slope of the dominant mode]
If $\sigma_1(\theta)$ of $M(\theta) = N\prod_k J_k(\theta)$ is simple with singular vectors $(u,v)$, then
$\frac{d\sigma_1}{d\theta} = \sum_k u^\top N P_\theta(L\!\leftarrow\!k{+}1)\,U_kV_k^\top P_\theta(k\!\leftarrow\!1)\,v$.
\end{proposition}
\begin{proof}
For a simple singular value, $d\sigma_1 = u^\top (dM)\, v$, and the product rule gives
$dM/d\theta = \sum_k N P(L\leftarrow k+1)U_kV_k^\top P(k\leftarrow 1)$.
\end{proof}
```

*Changements :*
- Le « lemme d'inversion de Lyapunov » et le chaos sont supprimés ; ils sont remplacés par des gains mesurés.
- Le théorème « Ho–Kalman/Sylvester » devient une proposition explicitement stationnaire, avec critère de rayon
  spectral.
- Le tableau T8 est présenté comme validant la réalisation variant dans le temps.
- « OOD » devient « prompts neufs, disjoints de l'échantillon de découverte ».

---

## Chapitre 5 (remplace les l. 334–351)

```latex
\chapter{Computation}
All Jacobians are computed in NeuroDSL, a Julia framework in which the model is an addressable computation graph.
\begin{itemize}
\item Each block Jacobian at the last token is obtained by centered finite differences along the $D$ coordinate
directions. Only block $k{+}1$ is recomputed, by graph locality.
\item Linearization variants are implemented as operator substitutions: pinned attention probabilities, frozen
normalization denominators, half-rule gated product.
\item Three agreement gates validate each Jacobian: step $\varepsilon$ vs.\ $2\varepsilon$, the product of block
Jacobians vs.\ direct multi-layer directional derivatives, and the reverse-mode gradients of each variant vs.\ the
transposed products. Their median errors are below $3\cdot10^{-4}$ and their maximal errors below $1.5\cdot10^{-3}$.
\item Storage is $L\cdot D^2$ floats per variant ($264$\,MB for Qwen2.5-1.5B).
\end{itemize}
```

*Changement :* la description est remplacée par la méthode réellement employée. Les affirmations sur la VRAM et sur
les « Jacobiennes analytiques » sont supprimées.

---

## Chapitre 6 (remplace les l. 354–376)

```latex
\chapter{Concentration of the End-to-End Jacobian}
\begin{definition}[Pullback form]
For $\mathcal{T}_{k\to L} = J_{L-1}\cdots J_k$, the positive semi-definite form
$g_k = \mathcal{T}_{k\to L}^\top\mathcal{T}_{k\to L}$ measures the output sensitivity to interventions at layer
$k$ along the clean trajectory.
\end{definition}
\begin{proposition}[Empirical concentration, Qwen2.5-1.5B]
The end-to-end map $N\mathcal{T}_{1\to L}$ of the true network has median effective rank $17.0$ (entropy
definition, 50 prompts). The effective rank of $g_k$ increases monotonically as fewer layers are composed: from
$13.5$ at $k=1$ to $57$ at $k=15$, $142$ at $k=19$ and above $1000$ at $k = 27$. The participation ratio
$(\tr g_k)^2/\tr(g_k^2)$ rises from $4.4$ to $452$. The concentration is therefore built cumulatively by the
composition of layers $\sim10$--$27$. Long products also contain near-degenerate directions
($\sigma_{\min}/\sigma_{\max}<10^{-12}$ for products of nine or more blocks).
\end{proposition}
This concentration of the end-to-end Jacobian agrees with the funnel reported by Fernando and Guitchounts
(2026) on other models. Frozen-normalization linearizations preserve the dominant output subspace (mean $\cos^2$ of
top-10 subspaces $0.65$) while adding the resonant mode of Chapter~4.
```

*Changements :*
- La direction est corrigée ; le « théorème » devient une proposition empirique chiffrée.
- La définition d'erank est explicite.
- La citation de Fernando et Guitchounts est ajoutée.
- « Métrique riemannienne » devient « forme semi-définie positive le long de la trajectoire ».
- Funnel-LLM est supprimé : le sous-espace dominant dépend du prompt, et la moyenne des Jacobiennes a un erank de
  96 contre 20 par prompt (WIND-1). La proposition n'est donc pas étayée.

---

## Chapitre 7 (remplace les l. 379–432)

```latex
\chapter{Conservation versus Transverse Fidelity}

\section{Setting}
Consider a gated MLP $f(v) = W_d(\phi(W_g v)\odot W_u v)$ without biases, with frozen input normalization $v = Fh$.
Let $g = W_gv$, $u = W_uv$, $A_g = W_gF$ and $A_u = W_uF$. The exact derivative with respect to $h$ is
$J = W_d[\diag(u\odot\phi'(g))A_g + \diag(\phi(g))A_u]$. By Lemma~\ref{lem:rmsnorm}, $J$ coincides with the
derivative of the live network on the transverse hyperplane $h^\perp$; let $P = I - \hat h\hat h^\top$. A
linearization $L$ is \emph{complete} if $Lh = f(v)$ (conservation) and \emph{transverse-exact} if $LP = JP$. The
\emph{Euler excess} is $E = Jh - f(v) = W_d(u\odot\phi'(g)\odot g)$.

Neuron-wise rules form the class
$\mathcal{N} = \{L_{\beta,\gamma} = W_d[\diag(\beta)A_g + \diag(\gamma)A_u]\}$. It contains the gradient
($\beta^\ast = u\odot\phi'(g)$, $\gamma^\ast = \phi(g)$) and the half rule ($\beta = \tfrac12 u\odot\phi(g)/g$,
$\gamma = \tfrac12\phi(g)$). Write $\delta = (\beta-\beta^\ast,\gamma-\gamma^\ast)$,
$\Lambda_h\delta = W_d(\delta_\beta\odot g + \delta_\gamma\odot u)$, and let $\mathcal{G}$ be the Gram matrix with
$\|(L_{\beta,\gamma}-J)P\|_F^2 = \delta^\top\mathcal{G}\delta$:
\[
\mathcal{G} = \begin{pmatrix} (W_d^\top W_d)\odot(A_gPA_g^\top) & (W_d^\top W_d)\odot(A_gPA_u^\top)\\
(W_d^\top W_d)\odot(A_uPA_g^\top) & (W_d^\top W_d)\odot(A_uPA_u^\top)\end{pmatrix}.
\]

\begin{theorem}[Conservation versus transverse fidelity]\label{thm:ctf}
\begin{enumerate}[label=(\alph*)]
\item There is a unique linear map that is both complete and transverse-exact:
$L^\star = J - E\,h^\top/\|h\|^2$.
\item If $\mathcal{G}\succ 0$ and $\Lambda_h$ is onto, every complete $L\in\mathcal{N}$ satisfies
$\|(L-J)P\|_F \ge \tau^\star$ with $\tau^{\star2} = E^\top(\Lambda_h\mathcal{G}^{-1}\Lambda_h^\top)^{-1}E$. The bound
is attained by a unique complete rule, and $\tau^\star \ge \sqrt{\lambda_{\min}(\mathcal{G})}\,\|E\|/\sigma_{\max}(\Lambda_h)$.
\item Consequently the gradient is the only transverse-exact rule in $\mathcal{N}$, and its completeness defect is
$E$. If $E\neq0$, no rule in $\mathcal{N}$ is both complete and transverse-exact, and $L^\star\notin\mathcal{N}$.
The half rule is complete.
\end{enumerate}
\end{theorem}
\begin{proof}
(a) $\R^D = \operatorname{span}(h)\oplus h^\perp$. Transverse exactness fixes $L$ on $h^\perp$ and completeness
fixes $Lh$. $L^\star$ satisfies both because $h^\top P = 0$.

(b) For $L = L_{\beta^\ast+\delta_\beta,\gamma^\ast+\delta_\gamma}$, we have $L - J = \Lambda(\delta)$, linear in
$\delta$. Completeness is equivalent to $\Lambda_h\delta = -E$. Writing $\Lambda(\delta)P$ as
$\sum_i d_i\otimes(\delta_{\beta,i}a_i + \delta_{\gamma,i}b_i)$, where $d_i$ is the $i$-th column of $W_d$ and
$a_i, b_i$ are the projected rows of $A_g, A_u$, gives
$\langle d_i\otimes a_i, d_l\otimes b_l\rangle_F = (d_i^\top d_l)(a_i^\top b_l)$, hence
$\|\Lambda(\delta)P\|_F^2 = \delta^\top\mathcal{G}\delta$. Minimizing this strictly convex quadratic on the affine
set $\{\Lambda_h\delta = -E\}$ gives the unique minimizer and the value $\tau^{\star2}$. The lower bound follows from
$\delta^\top\mathcal{G}\delta\ge\lambda_{\min}\|\delta\|^2$ and $\|E\|\le\sigma_{\max}(\Lambda_h)\|\delta\|$.

(c) $\mathcal{G}\succ0$ makes $\delta\mapsto\Lambda(\delta)P$ injective. For the half rule,
$\delta_\beta\odot g + \delta_\gamma\odot u = -u\odot\phi'(g)\odot g$, so $\Lambda_h\delta = -E$.
\end{proof}

\begin{proposition}[The half rule as a radial secant]
Let $\tilde f(v) = W_d((\sigma_0\odot W_gv)\odot W_uv)$ be the bilinear surrogate with the gate nonlinearity frozen
at $\sigma_0 = \sigma(g_0)$, so that $\tilde f(v_0) = f(v_0)$. Then the half rule equals
$\tfrac12 D\tilde f(v_0) = \int_0^1 D\tilde f(tv_0)\,dt$, the integrated gradient along the ray $[0, v_0]$. For any
linear readout $r$,
$\langle r, (D\tilde f(v_0) - \tfrac12 D\tilde f(v_0))\,v_0\rangle = \tfrac12\, v_0^\top\nabla^2(r\cdot\tilde f)\,v_0$.
\end{proposition}
\begin{proof}
$\tilde f$ is bilinear, so $D\tilde f(tv) = t\,D\tilde f(v)$ and $\int_0^1 t\,dt = \tfrac12$. For the quadratic form
$q = r\cdot\tilde f$, we have $\nabla q(v)\cdot v = v^\top\nabla^2 q\,v = 2q(v)$.
\end{proof}
The half rule is thus exact along the radial ray, which carries conservation. By Theorem~\ref{thm:ctf}, a
neuron-wise rule pays for this with a distortion of at least $\tau^\star$ on the transverse directions, which carry
content.

\section{Measurements on real weights}
Theorem~\ref{thm:ctf} was evaluated at 18 pre-registered points (three layers, two prompts, three models), with
weights read from the released checkpoints:
\begin{itemize}
\item parity of the recomputed MLP output $\le1.1\cdot10^{-6}$;
\item $\|E\|/\|f\| \ge 0.87$;
\item $\mathcal{G}\succ0$ at every point ($\lambda_{\min}/\lambda_{\max}\ge3.7\cdot10^{-5}$).
\end{itemize}

\begin{table}[h]\centering
\begin{tabular}{@{}lccc@{}}\toprule
 & $r^\star = \tau^\star/\|JP\|_F$ & $r_{1/2}$ (half rule) & effective Euler degree \\ \midrule
Qwen2.5-1.5B & $0.145$--$0.185$ & $0.515$--$0.643$ & $0.99$--$1.84$ \\
Gemma-2-2B & $0.192$--$0.229$ & $0.512$--$0.566$ & $1.36$--$1.99$ \\
GPT-2 small (single family) & $0.229$--$0.611$ & --- & $-0.38$--$1.65$ \\ \bottomrule
\end{tabular}
\caption{Minimal transverse distortion of complete neuron-wise rules, and distortion of the half rule.}
\end{table}

\section{The corrected linearization}
Applying $L^\star$ to every MLP sublayer yields the REC linearization: frozen denominators, exact transverse
derivative, complete radial response. The attention sublayer with pinned probabilities and frozen normalization is
already complete and transverse-exact. REC requires one directional derivative per MLP to compute $E$.

\begin{table}[h]\centering
\begin{tabular}{@{}lcccc@{}}\toprule
Qwen2.5-1.5B, 49 prompts & collapses & eff.\ rank (true: $17.0$) & fidelity $\rho$ (true grad.: $0.840$) & early-layer share (truth: $0.211$) \\ \midrule
Frozen gradient (AN/ANf) & $12/50$ & $5.45$ & $0.593$ & $0.262$ \\
Half rule (AH/AHf) & $0/50$ & $91.6$ & $0.559$ & $0.105$ \\
\textbf{REC} ($L^\star$) & $\mathbf{0/49}$ & $\mathbf{17.3}$ & $\mathbf{0.691}$ & $\mathbf{0.202}$ \\ \bottomrule
\end{tabular}
\caption{Fidelity: median Spearman correlation between linear attributions of the 28 MLP sublayers and the effect
of mean-ablating each sublayer output at the last token. Early-layer share: fraction of the attribution mass in
layers 1--10. Predictions pre-registered.}
\end{table}

\section{Across models}
The lifted loop of Lemma~\ref{lem:lift} factors the amplification into an open-loop strength and a loop factor. By
Weyl's inequalities, the amplification $g_N$ lies in $[o\ell-1, o\ell+1]$. The loop memory $\|G\|/\|G_{\mathrm{loc}}\|$
is the ratio of the full resolvent to its one-step part.

\begin{table}[h]\centering
\begin{tabular}{@{}lcccc@{}}\toprule
 & open-loop $o$ & loop factor $\ell$ & loop memory $\|G\|/\|G_{\mathrm{loc}}\|$ & $g_N$ \\ \midrule
GPT-2 small & $0.29$ & $1.06$ & $1.1$ & $0.99$ \\
Qwen2.5-1.5B & $0.61$ & $3.14$ & $5.5$ & $1.95$ \\
Gemma-2-2B & $0.61$ & $6.22$ & $24$ & $4.00$ \\ \bottomrule
\end{tabular}
\caption{The stability margin of the frozen linearization tracks the memory of the live network's normalization
feedback. Local loop gain and Euler degree do not separate the models. Medians; the predictions on held-out prompts
were pre-registered.}
\end{table}
```

*Changements :*
- La « borne de Taylor–Hessienne », dimensionnellement incohérente, est remplacée par le théorème 1 de T20 et par
  l'identité exacte de sécante radiale.
- Le tableau de profondeur est remplacé par la version pré-enregistrée (T20-B4), qui porte un groupe et des
  composantes explicites.
- La référence RelP (2026) est corrigée.
- « Horizontal routing » est supprimé.

---

## Conclusion (remplace les l. 434–435)

```latex
\chapter{Conclusion}
Freezing normalization denominators linearizes each normalization by a rank-one change. Along the depth of the
network, these changes close a scale-feedback loop whose critical gain is predicted by a second-order time-varying
realization, and whose strength across models follows the memory of the live network's normalization feedback.
For gated MLPs, conservation and transverse fidelity are incompatible within neuron-wise rules
(Theorem~\ref{thm:ctf}). The unique linearization with both properties is a rank-one radial correction of the
gradient; it removes the collapse while preserving the geometry and the depth profile of the true network.
```

---

## Bibliographie (à ajouter avant `\end{document}`)

```latex
\begin{thebibliography}{99}
\bibitem{ali2022} A.~Ali, T.~Schnake, O.~Eberle, G.~Montavon, K.-R.~M\"uller, L.~Wolf. XAI for Transformers: Better Explanations through Conservative Propagation. ICML 2022.
\bibitem{achtibat2024} R.~Achtibat et al. AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers. ICML 2024.
\bibitem{relp2025} F.~Rezaei Jafari, O.~Eberle, A.~Khakzar, N.~Nanda. RelP: Faithful and Efficient Circuit Discovery in Language Models via Relevance Patching. arXiv:2508.21258, 2025.
\bibitem{arora2026} A.~Arora, Z.~Wu, J.~Steinhardt, S.~Schwettmann. Language Model Circuits Are Sparse in the Neuron Basis. arXiv:2601.22594, 2026.
\bibitem{fernando2026} J.~Fernando, G.~Guitchounts. Dynamics of the Transformer Residual Stream: Coupling Spectral Geometry to Network Topology. arXiv:2605.14258, 2026.
\bibitem{you2025} W.~You, S.~Zeng, Y.-H.~H.~Tsai, M.~Yamada, H.~Zhao. When LRP Diverges from Leave-One-Out in Transformers. arXiv:2510.18810, 2025.
\bibitem{sundararajan2017} M.~Sundararajan, A.~Taly, Q.~Yan. Axiomatic Attribution for Deep Networks. ICML 2017.
\bibitem{rushing2024} C.~Rushing, N.~Nanda. Explorations of Self-Repair in Language Models. ICML 2024.
\bibitem{ameisen2025} E.~Ameisen et al. Circuit Tracing: Revealing Computational Graphs in Language Models. Transformer Circuits Thread, 2025.
\bibitem{hokalman1966} B.~L.~Ho, R.~E.~Kalman. Effective construction of linear state-variable models from input/output functions. Regelungstechnik 14, 1966.
\bibitem{dewilde1998} P.~Dewilde, A.-J.~van der Veen. Time-Varying Systems and Computations. Kluwer, 1998.
\end{thebibliography}
```

*Note pour l'auteur, hors texte de l'article.* Vérifier les listes d'auteurs d'Ali et al. 2022 et d'Ameisen et
al. 2025 avant soumission : elles sont citées de mémoire. Celles de RelP, Transluce, Fernando–Guitchounts et You et
al. ont été vérifiées sur arXiv.
