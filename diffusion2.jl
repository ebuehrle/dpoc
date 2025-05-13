using SumOfSquares, MosekTools, DynamicPolynomials, PGFPlots
using LinearAlgebra

@polyvar t x[1:2] u[1:2]
f = u
c = u'*u
K = @set([t;x;u]'*[t;x;u]<=10 && x'x>=1^2)

N = 10
T = range(0,1.5,N+1)[2:N+1]

m = SOSModel(Mosek.Optimizer)
@variable m V[1:N] Poly(monomials([t;x],0:8))
@objective m Max V[1](0.0,-0.8,-1.0)
@constraint m [i=1:N] differentiate(V[i],t) + differentiate(V[i],x)'*f + 3e-1*tr(differentiate(differentiate(V[i],x),x)) >= -c domain=K
c = @constraint m [i=1:N-1] subs(V[i],t=>T[i]) <= subs(V[i+1],t=>T[i]) domain=K
@constraint m subs(V[N],t=>T[N]) <= sum(100*(x-[1.5,0.0]).^2) domain=K
optimize!(m)

ν = dual.(c)
μ = expectation.(x',ν)
σ = sqrt.(expectation.(x'.^2,ν) - μ.^2)

save("diffusion2.pdf", Axis([
	Plots.Linear(T[1:N-1],μ[:,1],legendentry=L"\mu_1"),
	Plots.Linear(T[1:N-1],μ[:,2],legendentry=L"\mu_2"),
	Plots.Linear(T[1:N-1],σ[:,1],legendentry=L"\sigma_1"),
	Plots.Linear(T[1:N-1],σ[:,2],legendentry=L"\sigma_2"),
]))

save("diffusion2e.pdf", Axis([
	Plots.Linear(μ[:,1],μ[:,2]),
	Plots.Linear(cos.(range(0,2pi,100)),sin.(range(0,2pi,100)),style="blue, no marks, dashed"),
], xmin=-1.2, xmax=1.7, ymin=-1.2, ymax=1.2))
