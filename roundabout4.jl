using CSV, DataFrames
using SumOfSquares, MosekTools
using MultivariateMoments
using DynamicPolynomials
using LinearAlgebra
using PGFPlots

D = CSV.read("vehicle_tracks_000.csv", DataFrame) |>
    (d -> d[:,["x","y","vx","vy"]]) |>
    (d -> d .- [1000 1000 0 0]) |>
    (d -> d ./ [20 20 20 20]) |>
    (d -> filter(e -> -1 <= e[1] <= 1, d)) |>
    (d -> filter(e -> -1 <= e[2] <= 1, d))

@polyvar x[1:4] u[1:2]
f = [x[3:4]; u]
d = 4
φ = monomials(x,0:2d)

M = sum(dirac(φ,(x.=>collect(s))...) for s in eachrow(D)) * (1/size(D,1))
Q = let v = monomials(x,0:d)
    v'*inv(expectation.(v*v',[M]))*v
end

m = SOSModel(Mosek.Optimizer)
@variable m V Poly(φ)
@objective m Max V(0.2,-1.0,0.0, 0.0) - V(-1.0, 0.8, 0.0, 0.0)
c = @constraint m differentiate(V,x)'*f >= -Q domain=@set([x;u]'*[x;u]<=10)
optimize!(m)

μ = dual(c)
q = let v = monomials(x[1:2],0:d)
    v'*inv(expectation.(v*v',[μ]))*v
end
save("roundabout4.pdf", Axis([
    Plots.Image((x...)->1/q(x...),(-1,1),(-1,1)),
    Plots.Quiver(
        D[1:50:end, "x"],
        D[1:50:end, "y"],
        D[1:50:end, "vx"]/3,
        D[1:50:end, "vy"]/3,
        style="-stealth, blue, no markers"
    ),
], xmin=-1, xmax=1, ymin=-1, ymax=1))
