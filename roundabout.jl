using CSV, DataFrames
using MomentOpt, MosekTools
using DynamicPolynomials
using LinearAlgebra
using PGFPlots

D = CSV.read("vehicle_tracks_000.csv", DataFrame) |>
    (d -> d[:,["x","y","vx","vy"]]) |>
    (d -> d .- [1000 1000 0 0]) |>
    (d -> d ./ [20 20 20 20]) |>
    (d -> filter(e -> -1 <= e[1] <= 1, d)) |>
    (d -> filter(e -> -1 <= e[2] <= 1, d))

@polyvar t x[1:4] u[1:2]
f = [x[3:4]; u]
d = 3
φ = monomials([t;x],0:2d)

M = sum(DiracMeasure(x,collect(s)) for s in eachrow(D)) / size(D,1)
Q = let v = monomials(x,0:d)
    v'*inv(integrate.(v*v',M))*v
end

let v = monomials(x[1:2],0:d)
    q = v'*inv(integrate.(v*v',M))*v
    save("christoffel.pdf", Axis([
        Plots.Image((x...)->-log(q(x...)),(-1,1),(-1,1)),
        Plots.Quiver(
            D[1:50:end, "x"],
            D[1:50:end, "y"],
            D[1:50:end, "vx"],
            D[1:50:end, "vy"],
            style="-stealth, blue, no markers"
        ),
    ], xmin=-1, xmax=1, ymin=-1, ymax=1))
end

m = GMPModel(Mosek.Optimizer)
@variable m μ  Meas([t;x;u],support=@set([t;x;u]'*[t;x;u]<=10))
@variable m μ0 Meas([t;x;u],support=@set([t;x;u]'*[t;x;u]<=10 && t==0))
@variable m μT Meas([t;x;u],support=@set([t;x;u]'*[t;x;u]<=10 && t==3))
@objective m Min Mom(Q,μ)
@constraint m [i=1:length(φ)] Mom.(differentiate(φ[i],t) + differentiate(φ[i],x)'*f, μ) == Mom.(φ[i], μT-μ0)
@constraint m Mom.(monomials(x,0:2d), μ0) .== integrate.(monomials(x,0:2d), DiracMeasure(x,[ 0.3,-0.5,0.0, 0.0]))
optimize!(m)

q = let v = monomials(x[1:2],0:d)
    v'*inv(integrate.(v*v',μ))*v
end
save("roundabout.pdf", Plots.Image((x...)->1/q(x...),(-1,1),(-1,1)))
