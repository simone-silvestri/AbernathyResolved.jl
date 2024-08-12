using CairoMakie, SixelTerm, JLD2
using Statistics: mean

file = jldopen("abernathey_channel_averages.jld2")

iteration = keys(file["timeseries/t"])[end]

# file = jldopen("mytest.jld2")

if isnothing(iteration)
cx = file["cx"]
cy = file["cy"]
cz = file["cz"]
bx = file["bx"]
by = file["by"]
bz = file["bz"]
else
cx = file["timeseries/χu/" * iteration] 
cy = file["timeseries/χv/" * iteration] 
cz = file["timeseries/χw/" * iteration]  
bx = file["timeseries/∂xb²/" * iteration]
by = file["timeseries/∂yb²/" * iteration]
bz = file["timeseries/∂zb²/" * iteration]
end

cxm = mean(cx, dims = 1)[1, :, :]
cym = mean(cy, dims = 1)[1, :, :]
czm = mean(cz, dims = 1)[1, :, :]

bxm = mean(bx, dims = 1)[1, :, :]
bym = mean(by, dims = 1)[1, :, :]
bzm = mean(bz, dims = 1)[1, :, :]

κx = - cxm ./ bxm ./ 2
κy = - cym ./ bym ./ 2
κz = - czm ./ bzm ./ 2

cxc = deepcopy(cxm)
cyc = (cym[1:end-1, :] .+ cym[2:end, :]) / 2
czc = (czm[:, 1:end-1] .+ czm[:, 2:end]) / 2

bxc = deepcopy(bxm)
byc = (bym[1:end-1, :] .+ bym[2:end, :]) / 2
bzc = (bzm[:, 1:end-1] .+ bzm[:, 2:end]) / 2

κi = - (cxc .+ cyc .+ czc) ./ (bxc .+ byc .+ bzc) ./ 2

κix = - cxc ./ bzc ./ 2
κiy = - cyc ./ bzc ./ 2 
κiz = - czc ./ bzc ./ 2 

κx[isnan.(κx)] .= 0
κy[isnan.(κy)] .= 0
κz[isnan.(κz)] .= 0
κi[isnan.(κi)] .= 0

Nz = 90
Δz = [10.0 * ones(6)...,
      11.25, 12.625, 14.125, 15.8125, 17.75, 19.9375, 22.375, 25.125, 28.125, 31.625, 35.5, 39.75,
      42.0 * ones(56)...,
      39.75, 35.5, 31.625, 28.125, 25.125, 22.375, 19.9375, 17.75, 15.8125, 14.125, 12.625, 11.25,
      10.0 * ones(4)...]

z_faces = zeros(Nz+1)
for k in Nz : -1 : 1
    z_faces[k] = z_faces[k+1] - Δz[Nz - k + 1]
end

zF = z_faces
zC = (zF[1:end-1] .+ zF[2:end]) ./ 2

Δh = 5000

irange = 1:390

κixm = mean(κix[irange, :], dims = 1)[1, :] 
κiym = mean(κiy[irange, :], dims = 1)[1, :] 
κizm = mean(κiz[irange, :], dims = 1)[1, :] 
κxm = mean(κx[irange, :], dims = 1)[1, :] 
κym = mean(κy[irange, :], dims = 1)[1, :] 
κzm = mean(κz[irange, :], dims = 1)[1, :] 
κim = mean(κi[irange, :], dims = 1)[1, :] 

# jldsave("mytest.jld2";  κixm, κiym, κizm, κxm,κym, κzm, κx, κy, κz)


fig = Figure(); ax = Axis(fig[1, 1], ylabel = "z [m]", xlabel = "Diffusivity [m²/s]")

lines!(ax, κixm, zC, color = :blue,  linewidth = 2, label = "κx Sx² = - ⟨Px⟩ / ⟨(∂zb)²⟩") 
lines!(ax, κiym, zC, color = :green, linewidth = 2, label = "κy Sy² = - ⟨Py⟩ / ⟨(∂zb)²⟩") 
lines!(ax, κzm,  zF, color = :red,   linewidth = 2, label = "κz     = - ⟨Pz⟩ / ⟨(∂zb)²⟩") 

vlines!(ax, -1e-5; linestyle = :dash, linewidth = 0.5, color = :grey)
vlines!(ax,  1e-5; linestyle = :dash, linewidth = 0.5, color = :grey)
axislegend(ax, position = :rc) 
xlims!(ax, (-2e-5, 1.5e-4))

fig2 = Figure(); ax = Axis(fig2[1, 1], ylabel = "z [m]", xlabel = "log(abs(diffusivity))")

lines!(ax, log10.(abs.(κixm)), zC, color = :blue,  linewidth = 2, label = "abs(κx Sx²)") 
lines!(ax, log10.(abs.(κiym)), zC, color = :green, linewidth = 2, label = "abs(κy Sy²)") 
lines!(ax, log10.(abs.(κzm)),  zF, color = :red,   linewidth = 2, label = "abs(κz)") 
lines!(ax, log10.(abs.(κixm .+ κiym .+ κizm)), zC, color = :black, linewidth = 2, label = "abs(total)", linestyle = :dash) 

vlines!(ax, -5; linestyle = :dash, linewidth = 0.5, color = :grey)
axislegend(ax, position = :lc) 

xlims!(ax, (-10, -3))
