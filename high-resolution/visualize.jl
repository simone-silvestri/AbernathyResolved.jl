using CairoMakie, SixelTerm, JLD2
using Statistics: mean

file = jldopen("abernathey_channel_averages.jld2")

iteration = keys(file["timeseries/t"])[end]

cx = file["timeseries/χu/" * iteration] 
cy = file["timeseries/χv/" * iteration] 
cz = file["timeseries/χw/" * iteration] 

bx = file["timeseries/∂xb²/" * iteration]
by = file["timeseries/∂yb²/" * iteration]
bz = file["timeseries/∂zb²/" * iteration]

# for iter in iterations[9895450, 10716950]
#     cx .+= file["timeseries/χu/$iter"] ./ 3;
#     cy .+= file["timeseries/χv/$iter"] ./ 3;
#     cz .+= file["timeseries/χw/$iter"] ./ 3;
# 
#     bx .+= file["timeseries/∂xb²/$iter"] ./ 3;
#     by .+= file["timeseries/∂yb²/$iter"] ./ 3;
#     bz .+= file["timeseries/∂zb²/$iter"] ./ 3;
# end

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

zC = file["grid/zᵃᵃᶜ"][7:end-6]
zF = file["grid/zᵃᵃᶠ"][7:end-6]

ΔzC = file["grid/Δzᵃᵃᶜ"][7:end-6]
ΔzF = file["grid/Δzᵃᵃᶠ"][7:end-6]

Δh = 5000

irange = 1:390

κixm = mean(κix[irange, :], dims = 1)[1, :] 
κiym = mean(κiy[irange, :], dims = 1)[1, :] 
κizm = mean(κiz[irange, :], dims = 1)[1, :] 
κxm = mean(κx[irange, :], dims = 1)[1, :] 
κym = mean(κy[irange, :], dims = 1)[1, :] 
κzm = mean(κz[irange, :], dims = 1)[1, :] 
κim = mean(κi[irange, :], dims = 1)[1, :] 

jldsave("mytest.jld2";  κixm, κiym, κizm, κxm,κym, κzm, κx, κy, κz)


fig = Figure(); ax = Axis(fig[1, 1], ylabel = "z [m]", xlabel = "Diffusivity [m²/s]")

lines!(ax, κixm, zC, color = :blue,  linewidth = 2, label = "κx Sx² = - ⟨Px⟩ / ⟨(∂zb)²⟩") 
lines!(ax, κiym, zC, color = :green, linewidth = 2, label = "κy Sy² = - ⟨Py⟩ / ⟨(∂zb)²⟩") 
lines!(ax, κzm,  zF, color = :red,   linewidth = 2, label = "κz     = - ⟨Pz⟩ / ⟨(∂zb)²⟩") 

vlines!(ax, -1e-5; linestyle = :dash, linewidth = 0.5, color = :grey)
vlines!(ax,  1e-5; linestyle = :dash, linewidth = 0.5, color = :grey)
axislegend(ax, position = :rc) 
xlims!(ax, (-3e-5, 4e-5))

fig2 = Figure(); ax = Axis(fig2[1, 1], ylabel = "z [m]", xlabel = "log(abs(diffusivity))")

lines!(ax, log10.(abs.(κixm)), zC, color = :blue,  linewidth = 2, label = "abs(κx Sx²)") 
lines!(ax, log10.(abs.(κiym)), zC, color = :green, linewidth = 2, label = "abs(κy Sy²)") 
lines!(ax, log10.(abs.(κzm)),  zF, color = :red,   linewidth = 2, label = "abs(κz)") 
lines!(ax, log10.(abs.(κixm .+ κiym .+ κizm)), zC, color = :black, linewidth = 2, label = "abs(total)", linestyle = :dash) 

vlines!(ax, -5; linestyle = :dash, linewidth = 0.5, color = :grey)
axislegend(ax, position = :lc) 

xlims!(ax, (-10, -3))
