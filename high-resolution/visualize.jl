using Statistics: mean

file = jldopen("abernathey_channel_averages.jld2")

cx = file["timeseries/χu/9072879"] ./ 3;
cy = file["timeseries/χv/9072879"] ./ 3;
cz = file["timeseries/χw/9072879"] ./ 3;

bx = file["timeseries/∂xb²/9072879"] ./ 3;
by = file["timeseries/∂yb²/9072879"] ./ 3;
bz = file["timeseries/∂zb²/9072879"] ./ 3;

for iter in [9895450, 10716950]
    cx .+= file["timeseries/χu/$iter"] ./ 3;
    cy .+= file["timeseries/χv/$iter"] ./ 3;
    cz .+= file["timeseries/χw/$iter"] ./ 3;

    bx .+= file["timeseries/∂xb²/$iter"] ./ 3;
    by .+= file["timeseries/∂yb²/$iter"] ./ 3;
    bz .+= file["timeseries/∂zb²/$iter"] ./ 3;
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

κi = - (cxc .+ cyc .+ czc) ./ (bxc .+ byc .+ bzc)

κx[isnan.(κx)] .= 0
κy[isnan.(κy)] .= 0
κz[isnan.(κz)] .= 0
κi[isnan.(κi)] .= 0

zC = file["grid/zᵃᵃᶜ"][7:end-6]
zF = file["grid/zᵃᵃᶠ"][7:end-6]

ΔzC = file["grid/Δzᵃᵃᶜ"][7:end-6]
ΔzF = file["grid/Δzᵃᵃᶠ"][7:end-6]

Δh = 5000

irange = 1:330

κxm = mean(κx[irange, :], dims = 1)[1, :] 
κym = mean(κy[irange, :], dims = 1)[1, :] 
κzm = mean(κz[irange, :], dims = 1)[1, :] 
κim = mean(κi[irange, :], dims = 1)[1, :] 
