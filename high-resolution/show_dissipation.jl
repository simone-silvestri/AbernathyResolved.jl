using JLD2
using CairoMakie
using SixelTerm

file = jldopen("dissipation.jld2")

iterations = parse.(Int, keys(file["timeseries/t"]))
Ni = length(iterations[end-30:end])

xu = file["timeseries/χᵁ/" * string(iterations[end-30])] ./ Ni
xv = file["timeseries/χⱽ/" * string(iterations[end-30])] ./ Ni
xw = file["timeseries/χᵂ/" * string(iterations[end-30])] ./ Ni

dbu = file["timeseries/∂xb/" * string(iterations[end-30])] ./ Ni
dbv = file["timeseries/∂yb/" * string(iterations[end-30])] ./ Ni
dbw = file["timeseries/∂zb/" * string(iterations[end-30])] ./ Ni

for iter in iterations[end-29:end]
    @info "iteration $iter"
    xu .+= file["timeseries/χᵁ/" * string(iter)] ./ Ni
    xv .+= file["timeseries/χⱽ/" * string(iter)] ./ Ni
    xw .+= file["timeseries/χᵂ/" * string(iter)] ./ Ni
    
    dbu .+= file["timeseries/∂xb/" * string(iter)] ./ Ni
    dbv .+= file["timeseries/∂yb/" * string(iter)] ./ Ni
    dbw .+= file["timeseries/∂zb/" * string(iter)] ./ Ni
end

# xu = 0.5 .* (xu[2:end, :, :] .+ xu[1:end-1, :, :])
# xv = 0.5 .* (xv[:, 2:end, :] .+ xv[:, 1:end-1, :])
# xw = 0.5 .* (xw[:, :, 2:end] .+ xw[:, :, 1:end-1])

# dbu = 0.5 .* (dbu[2:end, :, :] .+ dbu[1:end-1, :, :])
# dbv = 0.5 .* (dbv[:, 2:end, :] .+ dbv[:, 1:end-1, :])
# dbw = 0.5 .* (dbw[:, :, 2:end] .+ dbw[:, :, 1:end-1])

xum = mean(- xu[7:205, 7:335, 7:46], dims = 1)
xvm = mean(- xv[7:205, 7:335, 7:46], dims = 1)
xwm = mean(- xw[7:205, 7:335, 7:46], dims = 1)

dbu2 = 2 .* dbu[7:205, 7:335, 7:46].^2 
dbv2 = 2 .* dbv[7:205, 7:335, 7:46].^2 
dbw2 = 2 .* dbw[7:205, 7:335, 7:46].^2 

dbu2m = mean(dbu2, dims = 1)
dbv2m = mean(dbv2, dims = 1)
dbw2m = mean(dbw2, dims = 1)

κu = xum ./ dbu2m
κv = xvm ./ dbv2m
κw = xwm ./ dbw2m

totx  = - (xu[7:205, 7:335, 7:46] .+ xv[7:205, 7:335, 7:46] .+ xw[7:205, 7:335, 7:46])
totdb = 2 .* (dbu[7:205, 7:335, 7:46].^2 .+ dbv[7:205, 7:335, 7:46].^2 .+ dbw[7:205, 7:335, 7:46].^2)
