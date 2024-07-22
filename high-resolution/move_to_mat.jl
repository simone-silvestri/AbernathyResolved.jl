using Statistics: mean
using JLD2
using MAT

file = jldopen("abernathey_channel_averages.jld2")

cx1 = file["timeseries/χu/9072879"];
cy1 = file["timeseries/χv/9072879"];
cz1 = file["timeseries/χw/9072879"];

bx1 = file["timeseries/∂xb²/9072879"];
by1 = file["timeseries/∂yb²/9072879"];
bz1 = file["timeseries/∂zb²/9072879"];

cx2 = file["timeseries/χu/9895450"];
cy2 = file["timeseries/χv/9895450"];
cz2 = file["timeseries/χw/9895450"];

bx2 = file["timeseries/∂xb²/9895450"];
by2 = file["timeseries/∂yb²/9895450"];
bz2 = file["timeseries/∂zb²/9895450"];

cx3 = file["timeseries/χu/10716950"];
cy3 = file["timeseries/χv/10716950"];
cz3 = file["timeseries/χw/10716950"];

bx3 = file["timeseries/∂xb²/10716950"];
by3 = file["timeseries/∂yb²/10716950"];
bz3 = file["timeseries/∂zb²/10716950"];

matfile = matopen("production_and_variance_derivatives.mat", "w")

write(matfile, "Px_0_10years", cx1)
write(matfile, "Py_0_10years", cy1)
write(matfile, "Pz_0_10years", cz1)

write(matfile, "dxb2_0_10years", bx1)
write(matfile, "dyb2_0_10years", by1)
write(matfile, "dzb2_0_10years", bz1)

write(matfile, "Px_10_20years", cx2)
write(matfile, "Py_10_20years", cy2)
write(matfile, "Pz_10_20years", cz2)

write(matfile, "dxb2_10_20years", bx2)
write(matfile, "dyb2_10_20years", by2)
write(matfile, "dzb2_10_20years", bz2)

write(matfile, "Px_20_30years", cx3)
write(matfile, "Py_20_30years", cy3)
write(matfile, "Pz_20_30years", cz3)

write(matfile, "dxb2_20_30years", bx3)
write(matfile, "dyb2_20_30years", by3)
write(matfile, "dzb2_20_30years", bz3)

close(matfile)