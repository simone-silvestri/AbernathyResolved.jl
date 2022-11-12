using Oceananigans
using Oceananigans.Utils
using Oceananigans.Units
using Oceananigans.Operators
using Oceananigans.Grids: architecture, znode, halo_size, on_architecture
using Oceananigans.Architectures: device, device_event, arch_array
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Fields: index_binary_search

using JLD2

# Architecture
arch = CPU()

# number of grid points
Nx = 400
Ny = 800
Nz = 30

# stretched grid 
k_center  = collect(1:Nz)
Δz_center = @. 10 * 1.125^(Nz - k_center)

const Lx = 1000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]
const Lz = sum(Δz_center)

z_faces = vcat([-Lz], -Lz .+ cumsum(Δz_center))
z_faces[Nz+1] = 0

grid = RectilinearGrid(arch,
                       topology = (Periodic, Bounded, Bounded),
                       size = (Nx, Ny, Nz),
                       halo = (5, 5, 5),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = z_faces)


file = jldopen("abernathey_channel_fields_fluxform_weno.jld2")

function load_prognostic_fields(file, grid)
    u = Field[]
    v = Field[]
    w = Field[]
    b = Field[]
    c = Field[]

    for iter in keys(file["timeseries/t"])
        @info "time $iter of $(keys(file["timeseries/t"])[end])"
        push!(u, set!( XFaceField(grid), file["timeseries/u/" * iter]))
        push!(v, set!( YFaceField(grid), file["timeseries/v/" * iter]))
        push!(w, set!( ZFaceField(grid), file["timeseries/w/" * iter]))
        push!(b, set!(CenterField(grid), file["timeseries/b/" * iter]))
        push!(c, set!(CenterField(grid), file["timeseries/c/" * iter]))
    end

    return u, v, w, b, c
end

GPUgrid = on_architecture(GPU(), grid)
    
function calculate_z★_diagnostics(b, grid, GPUgrid)

    arch = architecture(GPUgrid)

    z★ = []

    tmpz★ = CenterField(GPUgrid)
    for iter in eachindex(b)
       @info "time $iter of $(length(b))"
       push!(z★, CenterField(grid))
       z★_event = launch!(arch, GPUgrid, :xyz, _calculate_z★, tmpz★, arch_array(arch, b[iter].data), GPUgrid; dependencies = device_event(arch))
       wait(device(arch), z★_event)
       set!(z★[iter], Array(interior(tmpz★)))
    end
        
    return z★
end

@kernel function _calculate_z★(z★, b, grid)
    i, j, k = @index(Global, NTuple)
    
    Nx, Ny, Nz = size(grid)

    FT = eltype(z★)
    bl = b[i, j, k]
    A  = grid.Lx * grid.Ly

    z★[i, j, k] = 0.0
    @unroll for k′ in 1:Nz
        @show k′
        for j′ in 1:Ny, i′ in 1:Nx
                V = Vᶜᶜᶜ(i′, j′, k′, grid)
                z★[i, j, k] += V / A * FT(bl > b[i′, j′, k′])
        end
    end
end

function reorder_z★_and_b(z★, b)
    reordered_z★ = []
    reordered_b  = []

    Nx, Ny, Nz = size(b)
    for iter in eachindex(b)
        btmp = reshape(Array(interior(b[iter])),  Nx * Ny * Nz)
        ztmp = reshape(Array(interior(z★[iter])), Nx * Ny * Nz)
        perm = sortperm(ztmp)
        push!(reordered_b , btmp[perm])
        push!(reordered_z★, ztmp[perm])
    end

    return reordered_b, reordered_z★
end

function all_diagnostics(z★, b, z★_arr, b_arr, grid)

    Γ² = []
    Γ³ = []
    εᴿ = []

    for iter in eachindex(b)
        push!(Γ³, compute!(Field(b[iter] * z★[iter])))
        push!(Γ², CenterField(grid))

        event = _calculate_Γ²(Γ²[iter], z★[iter], z★_arr[iter], b_arr[iter], grid)
        wait(event)

        push!(εᴿ, compute!(Field(Γ²[iter] + Γ³[iter])))
    end

    return Γ², Γ³, εᴿ
end

@kernel function _calculate_Γ²(Γ², z★, z★_arr, b_arr, grid)
    i, j, k = @index(Global, NTuple)

    Nint = 10.0
     
    Γ²[i, j, k] = 0.0
        
    z_local  = znode(Center(), k, grid)
    z★_local = z★[i, j, k] 
    Δz       = (z_local - z★_local) / Nint
    zrange   = z★_local:Δz:z

    @unroll for z in zrange
        Γ²[i, j, k] += Δz * linear_interpolate(z★_arr, b_arr, z)
    end
end

@inline function linear_interpolate(x, y, x₀)
    i₁, i₂ = index_binary_search(x, xₒ, length(x))

    @inbounds y₂ = y[i₂]
    @inbounds y₁ = y[i₁]

    @inbounds x₂ = x[i₂]
    @inbounds x₁ = x[i₁]

    if x₁ == x₂
        return y₁
    else
        return (y₂ - y₁) / (x₂ - x₁) * (x₀ - x₁) + y₁
    end
end