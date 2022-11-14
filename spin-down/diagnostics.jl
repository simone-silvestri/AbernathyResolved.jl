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

function load_prognostic_fields(file, grid; var = "b")
    f = Field[]
    
    if var == "u"
        F = XFaceField
    elseif var == "v"
        F = YFaceField
    elseif var == "w"
        F = ZFaceField
    else
        F = CenterField
    end

    for iter in keys(file["timeseries/t"])
        @info "time $iter of $(keys(file["timeseries/t"])[end])"
        push!(f, set!(F(grid), file["timeseries/" * var * "/" * iter]))
    end

    return f
end
   
using Oceananigans.AbstractOperations: GridMetricOperation, volume

vol = compute!(Field(GridMetricOperation((Center, Center, Center), volume, grid)))

function calculate_z★_diagnostics(b, vol, grid)

    arch = architecture(grid)

    z★ = []

    for iter in eachindex(b)
       perm = sortperm(Array(interior(b[iter]))[:])
       sorted_b_field = (Array(interior(b[iter]))[:])[perm]
       sorted_v_field = (Array(interior(vol))[:])[perm]
       integrated_v   = cumsum(sorted_v_field)    

       @info "time $iter of $(length(b))"
       push!(z★, CenterField(grid))
       wall_clock = [time_ns()]
       z★_event = launch!(arch, grid, :xyz, _calculate_z★, z★[iter], b[iter], sorted_b_field, integrated_v, grid; dependencies = device_event(arch))
       wait(device(arch), z★_event)
       @info " wall time $(1e-9 * (time_ns() - wall_clock[1]))"
    end
        
    return z★
end

@kernel function _calculate_z★(z★, b, b_sorted, v_integrated, grid)
    i, j, k = @index(Global, NTuple)
    bl  = b[i, j, k]
    A   = grid.Lx * grid.Ly
    i₁  = searchsortedfirst(b_sorted, bl)
    z★[i, j, k] = v_integrated[i₁] / A
end

function reorder_z★_and_b(z★, b)
    reordered_z★ = []
    reordered_b  = []

    for iter in eachindex(b)
        perm           = sortperm(Array(interior(z★[iter]))[:])
        sorted_b_field = (Array(interior(b[iter]))[:])[perm]
        sorted_z_field = (Array(interior(z★[iter]))[:])[perm]
        @info "time $iter of $(length(b))"
        push!(reordered_b , sorted_b_field)
        push!(reordered_z★, sorted_z_field)
    end

    return reordered_b, reordered_z★
end

function all_diagnostics(z★, b, z★_arr, b_arr, grid)

    Γ² = []
    Γ³ = []
    εᴿ = []
    RPE = []

    for iter in eachindex(b)
        push!(Γ³, compute!(Field(b[iter] * z★[iter])))
        push!(Γ², CenterField(grid))

        event = _calculate_Γ²(Γ²[iter], z★[iter], z★_arr[iter], b_arr[iter], grid)
        wait(event)

        push!(εᴿ, compute!(Field(Γ²[iter] + Γ³[iter])))
        push!(RPE, compute!(Field(Integral(εᴿ)))[1, 1, 1])
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