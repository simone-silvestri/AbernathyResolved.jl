using Oceananigans
using Oceananigans.Utils
using Oceananigans.Units
using Oceananigans.Operators
using Oceananigans.Grids: architecture, znode, halo_size, on_architecture
using Oceananigans.Architectures: device, device_event, arch_array
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.AbstractOperations: GridMetricOperation, volume


using JLD2

include("utils.jl")

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

function load_prognostic_fields(file, grid; var = "b", iterations = :all, skip = 1)
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

    for (idx, iter) in enumerate(keys(file["timeseries/t"])[1:skip:end])
        @info "time $iter of $(keys(file["timeseries/t"])[end])"
        push!(f, set!(F(grid), file["timeseries/" * var * "/" * iter]))

        if idx == iterations
            break
        end
    end

    return f
end
   
function calculate_z★_diagnostics(b, vol, grid; iterations = :all)

    arch = architecture(grid)

    z★ = Field[]

    if iterations == :all
        iters = eachindex(b)
    else
        iters = 1:iterations
    end

    for iter in iters
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

function all_diagnostics(z★, b, grid; iterations = :all)

    arch = architecture(grid)

    Γ² = Field[]
    Γ³ = Field[]
    εᴿ = Field[]

    if iterations == :all
        iters = eachindex(b)
    else
        iters = 1:iterations
    end

    for iter in iters
        push!(Γ³, compute!(Field(b[iter] * z★[iter])))
        push!(Γ², CenterField(grid))

        perm   = sortperm(Array(interior(z★[iter]))[:])
        b_arr  = (Array(interior(b[iter]))[:])[perm]
        z★_arr = (Array(interior(z★[iter]))[:])[perm]
    
        @info "compute all diagnostics iteration $iter"
        Γ²_event = launch!(arch, grid, :xyz, _calculate_Γ², Γ²[iter], z★[iter], z★_arr, b_arr, grid; dependencies = device_event(arch))
        wait(device(arch), Γ²_event)
        
        push!(εᴿ, compute!(Field(Γ²[iter] + Γ³[iter])))
    end

    return Γ², Γ³, εᴿ
end

@kernel function _calculate_Γ²(Γ², z★, z★_arr, b_arr, grid)
    i, j, k = @index(Global, NTuple)

    Nint = 10.0
     
    Γ²[i, j, k] = 0.0
         
    z_local  = znode(Center(), k, grid) + grid.Lz
    z★_local = z★[i, j, k] 
    Δz       = (z_local - z★_local) / Nint
    zrange   = z★_local:Δz:z_local

    @unroll for z in zrange
        Γ²[i, j, k] += Δz * linear_interpolate(z★_arr, b_arr, z)
    end
end

@inline function linear_interpolate(x, y, x₀)
    i₁ = searchsortedfirst(x, x₀)
    i₂ =  searchsortedlast(x, x₀)

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

function calc_all_diagnostics(file, grid; iterations = :all, skip = 1)
    vol = compute!(Field(GridMetricOperation((Center, Center, Center), volume, grid)))
    b   = load_prognostic_fields(file, grid; var = "b", iterations, skip)

    @info "loaded b field"
    z★ = calculate_z★_diagnostics(b, vol, grid; iterations) 

    @info "calculated z★"
    Γ², Γ³, εᴿ = all_diagnostics(z★, b, grid; iterations)

    @info "calculating RPE"
    RPE = Float64[]
    
    for i in eachindex(εᴿ)
        @info "computing RPE at $i"
        push!(RPE, sum(compute!(Field(εᴿ[i] * vol))))
    end

    return (; b, z★, Γ², Γ³, εᴿ, RPE)
end

function calc_vort(u, v)
    
    ζ = Field[]

    for i in eachindex(u)
        @info "calculating vorticity at $i"
        ζ_op = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid; computed_dependencies = (u[i], v[i]))
        push!(ζ, compute!(Field(ζ_op)))
    end

    return ζ
end

function calc_spectra(u)

    spectra = Vector(undef, length(u))
    for i in eachindex(u)
        @info "calculating spectra at $i"
        spectra[i] = power_spectrum_1d_x(interior(u[i])[:, :, end], xnodes(u[i]), ynodes(u[i]))
    end

    return spectra
end


function interiors(fieldvec::Vector)
    inter = []

    for i in eachindex(fieldvec)
        push!(inter, interior(fieldvec[i]))
    end

    return inter
end

function only_the_interior(outputs)

    b  = interiors(outputs.b)
    z★ = interiors(outputs.z★)
    Γ² = interiors(outputs.Γ²)
    Γ³ = interiors(outputs.Γ³)
    εᴿ = interiors(outputs.εᴿ)

    return (; b, z★, Γ², Γ³, εᴿ)
end

# file = jldopen("abernathey_channel_fields_fluxform_weno.jld2")
# weno = calc_all_diagnostics(file, grid; skip = 4)
# 
# jldsave("weno-all-skip-3.jld2", output = weno)
# 
# weno = nothing
# GC.gc()

# file   = jldopen("abernathey_channel_fields_centered_momentum.jld2")
# center = calc_all_diagnostics(file, grid; skip = 4)

# jldsave("center-all-skip-3.jld2", output = center)
