using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils
using Oceananigans.Operators
using Oceananigans.OutputReaders: FieldDataset
using Oceananigans.Grids: on_architecture
using Oceananigans.BoundaryConditions
using Oceananigans.AbstractOperations: GridMetricOperation, Az
using Oceananigans.Models.HydrostaticFreeSurfaceModels: GeneralizedSpacingGrid, ZStar
using Oceananigans.Advection: _advective_tracer_flux_x, _advective_tracer_flux_y, _advective_tracer_flux_z
using KernelAbstractions: @kernel, @index
using Statistics: mean

const Lx = 1000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]

# number of grid points
const Nx = 200
const Ny = 400
const Nz = 40

@inline exponential_profile(z; Lz, h) = (exp(z / h) - exp( - Lz / h)) / (1 - exp( - Lz / h)) 

function exponential_z_faces(Nz, Depth; h = Nz / 4.5)

    z_faces = exponential_profile.((1:Nz+1); Lz = Nz, h)

    # Normalize
    z_faces .-= z_faces[1]
    z_faces .*= - Depth / z_faces[end]
    
    z_faces[1] = 0.0

    return reverse(z_faces)
end

grid = RectilinearGrid(GPU(),
                       topology = (Periodic, Bounded, Bounded),
                       size = (Nx, Ny, Nz),
                       halo = (6, 6, 6),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = exponential_z_faces(Nz, 3000))

cpu_grid = on_architecture(CPU(), grid)

function tracer_conservation(grid)
    b  = FieldTimeSeries("abernathey_channel_snapshots.jld2", "b";   backend = OnDisk())
    sⁿ  = FieldTimeSeries("abernathey_channel_snapshots.jld2", "sⁿ"; backend = OnDisk())
    AzC = GridMetricOperation((Center, Center, Center), Az, grid)
    AzC = compute!(Field(AzC))

    ΔzC = CenterField(grid)

    bT = Float64[]

    for i in 1:2:length(b)
        @info "doing iteration $i"
        launch!(CPU(), grid, :xyz, _calculate_Δz, ΔzC, grid, sⁿ[i])        
        push!(bT, sum(interior(b[i]) .* interior(ΔzC) .* interior(AzC)))
    end

    return bT
end

@kernel function _calculate_ΔzC(ΔzC, grid, sⁿ)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ΔzC[i, j, k] = grid.Δzᵃᵃᶜ[k] * sⁿ[i, j, 1] 
    end
end

function average_dissipation()
    dataset = FieldDataset("dissipation.jld2"; architecture = CPU(), backend = OnDisk()).fields

    Nt = length(dataset["χᵁ"])
    χᵁ = mean(parent(dataset["χᵁ"][1]),  dims = (1, 4)) ./ Nt
    χⱽ = mean(parent(dataset["χⱽ"][1]),  dims = (1, 4)) ./ Nt
    χᵂ = mean(parent(dataset["χᵂ"][1]),  dims = (1, 4)) ./ Nt
    bx = mean(parent(dataset["∂xb"][1]), dims = (1, 4)) ./ Nt
    by = mean(parent(dataset["∂yb"][1]), dims = (1, 4)) ./ Nt
    bz = mean(parent(dataset["∂zb"][1]), dims = (1, 4)) ./ Nt

    for i in 2:Nt
        @info "doing iteration $i"
        χᵁ .+= mean(parent(dataset["χᵁ"][i]),  dims = (1, 4)) ./ Nt
        χⱽ .+= mean(parent(dataset["χⱽ"][i]),  dims = (1, 4)) ./ Nt
        χᵂ .+= mean(parent(dataset["χᵂ"][i]),  dims = (1, 4)) ./ Nt
        bx .+= mean(parent(dataset["∂xb"][i]), dims = (1, 4)) ./ Nt
        by .+= mean(parent(dataset["∂yb"][i]), dims = (1, 4)) ./ Nt
        bz .+= mean(parent(dataset["∂zb"][i]), dims = (1, 4)) ./ Nt
    end

    return (; χᵁ, χⱽ, χᵂ, bx, by, bz)
end

function precompute_dissipation(grid, cpu_grid)
    dataset = FieldDataset("abernathey_channel_snapshots.jld2"; architecture = CPU(), backend = OnDisk()).fields

    grid = GeneralizedSpacingGrid(grid, ZStar())

    advection = WENO(grid; order = 7)

    χ = implicit_dissipation(dataset, grid, cpu_grid, advection)

    return χ
end

function implicit_dissipation(dataset, grid, cpu_grid, advection)
    times    = dataset["b"].times
    newtimes = times[1:2:end]
    χᵁ = FieldTimeSeries{Face, Center, Center}(cpu_grid, newtimes; backend = OnDisk(), path = "dissipation.jld2", name = "χᵁ")
    χⱽ = FieldTimeSeries{Center, Face, Center}(cpu_grid, newtimes; backend = OnDisk(), path = "dissipation.jld2", name = "χⱽ")
    χᵂ = FieldTimeSeries{Center, Center, Face}(cpu_grid, newtimes; backend = OnDisk(), path = "dissipation.jld2", name = "χᵂ")
    
    ∂xb = FieldTimeSeries{Face, Center, Center}(cpu_grid, newtimes; backend = OnDisk(), path = "dissipation.jld2", name = "∂xb")
    ∂yb = FieldTimeSeries{Center, Face, Center}(cpu_grid, newtimes; backend = OnDisk(), path = "dissipation.jld2", name = "∂yb")
    ∂zb = FieldTimeSeries{Center, Center, Face}(cpu_grid, newtimes; backend = OnDisk(), path = "dissipation.jld2", name = "∂zb")

    u    = XFaceField(grid)
    v    = YFaceField(grid)
    w    = ZFaceField(grid)
    bⁿ   = CenterField(grid)
    bⁿ⁻¹ = CenterField(grid)

    Nt = length(χᵁ)

    Xu = XFaceField(grid)
    Xv = YFaceField(grid)
    Xw = ZFaceField(grid)

    bx = XFaceField(grid)
    by = YFaceField(grid)
    bz = ZFaceField(grid)
        
    for i in 1:Nt
        @info "doing $i"

        io⁻ = i * 2 - 1
        io⁺ = i * 2   

        # Adapting the grid
        set!(grid.Δzᵃᵃᶠ.sⁿ, dataset["sⁿ"][io⁻])
        fill_halo_regions!(grid.Δzᵃᵃᶠ.Δ)

        # Extracting the velocities
        set!(u, dataset["u"][io⁻])
        set!(v, dataset["v"][io⁻])
        set!(w, dataset["w"][io⁻])

        set!(bⁿ⁻¹, dataset["b"][io⁻])
        set!(bⁿ  , dataset["b"][io⁺])

        fill_halo_regions!((bⁿ⁻¹, bⁿ, u, v, w))

        @info "starting the launch!!"
        launch!(GPU(), grid, :xyz, _compute_dissipation!, Xu, Xv, Xw, grid, advection, u, v, w, bⁿ, bⁿ⁻¹)
        launch!(GPU(), grid, :xyz, _compute_gradient!,    bx, by, bz, grid, bⁿ⁻¹)
        set!(χᵁ, Xu, i)
        set!(χⱽ, Xv, i)
        set!(χᵂ, Xw, i)

        set!(∂xb, bx, i)
        set!(∂yb, by, i)
        set!(∂zb, bz, i)
    end

    return (; χᵁ, χⱽ, χᵂ)
end

@kernel function _compute_gradient!(bx, by, bz, grid, b)
    i, j, k = @index(Global, NTuple)
    
    @inbounds begin
        bx[i, j, k] = ∂xᶠᶜᶜ(i, j, k, grid, b)
        by[i, j, k] = ∂yᶜᶠᶜ(i, j, k, grid, b)
        bz[i, j, k] = ∂zᶜᶜᶠ(i, j, k, grid, b)
    end
end

@kernel function _compute_dissipation!(χᵁ, χⱽ, χᵂ, grid, advection, uⁿ⁻¹, vⁿ⁻¹, wⁿ⁻¹, b, bⁿ⁻¹)
    i, j, k = @index(Global, NTuple)

    @inbounds χᵁ[i, j, k] = compute_χᵁ(i, j, k, grid, advection, uⁿ⁻¹, b, bⁿ⁻¹)
    @inbounds χⱽ[i, j, k] = compute_χⱽ(i, j, k, grid, advection, vⁿ⁻¹, b, bⁿ⁻¹)
    @inbounds χᵂ[i, j, k] = compute_χᵂ(i, j, k, grid, advection, wⁿ⁻¹, b, bⁿ⁻¹)
end

@inline b★(i, j, k, grid, bⁿ, bⁿ⁻¹) = @inbounds (bⁿ[i, j, k] + bⁿ⁻¹[i, j, k]) / 2
@inline b²(i, j, k, grid, b₁, b₂)   = @inbounds (b₁[i, j, k] * b₂[i, j, k])

@inline function compute_χᵁ(i, j, k, grid, advection, U, bⁿ, bⁿ⁻¹)
   
    δˣb★ = δxᶠᶜᶜ(i, j, k, grid, b★, bⁿ, bⁿ⁻¹)
    δˣb² = δxᶠᶜᶜ(i, j, k, grid, b², bⁿ, bⁿ⁻¹)

    𝒜x = _advective_tracer_flux_x(i, j, k, grid, advection, U, bⁿ⁻¹)
    𝒟x = @inbounds Axᶠᶜᶜ(i, j, k, grid) * U[i, j, k] * δˣb²

    return (𝒜x * 2 * δˣb★ - 𝒟x) / Vᶠᶜᶜ(i, j, k, grid)
end

@inline function compute_χⱽ(i, j, k, grid, advection, V, bⁿ, bⁿ⁻¹)
   
    δʸb★ = δyᶜᶠᶜ(i, j, k, grid, b★, bⁿ, bⁿ⁻¹)
    δʸb² = δyᶜᶠᶜ(i, j, k, grid, b², bⁿ, bⁿ⁻¹)

    𝒜y = _advective_tracer_flux_y(i, j, k, grid, advection, V, bⁿ⁻¹)
    𝒟y = @inbounds Ayᶜᶠᶜ(i, j, k, grid) * V[i, j, k] * δʸb²

    return (𝒜y * 2 * δʸb★ - 𝒟y) / Vᶜᶠᶜ(i, j, k, grid)
end

@inline function compute_χᵂ(i, j, k, grid, advection, W, bⁿ, bⁿ⁻¹)
   
    δᶻb★ = δzᶜᶜᶠ(i, j, k, grid, b★, bⁿ, bⁿ⁻¹)
    δᶻb² = δzᶜᶜᶠ(i, j, k, grid, b², bⁿ, bⁿ⁻¹)

    𝒜z = _advective_tracer_flux_z(i, j, k, grid, advection, W, bⁿ⁻¹)
    𝒟z = @inbounds Azᶜᶜᶠ(i, j, k, grid) * W[i, j, k] * δᶻb²

    return (𝒜z * 2 * δᶻb★ - 𝒟z) / Vᶜᶜᶠ(i, j, k, grid)
end

