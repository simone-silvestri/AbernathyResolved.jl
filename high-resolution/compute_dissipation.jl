using Oceananigans.Operators
import Oceananigans.Advection: _advective_tracer_flux_x, _advective_tracer_flux_y, _advective_tracer_flux_z 

_advective_tracer_flux_x(i, j, k, grid, advection::TracerAdvection, args...) = 
    _advective_tracer_flux_x(i, j, k, grid, advection.x, args...)

_advective_tracer_flux_y(i, j, k, grid, advection::TracerAdvection, args...) = 
    _advective_tracer_flux_y(i, j, k, grid, advection.y, args...)

_advective_tracer_flux_z(i, j, k, grid, advection::TracerAdvection, args...) = 
    _advective_tracer_flux_z(i, j, k, grid, advection.z, args...)

function update_velocities(simulation)
    uⁿ⁻¹ = simulation.model.auxiliary_fields.uⁿ⁻¹
    vⁿ⁻¹ = simulation.model.auxiliary_fields.vⁿ⁻¹
    wⁿ⁻¹ = simulation.model.auxiliary_fields.wⁿ⁻¹
    bⁿ⁻¹ = simulation.model.auxiliary_fields.bⁿ⁻¹

    u = simulation.model.velocities.u
    v = simulation.model.velocities.v
    w = simulation.model.velocities.w
    b = simulation.model.tracers.b

    parent(uⁿ⁻¹) .= parent(u)
    parent(vⁿ⁻¹) .= parent(v)
    parent(wⁿ⁻¹) .= parent(w)
    parent(bⁿ⁻¹) .= parent(b)

    return nothing
end

function compute_χ_values(simulation)
    model = simulation.model
    advection = model.advection.b
    grid = model.grid
    arch = architecture(grid)

    b = model.tracers.b
    bⁿ⁻¹ = model.auxiliary_fields.bⁿ⁻¹
    uⁿ⁻¹ = model.auxiliary_fields.uⁿ⁻¹
    vⁿ⁻¹ = model.auxiliary_fields.vⁿ⁻¹
    wⁿ⁻¹ = model.auxiliary_fields.wⁿ⁻¹
    χu   = model.auxiliary_fields.χu
    χv   = model.auxiliary_fields.χv
    χw   = model.auxiliary_fields.χw
    ∂xb² = model.auxiliary_fields.∂xb²
    ∂yb² = model.auxiliary_fields.∂yb²
    ∂zb² = model.auxiliary_fields.∂zb²

    launch!(arch, grid, :xyz, _compute_dissipation!, χu, χv, χw, ∂xb², ∂yb², ∂zb², uⁿ⁻¹, vⁿ⁻¹, wⁿ⁻¹, grid, advection, b, bⁿ⁻¹)

    return nothing
end

@kernel function _compute_dissipation!(χu, χv, χw, ∂xb², ∂yb², ∂zb², uⁿ⁻¹, vⁿ⁻¹, wⁿ⁻¹, grid, advection, b, bⁿ⁻¹)
    i, j, k = @index(Global, NTuple)

    @inbounds χu[i, j, k] = compute_χᵁ(i, j, k, grid, advection, uⁿ⁻¹, b, bⁿ⁻¹)
    @inbounds χv[i, j, k] = compute_χⱽ(i, j, k, grid, advection, vⁿ⁻¹, b, bⁿ⁻¹)
    @inbounds χw[i, j, k] = compute_χᵂ(i, j, k, grid, advection, wⁿ⁻¹, b, bⁿ⁻¹)

    @inbounds ∂xb²[i, j, k] = Axᶠᶜᶜ(i, j, k, grid) * δxᶠᶜᶜ(i, j, k, grid, bⁿ⁻¹)^2 / Δxᶠᶜᶜ(i, j, k, grid)
    @inbounds ∂yb²[i, j, k] = Ayᶜᶠᶜ(i, j, k, grid) * δyᶜᶠᶜ(i, j, k, grid, bⁿ⁻¹)^2 / Δyᶜᶠᶜ(i, j, k, grid)
    @inbounds ∂zb²[i, j, k] = Azᶜᶜᶠ(i, j, k, grid) * δzᶜᶜᶠ(i, j, k, grid, bⁿ⁻¹)^2 / Δzᶜᶜᶠ(i, j, k, grid)
end

@inline b★(i, j, k, grid, bⁿ, bⁿ⁻¹) = @inbounds (bⁿ[i, j, k] + bⁿ⁻¹[i, j, k]) / 2
@inline b²(i, j, k, grid, b₁, b₂)   = @inbounds (b₁[i, j, k] * b₂[i, j, k])

@inline function compute_χᵁ(i, j, k, grid, advection, U, bⁿ, bⁿ⁻¹)
   
    δˣb★ = δxᶠᶜᶜ(i, j, k, grid, b★, bⁿ, bⁿ⁻¹)
    δˣb² = δxᶠᶜᶜ(i, j, k, grid, b², bⁿ, bⁿ⁻¹)

    𝒜x = _advective_tracer_flux_x(i, j, k, grid, advection, U, bⁿ⁻¹)
    𝒟x = @inbounds Axᶠᶜᶜ(i, j, k, grid) * U[i, j, k] * δˣb²

    return 𝒜x * 2 * δˣb★ - 𝒟x
end

@inline function compute_χⱽ(i, j, k, grid, advection, V, bⁿ, bⁿ⁻¹)
   
    δʸb★ = δyᶜᶠᶜ(i, j, k, grid, b★, bⁿ, bⁿ⁻¹)
    δʸb² = δyᶜᶠᶜ(i, j, k, grid, b², bⁿ, bⁿ⁻¹)

    𝒜y = _advective_tracer_flux_y(i, j, k, grid, advection, V, bⁿ⁻¹)
    𝒟y = @inbounds Ayᶜᶠᶜ(i, j, k, grid) * V[i, j, k] * δʸb²

    return 𝒜y * 2 * δʸb★ - 𝒟y
end

@inline function compute_χᵂ(i, j, k, grid, advection, W, bⁿ, bⁿ⁻¹)
   
    δᶻb★ = δzᶜᶜᶠ(i, j, k, grid, b★, bⁿ, bⁿ⁻¹)
    δᶻb² = δzᶜᶜᶠ(i, j, k, grid, b², bⁿ, bⁿ⁻¹)

    𝒜z = _advective_tracer_flux_z(i, j, k, grid, advection, W, bⁿ⁻¹)
    𝒟z = @inbounds Azᶜᶜᶠ(i, j, k, grid) * W[i, j, k] * δᶻb²

    return 𝒜z * 2 * δᶻb★ - 𝒟z
end

