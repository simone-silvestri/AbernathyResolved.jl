pushfirst!(LOAD_PATH, @__DIR__)

using Printf
using Statistics
using CUDA 

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEMixingLength, CATKEVerticalDiffusivity
using Oceananigans.TurbulenceClosures: FivePointHorizontalFilter

using Oceananigans
using Oceananigans.Units
using Oceananigans.Advection: VelocityStencil
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.Operators
using Oceananigans.TurbulenceClosures
using Oceananigans.Advection: TracerAdvection
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ZStar, ZStarSpacingGrid, VelocityFields
using Oceananigans.Utils: ConsecutiveIterations
using KernelAbstractions: @kernel, @index

const Lx = 1000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]

# CUDA.device!(1)

include("compute_dissipation.jl")

# Architecture
arch = GPU()

pickup = false # "abernathey_channel_iteration181044.jld2"

# number of grid points
Nx = 200
Ny = 400
Nz = 90

# Thirty levels spacing
# Δz = [10,  10,  10,  12,  14,  18,
#       20,  23,  27,  31,  35,  40,
#       46,  53,  60,  68,  76,  86,
#       98,  110, 125, 141, 159, 180,
#       203, 230, 260, 280, 280, 280]
      
# Ninty levels spacing
Δz = [10.0 * ones(6)...,
      11.25, 12.625, 14.125, 15.8125, 17.75, 19.9375, 22.375, 25.125, 28.125, 31.625, 35.5, 39.75,
      42.0 * ones(56)...,
      39.75, 35.5, 31.625, 28.125, 25.125, 22.375, 19.9375, 17.75, 15.8125, 14.125, 12.625, 11.25,
      10.0 * ones(4)...]

z_faces = zeros(Nz+1)
for k in Nz : -1 : 1
    z_faces[k] = z_faces[k+1] - Δz[Nz - k + 1]
end

grid = RectilinearGrid(arch,
                       topology = (Periodic, Bounded, Bounded),
                       size = (Nx, Ny, Nz),
                       halo = (6, 6, 6),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = z_faces)

const Lz = grid.Lz

@info "Built a grid: $grid."

#####
##### Boundary conditions
#####

α  = 2e-4     # [K⁻¹] thermal expansion coefficient 
g  = 9.8061   # [m/s²] gravitational constant
cᵖ = 3994.0   # [J/K]  heat capacity
ρ  = 999.8    # [kg/m³] reference density

parameters = (
    Ly  = grid.Ly,
    Lz  = grid.Lz,
    Δy  = grid.Δyᵃᶜᵃ,
    Qᵇ  = 10 / (ρ * cᵖ) * α * g, # buoyancy flux magnitude [m² s⁻³]    
    y_shutoff = 5 / 6 * grid.Ly, # shutoff location for buoyancy flux [m] 
    τ  = 0.1 / ρ,                # surface kinematic wind stress [m² s⁻²]
    μ  = 1.102e-3,               # bottom drag damping time-scale [ms⁻¹]
    Lsponge = 9 / 10 * Ly,       # sponge region for buoyancy restoring [m]
    ν  = 3e-4,                   # viscosity for "no-slip" lateral boundary conditions
    ΔB = 8 * α * g,              # surface vertical buoyancy gradient [s⁻²]
    H  =  grid.Lz,               # domain depth [m]
    h  = 1000.0,                 # exponential decay scale of stable stratification [m]
    λt = 7.0days                 # relaxation time scale [s]
)


# Initial condition from MITgcm
Tinit = Array{Float64}(undef, Nx*Ny*Nz)
read!("tIni_80y_90L.bin", Tinit)
Tinit = bswap.(Tinit) |> Array{Float64}
Tinit = reshape(Tinit, Nx, Ny, Nz)
binit = reverse(Tinit, dims = 3) .* α .* g

@inline function buoyancy_flux(i, j, grid, clock, model_fields, p)
    y = ynode(j, grid, Center())
    Q = ifelse(y > p.y_shutoff, zero(grid), p.Qᵇ * cos(3π * y / p.Ly))
    return Q
end

buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, discrete_form = true, parameters = parameters)

@inline initial_buoyancy(z, p) = p.ΔB * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))

@inline mask(y, p) = max(0, (y - p.Ly + p.Lsponge) / p.Lsponge)

@inline function buoyancy_relaxation(i, j, k, grid, clock, model_fields, p)
    timescale = p.λt
    z = znode(k, grid, Center())
    y = ynode(j, grid, Center())

    target_b = initial_buoyancy(z, p)
    
    b = @inbounds model_fields.b[i, grid.Ny, k]

    return mask(y, p) / timescale * (target_b - b)
end

buoyancy_restoring = Forcing(buoyancy_relaxation; discrete_form = true, parameters)

@inline function u_stress(i, j, grid, clock, model_fields, p)
    y = ynode(j, grid, Center())
    return - p.τ * sin(π * y / p.Ly)
end

u_stress_bc = FluxBoundaryCondition(u_stress; discrete_form = true, parameters)

@inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds - p.μ * model_fields.u[i, j, 1]
@inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds - p.μ * model_fields.v[i, j, 1]

u_drag_bc = FluxBoundaryCondition(u_drag; discrete_form = true, parameters)
v_drag_bc = FluxBoundaryCondition(v_drag; discrete_form = true, parameters)

b_bcs = FieldBoundaryConditions(top = buoyancy_flux_bc)
u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
v_bcs = FieldBoundaryConditions(bottom = v_drag_bc)

#####
##### Coriolis
#####

coriolis = BetaPlane(f₀ = -1e-4, β = 1e-11)

#####
##### Forcing and initial condition
#####

# closure
mixing_length = CATKEMixingLength(; Cᵇ = 0.01)
closure = CATKEVerticalDiffusivity(; mixing_length)

closure = ConvectiveAdjustmentVerticalDiffusivity(background_κz = 1e-5,
						  convective_κz = 0.1,
					          background_νz = 1e-4,
						  convective_νz = 0.1)

#####
##### Model building
#####

momentum_advection = VectorInvariant(vertical_scheme   = WENO(),
                                     vorticity_scheme  = WENO(; order = 9),
                                     divergence_scheme = WENO())

@info "Building a model..."

tracer_advection = TracerAdvection(WENO(; order = 7), WENO(; order = 7), Centered())

free_surface = SplitExplicitFreeSurface(grid; substeps = 90)

bⁿ⁻¹ = CenterField(grid)
Uⁿ⁻¹ = VelocityFields(grid)
χ    = VelocityFields(grid)
∂b²  = VelocityFields(grid)

auxiliary_fields = (; bⁿ⁻¹, 
                      uⁿ⁻¹ = Uⁿ⁻¹.u,
                      vⁿ⁻¹ = Uⁿ⁻¹.v,
                      wⁿ⁻¹ = Uⁿ⁻¹.w,
                      χu   = χ.u,
                      χv   = χ.v,
                      χw   = χ.w,
                      ∂xb² = ∂b².u,
                      ∂yb² = ∂b².v,
                      ∂zb² = ∂b².w)

model = HydrostaticFreeSurfaceModel(; grid,
                                      free_surface,
                                      momentum_advection,
                                      tracer_advection,
                                      buoyancy = BuoyancyTracer(),
                                      coriolis,
                                      generalized_vertical_coordinate = ZStar(),
                                      closure,
                                      tracers = (:b, :e),
                                      forcing = (; b = buoyancy_restoring),
                                      auxiliary_fields,
                                      boundary_conditions = (b = b_bcs, u = u_bcs, v = v_bcs))

@info "Built $model."

#####
##### Initial conditions
#####

# resting initial condition
bᵢ(x, y, z) = parameters.ΔB * (exp(z / parameters.h) - exp(-grid.Lz / parameters.h)) / 
                              (1 - exp(-grid.Lz / parameters.h)) * (1 + cos(20π * x / Lx) / 100)

set!(model, b = binit, e = 1e-6) 

#####
##### Simulation building
#####

Δt₀ = 1minutes

simulation = Simulation(model; Δt = Δt₀, stop_time = 100days)

# add progress callback
wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u, w): (%6.3e, %6.3e) m/s, next Δt: %s\n",
        100 * (sim.model.clock.time / sim.stop_time),
        sim.model.clock.iteration,
        prettytime(sim.model.clock.time),
        prettytime(1e-9 * (time_ns() - wall_clock[1])),
        maximum(abs, interior(sim.model.velocities.u)),
        maximum(abs, interior(sim.model.velocities.w)),
        prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(20))

wizard = TimeStepWizard(cfl=0.3, max_change=1.1, max_Δt=5minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(360days),
                                                        prefix = "abernathey_channel",
                                                        overwrite_existing = true)

wizard = TimeStepWizard(cfl=0.2, max_change=1.1, max_Δt=6minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

#####
##### Diagnostics
#####

simulation.stop_time = 130 * 360days # Run for 500 years!

simulation.callbacks[:compute_diagnostics] = Callback(compute_χ_values,  IterationInterval(1))
simulation.callbacks[:update_velocities]   = Callback(update_velocities, IterationInterval(1))

u, v, w = model.velocities
b = model.tracers.b
outputs = (; u, v, w, b)

grid_variables = (; sⁿ = model.grid.Δzᵃᵃᶠ.sⁿ, ∂t_∂s = model.grid.Δzᵃᵃᶠ.∂t_∂s)
snapshot_outputs = merge(model.velocities,  model.tracers)
snapshot_outputs = merge(snapshot_outputs,  grid_variables, model.auxiliary_fields)
average_outputs  = merge(snapshots_outputs, model.auxiliary_fields)

#####
##### Build checkpointer and output writer
#####

simulation.output_writers[:snapshots] = JLD2OutputWriter(model, snapshot_outputs, 
                                                         schedule = ConsecutiveIterations(TimeInterval(180days)),
                                                         filename = "abernathey_channel_snapshots",
                                                         overwrite_existing = true)

simulation.output_writers[:averages] = JLD2OutputWriter(model, average_outputs, 
                                                        schedule = AveragedTimeInterval(10 * 360days, stride = 10),
                                                        filename = "abernathey_channel_averages",
                                                        overwrite_existing = true)

@info "Running the simulation..."

run!(simulation; pickup)


