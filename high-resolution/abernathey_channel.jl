ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, @__DIR__)

using Printf
using Statistics
using CUDA 

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity
using Oceananigans.TurbulenceClosures: FivePointHorizontalFilter

using Oceananigans
using Oceananigans.Units
using Oceananigans.Advection: VelocityStencil
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.Operators
using Oceananigans.TurbulenceClosures
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ZStar, ZStarSpacingGrid
using Oceananigans.Utils: ConsecutiveIterations

const Lx = 1000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]

# Architecture
arch = GPU()

years  = 365days
pickup = false

# number of grid points
Nx = 200
Ny = 400
Nz = 40

"""
    function exponential_z_faces(; Nz = 69, Lz = 4000.0, e_folding = 0.06704463421863584)

generates an array of exponential z faces 

"""
@inline exponential_profile(z; Lz, h) = (exp(z / h) - exp( - Lz / h)) / (1 - exp( - Lz / h)) 

function exponential_z_faces(Nz, Depth; h = Nz / 4.5)

    z_faces = exponential_profile.((1:Nz+1); Lz = Nz, h)

    # Normalize
    z_faces .-= z_faces[1]
    z_faces .*= - Depth / z_faces[end]
    
    z_faces[1] = 0.0

    return reverse(z_faces)
end

grid = RectilinearGrid(arch,
                       topology = (Periodic, Bounded, Bounded),
                       size = (Nx, Ny, Nz),
                       halo = (6, 6, 6),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = exponential_z_faces(Nz, 3000))

@info "Built a grid: $grid."

#####
##### Boundary conditions
#####

α  = 2e-4     # [K⁻¹] thermal expansion coefficient 
g  = 9.8061   # [m/s²] gravitational constant
cᵖ = 3994.0   # [J/K]  heat capacity
ρ  = 999.8    # [kg/m³] reference density

parameters = (
    Ly = grid.Ly,
    Lz = grid.Lz,
    Qᵇ = 10 / (ρ * cᵖ) * α * g,     # buoyancy flux magnitude [m² s⁻³]    
    y_shutoff = 9 / 10 *  grid.Ly,  # shutoff location for buoyancy flux [m] and start restoring
    τ = 0.1 / ρ,                    # surface kinematic wind stress [m² s⁻²]
    μ = 0.003,                      # bottom drag damping time-scale [s⁻¹]
    ΔB = 8 * α * g,                 # surface vertical buoyancy gradient [s⁻²]
    H =  grid.Lz,                   # domain depth [m]
    h = 1000.0,                     # exponential decay scale of stable stratification [m]
    λt = 7.0days                    # relaxation time scale [s]
)

@inline function buoyancy_flux(i, j, grid, clock, model_fields, p)
    y = ynode(j, grid, Center())
    Q = ifelse(y > p.y_shutoff, zero(grid), p.Qᵇ * sin(3π * y / p.y_shutoff))
    return Q
end

buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, discrete_form = true, parameters = parameters)

@inline initial_buoyancy(z, p) = p.ΔB * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))

@inline function buoyancy_relaxation(i, j, k, grid, clock, model_fields, p)
    timescale = p.λt
    z = znode(k, grid, Center())
    y = ynode(j, grid, Center())
    target_b = initial_buoyancy(z, p)
    b = @inbounds model_fields.b[i, grid.Ny, k]
    Lsponge = grid.Ly - p.y_shutoff
    mask = max(zero(grid), (y - p.y_shutoff) / Lsponge)

    return mask / timescale * (target_b - b)
end

buoyancy_restoring = Forcing(buoyancy_relaxation, discrete_form = true, parameters = parameters)

@inline function u_stress(i, j, grid, clock, model_fields, p)
    y = ynode(j, grid, Center())
    return -p.τ * sin(π * y / p.Ly)
end

u_stress_bc = FluxBoundaryCondition(u_stress, discrete_form = true, parameters = parameters)

@inline ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2

@inline speedᶠᶜᶜ(i, j, k, grid, U) = sqrt(ℑxyᶠᶜᵃ(i, j, k, grid, ϕ², U.v) + U.u[i, j, k]^2)
@inline speedᶜᶠᶜ(i, j, k, grid, U) = sqrt(ℑxyᶜᶠᵃ(i, j, k, grid, ϕ², U.u) + U.v[i, j, k]^2)

@inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * speedᶠᶜᶜ(i, j, 1, grid, model_fields) * model_fields.u[i, j, 1]
@inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * speedᶜᶠᶜ(i, j, 1, grid, model_fields) * model_fields.v[i, j, 1]

u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form = true, parameters = parameters)
v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form = true, parameters = parameters)

b_bcs = FieldBoundaryConditions(top = buoyancy_flux_bc)

u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
v_bcs = FieldBoundaryConditions(bottom = v_drag_bc)

#####
##### Coriolis
#####

const f = -1e-4
const β = 1e-11
coriolis = BetaPlane(f₀ = f, β = β)

#####
##### Forcing and initial condition
#####

# closure

κz = 1e-7   # [m²/s] vertical diffusivity
νz = 1e-5   # [m²/s] vertical viscosity

vertical_closure      = VerticalScalarDiffusivity(ν = νz, κ = κz)
convective_adjustment = RiBasedVerticalDiffusivity(; horizontal_Ri_filter =FivePointHorizontalFilter())

#####
##### Model building
#####

momentum_advection = WENO(; order = 7) #VectorInvariant(vorticity_scheme = WENO(; order = 9),
                              #       divergence_scheme = WENO(),
                               #        vertical_scheme = Centered())

@info "Building a model..."

tracer_advection = Oceananigans.Advection.TracerAdvection(WENO(; order = 7), WENO(; order = 7), Centered())

model = HydrostaticFreeSurfaceModel(; grid = grid,
                                    free_surface = SplitExplicitFreeSurface(grid; cfl = 0.7),
                                    momentum_advection,
                                    tracer_advection,
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = coriolis,
                                    generalized_vertical_coordinate = ZStar(),
                                    closure = (convective_adjustment, vertical_closure),
                                    tracers = :b,
                                    forcing = (; b = buoyancy_restoring),
                                    boundary_conditions = (b = b_bcs, u = u_bcs, v = v_bcs))

@info "Built $model."

#####
##### Initial conditions
#####

# resting initial condition
ε(σ) = σ * randn()
bᵢ(x, y, z) = parameters.ΔB * (exp(z / parameters.h) - exp(-grid.Lz / parameters.h)) / (1 - exp(-grid.Lz / parameters.h)) + ε(1e-8)

set!(model, b = bᵢ) #, e = 1e-6)

#####
##### Simulation building
#####

Δt₀ = 1minutes
stop_time = 100 * 360days

simulation = Simulation(model, Δt = Δt₀, stop_time = 10days)

# add timestep wizard callback
wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=15minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# add progress callback
wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u, w): (%6.3e, %6.3e) m/s, next Δt: %s\n",
        100 * (sim.model.clock.time / sim.stop_time),
        sim.model.clock.iteration,
        prettytime(sim.model.clock.time),
        prettytime(1e-9 * (time_ns() - wall_clock[1])),
        maximum(abs, sim.model.velocities.u),
        maximum(abs, sim.model.velocities.w),
        prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(20))

# run!(simulation)

simulation.stop_time = stop_time

wizard = TimeStepWizard(cfl=0.3, max_change=1.1, max_Δt=15minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

#####
##### Diagnostics
#####

u, v, w = model.velocities
b = model.tracers.b

ζ = Field(∂x(v) - ∂y(u))

B = Field(Average(b, dims = 1))
V = Field(Average(v, dims = 1))
W = Field(Average(w, dims = 1))

b′ = b - B
v′ = v - V
w′ = w - W

v′b′ = Field(Average(v′ * b′, dims = 1))
w′b′ = Field(Average(w′ * b′, dims = 1))

outputs = (; b, ζ, w)

averaged_outputs = (; v′b′, w′b′, B)

grid_variables = (; sⁿ = model.grid.Δzᵃᵃᶠ.sⁿ, ∂t_∂s = model.grid.Δzᵃᵃᶠ.∂t_∂s)
snapshot_outputs = merge(model.velocities, model.tracers)
snapshot_outputs = merge(snapshot_outputs, grid_variables)

#####
##### Build checkpointer and output writer
#####

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(100days),
                                                        prefix = "abernathey_channel",
                                                        overwrite_existing = true)

simulation.output_writers[:snapshots] = JLD2OutputWriter(model, snapshot_outputs, 
                                                         schedule = ConsecutiveIterations(TimeInterval(30days)),
                                                         filename = "abernathey_channel_snapshots",
                                                         verbose = true,
                                                         overwrite_existing = true)

simulation.output_writers[:averages] = JLD2OutputWriter(model, averaged_outputs, 
                                                        schedule = AveragedTimeInterval(5years, stride = 10),
                                                        filename = "abernathey_channel_averages",
                                                        verbose = true,
                                                        overwrite_existing = true)

@info "Running the simulation..."

try
    run!(simulation; pickup)
catch err
    @info "run! threw an error! The error message is"
    showerror(stdout, err)
end
