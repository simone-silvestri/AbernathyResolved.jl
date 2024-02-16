ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, @__DIR__)

using Printf
using Statistics
using CUDA 

using Oceananigans
using Oceananigans.Units
using Oceananigans.Advection: VelocityStencil
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.Operators
using Oceananigans.TurbulenceClosures
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ZStar, ZStarSpacingGrid

const Lx = 1000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]

# Architecture
arch = GPU()

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
    y_shutoff = 5 / 6 *  grid.Ly,   # shutoff location for buoyancy flux [m]
    τ = 0.1 / ρ,                    # surface kinematic wind stress [m² s⁻²]
    μ = 1 / 30days,                 # bottom drag damping time-scale [s⁻¹]
    ΔB = 8 * α * g,                 # surface vertical buoyancy gradient [s⁻²]
    H =  grid.Lz,                   # domain depth [m]
    h = 1000.0,                     # exponential decay scale of stable stratification [m]
    y_sponge = 19 / 20 *  grid.Ly,  # southern boundary of sponge layer [m]
    λt = 7.0days                    # relaxation time scale [s]
)

@inline function buoyancy_flux(i, j, grid, clock, model_fields, p)
    y = ynode(j, grid, Center())
    return ifelse(y < p.y_shutoff, p.Qᵇ * cos(3π * y / p.Ly), 0.0)
end

buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, discrete_form = true, parameters = parameters)

@inline function u_stress(i, j, grid, clock, model_fields, p)
    y = ynode(j, grid, Center())
    return -p.τ * sin(π * y / p.Ly)
end

u_stress_bc = FluxBoundaryCondition(u_stress, discrete_form = true, parameters = parameters)

@inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * p.Lz * model_fields.u[i, j, 1]
@inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * p.Lz * model_fields.v[i, j, 1]

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

@inline initial_buoyancy(z, p) = p.ΔB * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))
@inline mask(y, p) = max(0.0, y - p.y_sponge) / (Ly - p.y_sponge)

@inline function buoyancy_relaxation(i, j, k, grid, clock, model_fields, p)
    timescale = p.λt
    y = ynode(j, grid, Center())
    z = znode(k, grid, Center())
    target_b = initial_buoyancy(z, p)
    b = @inbounds model_fields.b[i, j, k]

    return -1 / timescale * mask(y, p) * (b - target_b)
end

Fb = Forcing(buoyancy_relaxation, discrete_form = true, parameters = parameters)

# closure

κz = 1e-7   # [m²/s] vertical diffusivity
νz = 1e-5   # [m²/s] vertical viscosity

vertical_closure      = VerticalScalarDiffusivity(ν = νz, κ = κz)
convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0)

#####
##### Model building
#####

@info "Building a model..."

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    free_surface = SplitExplicitFreeSurface(; cfl = 0.75, grid),
                                    momentum_advection = WENO(grid; order = 7),
                                    tracer_advection   = WENO(grid; order = 7),
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = coriolis,
                                    generalized_vertical_coordinate = ZStar(),
                                    closure = (convective_adjustment, vertical_closure),
                                    tracers = :b,
                                    boundary_conditions = (b = b_bcs, u = u_bcs, v = v_bcs),
                                    forcing = (; b = Fb))

@info "Built $model."

#####
##### Initial conditions
#####

# resting initial condition
ε(σ) = σ * randn()
bᵢ(x, y, z) = parameters.ΔB * (exp(z / parameters.h) - exp(-grid.Lz / parameters.h)) / (1 - exp(-grid.Lz / parameters.h)) + ε(1e-8)

set!(model, b = bᵢ)

#####
##### Simulation building
#####

Δt₀ = 1minutes
stop_time = 360000days

simulation = Simulation(model, Δt = Δt₀, stop_time = stop_time)

# add timestep wizard callback
wizard = TimeStepWizard(cfl=0.25, max_change=1.1, max_Δt=20minutes)
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

#####
##### Build checkpointer and output writer
#####

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(100days),
                                                        prefix = "abernathey_channel",
                                                        overwrite_existing = true)

# simulation.output_writers[:averages] = JLD2OutputWriter(model, averaged_outputs,
#                                                         schedule = AveragedTimeInterval(1days, window = 1days, stride = 1),
#                                                         filename = "abernathey_channel_averages",
#                                                         verbose = true,
#                                                         overwrite_existing = true)

@info "Running the simulation..."

try
    run!(simulation, pickup = false)
catch err
    @info "run! threw an error! The error message is"
    showerror(stdout, err)
end