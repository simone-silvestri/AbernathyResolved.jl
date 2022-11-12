using Pkg
Pkg.activate("/home/ssilvest/stable_oceananigans/Oceananigans.jl/")
using Oceananigans
ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, @__DIR__)

using Printf
using Statistics
using CUDA 

using Oceananigans
using Oceananigans.Units
using Oceananigans.Advection: VelocityStencil, VorticityStencil
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.Operators
using Oceananigans.TurbulenceClosures
using JLD2

const Lx = 1000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]

CUDA.device!(1)

# Architecture
arch = GPU()

# number of grid points
Nx = 400
Ny = 800
Nz = 30

# stretched grid 
k_center = collect(1:Nz)
Δz_center = @. 10 * 1.125^(Nz - k_center)

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

@info "Built a grid: $grid."

#####
##### Boundary conditions
#####

α  = 2e-4     # [K⁻¹] thermal expansion coefficient 
g  = 9.8061   # [m/s²] gravitational constant
cᵖ = 3994.0   # [J/K]  heat capacity
ρ  = 999.8    # [kg/m³] reference density

parameters = (
    Ly = Ly,
    Lz = Lz,
    Qᵇ = 10 / (ρ * cᵖ) * α * g,          # buoyancy flux magnitude [m² s⁻³]    
    y_shutoff = 5 / 6 * Ly,              # shutoff location for buoyancy flux [m]
    τ = 0.1 / ρ,                         # surface kinematic wind stress [m² s⁻²]
    μ = 1 / 30days,                      # bottom drag damping time-scale [s⁻¹]
    ΔB = 8 * α * g,                      # surface vertical buoyancy gradient [s⁻²]
    H = Lz,                              # domain depth [m]
    h = 1000.0,                          # exponential decay scale of stable stratification [m]
    y_sponge = 19 / 20 * Ly,             # southern boundary of sponge layer [m]
    λt = 7.0days                         # relaxation time scale [s]
)

#####
##### Coriolis
#####

const f = -1e-4
const β = 1e-11
coriolis = BetaPlane(f₀ = f, β = β)

#####
##### Model building
#####

using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, HorizontalDivergenceScalarBiharmonicDiffusivity

@show νh = (Lx / Nx)^4 / 5days

horizontal_viscosity  = HorizontalDivergenceScalarBiharmonicDiffusivity(ν = νh)
vertical_viscosity    = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν = 1e-5)
convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0)

closure = (horizontal_viscosity, convective_adjustment)

@info "Building a model..."
model = HydrostaticFreeSurfaceModel(; grid, closure,
                                    free_surface = ImplicitFreeSurface(),
                                    momentum_advection = WENO(vector_invariant=VelocityStencil()),
                                    tracer_advection = WENO(grid),
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = coriolis,
                                    tracers = (:b, :c))

@info "Built $model."

#####
##### Initial conditions
#####

# resting initial condition
file = jldopen("restart_file.jld2")
u_init = file["u/data"][6:end-5, 6:end-5, 6:end-5]
v_init = file["v/data"][6:end-5, 6:end-5, 6:end-5]
w_init = file["w/data"][6:end-5, 6:end-5, 6:end-5]
b_init = file["b/data"][6:end-5, 6:end-5, 6:end-5]
η_init = file["η/data"][6:end-5, 6:end-5, :]

c_init = ((b_init .- minimum(b_init)) ./ (maximum(b_init) - minimum(b_init)))

Gu⁻ = file["timestepper/Gⁿ/u/data"][6:end-5, 6:end-5, 6:end-5]
Gv⁻ = file["timestepper/Gⁿ/v/data"][6:end-5, 6:end-5, 6:end-5]
Gb⁻ = file["timestepper/Gⁿ/b/data"][6:end-5, 6:end-5, 6:end-5]

u, v, w = model.velocities
b = model.tracers.b
c = model.tracers.c

set!(u, u_init)
set!(v, v_init)
set!(w, w_init)
set!(b, b_init)
set!(c, c_init)

set!(model.timestepper.G⁻.u, Gu⁻)
set!(model.timestepper.G⁻.v, Gv⁻)
set!(model.timestepper.G⁻.b, Gb⁻)

model.clock.iteration = 1

#####
##### Simulation building
#####

Δt₀       = 1minutes
stop_time = 100days

simulation = Simulation(model, Δt = Δt₀, stop_time = stop_time)

# add timestep wizard callback
wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=3minutes)
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

simulation.output_writers[:averages] = JLD2OutputWriter(model, (; u, v, w, b, c),
                                                        schedule = TimeInterval(12hours),
                                                        filename = "abernathey_channel_fields_velocity_stencil",
                                                        overwrite_existing = true)

@info "Running the simulation..."

try
    run!(simulation, pickup = false)
catch err
    @info "run! threw an error! The error message is"
    showerror(stdout, err)
end

# #####
# ##### Visualization
# #####

using Plots

grid = RectilinearGrid(CPU(),
                       topology = (Periodic, Bounded, Bounded),
                       size = (grid.Nx, grid.Ny, grid.Nz),
                       halo = (5, 5, 5),
                       x = (0, grid.Lx),
                       y = (0, grid.Ly),
                       z = z_faces)

xζ, yζ, zζ = nodes((Face, Face, Center), grid)
xc, yc, zc = nodes((Center, Center, Center), grid)
xw, yw, zw = nodes((Center, Center, Face), grid)