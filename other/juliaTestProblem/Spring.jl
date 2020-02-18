# Equations of motion and simulation functions for the double mass-spring-damper system

using Plots
using DelimitedFiles

frequency = 1/3.0*[1.0, 1.0]  # Hz
stiffness = (frequency * (2 * pi)).^2
damping = 0.05
dt = 1/(200*maximum(frequency))

K0 = [sum(stiffness) -stiffness[2]; 
        -stiffness[2] sum(stiffness)]
K(x) = K0 .+ sqrt(sum(x.^2))*K0
C(x) = damping * K0

function xddot(x, xdot, f)
    return f - K(x)*x - C(x)*xdot
end

function staticSolve(f)
    return K\f
end

function sim(f)
    N = size(f)[2]
    x = [0; 0]
    xdot = [0; 0]
    xOut = zeros(2, N)
    for ii in 1:N
        println("Iteration $ii")
        println("x: $x, xdot: $xdot")
        println("f: $(f[:,ii])")
        # x = staticSolve(f[:,ii])
        xOut[:,ii] = x
        xdot = xdot + dt*xddot(x, xdot, f[:,ii])
        x = x + dt*xdot
    end
    return xOut
end

tt = 0:dt:3/minimum(frequency)
fApplied = zeros(2, length(tt))
noise = sinpi.(2*2*maximum(frequency).*tt)
fApplied[2,:] = noise*0.05 .+ (sinpi.(2*1/2.0.*minimum(frequency)*tt)).*(1/2.0*(1 .- cospi.((2*3/tt[length(tt)]).*tt)))
# fApplied[2,:] .= 1.0
xOut = sim(fApplied * 10*maximum(stiffness))

# plot(tt, xOut[1,:]/maximum(xOut[1,:]), label="Output")
# plot!(tt, fApplied[2,:]/maximum(fApplied[2,:]), label="Input", alpha=0.2)
# # scatter(xOut[1,:], fApplied[2,:])
# xlabel!("Time, s")
# # ylabel!("Output")
# # ylabel!("Input")
# savefig("spring.pdf")

f = open("spring.csv", "w")
for ii in 1:length(tt)
    write(f, "$(tt[ii]),$(fApplied[2,ii]),$(xOut[1,ii])\n")
end
close(f)

