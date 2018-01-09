import math

g = 9.81    # gravity
v0 = 30
s0 = 25

t = 4.0

s = -0.5*g*t**2 + v0*t + s0

print("The position of the ball at time t = %g is %g" % (t,s))


# time it hits the ground : set s(t) = 0 and solve

t_ground = (v0 + math.sqrt(v0**2 + 2*g*s0))/g

print('Time ball hits the ground is %g' % t_ground)


t_max = v0/g
s_max = -0.5*g*t_max**2 + v0*t_max + s0

print('Maximum height is reached at t = %g and height is %g' % (t_max,s_max))