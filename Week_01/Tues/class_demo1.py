import math

g = 9.81
v0 = 30
s0 = 25

t = 4.0;

print("Position of ball at t = %g is %g" %(t,-0.5*g*t**2 + v0*t + s0))


# To find the time at which the ball hits the ground

t_ground = (v0 + math.sqrt(v0**2 + 2*g*s0))/g
print('The ball hits the ground at %g' %(t_ground))

t_max = v0/g
h_max = -0.5*g*t**2 + v0*t + s0
print("Ball reaches a height of %g at t = %g" % (h_max,t_max))






