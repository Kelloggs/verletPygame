"""
Verlet integration scheme for a deformable object using a mesh-based
Mass Spring System. This has just been written to test pygame.

Require: numpy, pygame

TODO: 
- add less naive collision handling and response between objects

Author: Jens Cornelis
"""

from numpy import array, linalg, cross, dot
import math
import pygame
import sys

#globalsb
world_size = 1000,700  
world_rect = pygame.Rect(0,0, world_size[0], world_size[1])
num_iterations = 10 #iteration for constraint relaxation
pickedParticle = None
mousePosition = 0,0 
frames = 25
drawVelocities = False

class Material:
	def __init__(self, stiffness=0.3, friction=0.1):
		self.stiffness = stiffness
		self.friction = friction

class MSSObject:
	def __init__(self, vertices, indexedSprings, screen, material):
		self.screen = screen
		self.particles = []
		self.material = material

		#set up particles
		for vertex in vertices:
			self.particles.append(Particle(vertex, screen, self))

		self.mass = len(self.particles)

		#set up springs/constraints
		self.constraints = []
		for spring in indexedSprings:
			self.constraints.append(Constraint(self.particles[spring[0]], self.particles[spring[1]]))

		#initial draw
		self.draw()

	def setMass(self, mass):
		self.mass = mass
		partialMass = mass / float(len(self.particles))
		for particle in self.particles:
			particle.mass = partialMass

	def get_rect(self):
		x_min, y_min = world_size[0], world_size[1]
		x_max, y_max = 0, 0
		for particle in self.particles:
			if particle.x[0] > x_max:
				x_max = particle.x[0]
			if particle.x[0] < x_min:
				x_min = particle.x[0]
			if particle.x[1] > y_max:
				y_max = particle.x[1]
			if particle.x[1] < y_min:
				y_min = particle.x[1]
		return pygame.Rect(x_min, y_min, x_max - x_min, y_max - y_min)

	def update(self):
		for particle in self.particles:
			particle.update()

	def draw(self):
		for constraint in self.constraints:
			pos1 = (constraint.p1.x[0], constraint.p1.x[1])
			pos2 = (constraint.p2.x[0], constraint.p2.x[1])
			pygame.draw.aaline(self.screen, (0,0,255), pos1, pos2)

		for particle in self.particles:
			particle.draw()

class DeformableSphere(MSSObject):
	def __init__(self, center, radius, vertices, indexedSprings, screen, material=Material()):
		MSSObject.__init__(self, vertices, indexedSprings, screen, material)
		self.radius = radius
		self.center = center

	def movePointOut(self, point):
		direction = point - self.center
		directionLength = linalg.norm(direction)
		normalizedDirection = direction/directionLength
		distMax = 0
		for particle in self.particles:
			tmp = dot(normalizedDirection, particle.x - self.center)
			if tmp > distMax:
				distMax = tmp
		if linalg.norm(direction) < distMax:
			diff = (self.radius - directionLength)/directionLength	
			return direction*diff
		else:
			return (0,0)


class DeformableCube(MSSObject):
	def __init__(self, vertices, indexedSprings, screen, material=Material()):
		MSSObject.__init__(self, vertices, indexedSprings, screen, material)
		raise Error("Not yet implemented")

	def movePointOut(self, point):
		return (0,0)

class Constraint:
	def __init__(self, p1, p2):
		self.p1 = p1
		self.p2 = p2
		self.restlength = linalg.norm(p1.x - p2.x)

class Particle:
	def __init__(self, x, screen, parentObject, mass = 1.0):
		#set up physical quantities
		self.x = x
		self.oldx = x
		self.force = array([0., 0.])
		self.mass = mass
		self.image = pygame.image.load("sphere.png")
		self.picked = False
		self.parentObject = parentObject
		self.velocity = array([0,0]) 

		#set bounding volume and position
		self.bv = self.image.get_rect()
		self.radius = self.bv[2]/2.0
		self.bv[0] = self.x[0] - self.radius
		self.bv[1] = self.x[1] - self.radius

		#initial drawing
		self.screen = screen
		self.draw()

	def draw(self):
		self.screen.blit(self.image, self.bv)
		if drawVelocities:
			pygame.draw.line(self.screen, (255,0,0), self.x, self.x + self.velocity)

	def update(self):
		self.bv[0] = self.x[0] - self.radius
		self.bv[1] = self.x[1] - self.radius


def computeForces(objects):
	#add gravitational forces and friction
	for obj in objects:
		for particle in obj.particles:
			particle.force = array([0.0, particle.mass * 9.81 * 100.0])

def computeFriction(objects):
	for obj in objects:
		for particle in obj.particles:
			friction = particle.parentObject.material.friction
			if not particle.x[0] < (world_size[0] - particle.radius): 
				delta = particle.x[1] - particle.oldx[1]
				depth = math.fabs(world_size[0] - particle.radius - particle.x[0])
				particle.oldx[1] = particle.x[1] - depth*friction*delta
			if not (particle.x[0] > particle.radius):
				delta = particle.x[1] - particle.oldx[1]
				depth = math.fabs(particle.radius - particle.x[0])
				particle.oldx[1] = particle.x[1] - depth*friction*delta
			if not particle.x[1] < (world_size[1] - particle.radius): 
				delta = particle.x[0] - particle.oldx[0]
				depth = math.fabs(world_size[1] - particle.radius - particle.x[1])
				particle.oldx[0] = particle.x[0] - depth*friction*delta
			if not (particle.x[1] > particle.radius):
				delta = particle.x[0] - particle.oldx[0]
				depth = math.fabs(particle.radius - particle.x[1])
				particle.oldx[0] = particle.x[0] - depth*friction*delta


def verlet(h, objects):
	for obj in objects:
		for particle in obj.particles:
			x = array([particle.x[0], particle.x[1]])
			tmp = array([particle.x[0], particle.x[1]])
			oldx = particle.oldx
			a = particle.force / particle.mass
			particle.x += x - oldx + a*h*h
			#compute velocity as central difference of positions
			particle.velocity = (particle.x - particle.oldx)/(2.0*h)
			particle.oldx = tmp
	

def satisfyConstraints(objects):
	for val in range(num_iterations):
		for obj in objects:
			#check and solve world collisions
			for particle in obj.particles:
				particle.x[0] = min(max(particle.x[0], particle.radius), world_size[0] - particle.radius)
				particle.x[1] = min(max(particle.x[1], particle.radius), world_size[1] - particle.radius)

			#solve constraints deformable object
			for constraint in obj.constraints:
				p1 = constraint.p1
				p2 = constraint.p2
				delta = p2.x - p1.x
				deltalength = linalg.norm(delta)
				diff = (deltalength - constraint.restlength)/deltalength

				#make material stiffness linear to solver iterations and apply to
				#particle positions
				k = 1 - (1 - obj.material.stiffness)**(1.0/float(num_iterations))
				p1.x += delta*0.5*diff*k
				p2.x -= delta*0.5*diff*k

			#constraint for picked particle to act on mouse action
			if pickedParticle:
				delta = pickedParticle.x - mousePosition
				deltalength = linalg.norm(delta)
				if deltalength > 0:
					diff = (0 - deltalength)/deltalength
					pickedParticle.x += delta*diff

			for particle in obj.particles:
				for obj2 in objects:
					if obj == obj2:
						continue
					else:
						if obj2.get_rect().collidepoint(particle.x):
							particle.x += obj2.movePointOut(particle.x)
							
def create2DBall(screen, center, radius, particles, material):
	'''Convenience method to generate a ball mesh'''
	p = []
	p.append(center)
	for val in range(particles):
		angle = val * 360./particles
		tmp_x = center[0] + radius*math.cos((angle*math.pi)/180.)
		tmp_y = center[1] + radius*math.sin((angle*math.pi)/180.)
		p.append(array([tmp_x, tmp_y]))
	
	c = []
	for val in range(1, len(p) - 1):
		c.append((val, val + 1))
		c.append((0, val))
	c.append((1, len(p) - 1))
	c.append((0, len(p) - 1))
	
	return DeformableSphere(center, radius, p, c, screen, material)

def create2DCube(screen, rect, material):
	p = []
	p.append(array([rect[0], rect[1]]))
	p.append(array([rect[0], rect[1] + rect[3]]))
	p.append(array([rect[0] + rect[2], rect[1]]))
	p.append(array([rect[0] + rect[2], rect[1] + rect[3]]))
	c = (0,1), (1,2), (2,3), (3, 0), (0,2), (1,3)

	return DeformableCube(p, c, screen, material)


def main():
	global pickedParticle, mousePosition, mouseClickPosition
	#initialization of pygame and window
	pygame.init()
	screen = pygame.display.set_mode(world_size)
	
	#setting up objects
	objects = []
	mat1 = Material(0.9)
	mat2 = Material(0.6)
	sphere = create2DBall(screen, array([300., 300.]), 100., 8, mat2)
	objects.append(sphere)
	sphere2 = create2DBall(screen, array([100., 100.]), 80., 7, mat1)
	objects.append(sphere2)

	clock = pygame.time.Clock()

	#main simulation loop
	while True:
		#set clock of pygame to predefined frames for equal timesteps
		clock.tick(frames)

		#clear screen with background color
		screen.fill((80,80,80))

		for event in pygame.event.get():
			#stop the program if user wants us to
			if event.type == pygame.QUIT:
				sys.exit()
			#flag particle as picked if user clicked on it
			if event.type == pygame.MOUSEBUTTONDOWN:
				for obj in objects:
					for particle in obj.particles:
						if particle.bv.collidepoint(event.pos):
							particle.picked = True
							pickedParticle = particle
			if event.type == pygame.MOUSEMOTION:
				mousePosition = event.pos
			if event.type == pygame.MOUSEBUTTONUP:	
				if pickedParticle:
					pickedParticle.picked = False
					pickedParticle = None

		#compute external forces
		computeForces(objects)
		computeFriction(objects)

		#compute timestep according to frame set py pygame.clock
		h=frames/1000.

		#do integration step and satisfy constraints
		verlet(h, objects)
		satisfyConstraints(objects)

		#update and draw particles
		for obj in objects:
			obj.update()
			obj.draw()

		#make everything visible
		pygame.display.flip()


#########################################
if __name__ == '__main__':
	main()
#########################################