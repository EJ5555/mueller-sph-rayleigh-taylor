#if __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <iostream>
#include <vector>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

// "Particle-Based Fluid Simulation for Interactive Applications"
// solver parameters
static Vector2d G(0.f, 12000.f * -9.8f); // external (gravitational) forces
const static float REST_DENS = 1000.f;		 // rest density
const static float REST_DENS2 = 1000.f;
const static float GAS_CONST = 3250.f;		 // const for equation of state
const static float GAMMA = 1.f; 	// Inkompressibilität
const static float H = 16.f;				 // kernel radius
const static float HSQ = H * H;				 // radius^2 for optimization
const static float MASS = 65.f;				 // assume all particles have the same mass
const static float MASS2 = 60.f;
const static float VISC = 250.f;			 // viscosity constant
const static float DT = 0.0008f;			 // integration timestep

// smoothing kernels defined in Müller and their gradients
const static float POLY6 = 315.f / (65.f * M_PI * pow(H, 9.f));
const static float SPIKY_GRAD = -45.f / (M_PI * pow(H, 6.f));
const static float VISC_LAP = 45.f / (M_PI * pow(H, 6.f));

// simulation parameters
const static float EPS = H; // boundary epsilon
const static float BOUND_DAMPING = -0.5f;

//leapfrog
static bool first = true;

// particle data structure
// stores position, velocity, and force for integration
// stores density (rho) and pressure values for SPH
struct Particle
{
	Particle(float _x, float _y, float _m, float _rd, string _color) : x(_x, _y), v(0.f, 0.f), f(0.f, 0.f), rho(0), p(0.f), m(_m), rd(_rd), color(_color) {}
	Vector2d x, v, f;
	float rho, p;
  float m;
	float rd;
  const string color;
};

// solver data
static vector<Particle> particles;
//static vector<Particle> particles2;

// interaction
const static int MAX_PARTICLES = 3000;
const static int DAM_PARTICLES = 40000;
const static int BLOCK_PARTICLES = 250;

// rendering projection parameters
const static int WINDOW_WIDTH = 800;
const static int WINDOW_HEIGHT = 600;
const static double VIEW_WIDTH = 1.5 * 800.f;
const static double VIEW_HEIGHT = 1.5 * 600.f;

void InitSPH(void)
{
	cout << "initializing dam break with " << MAX_PARTICLES << " particles" << endl;
	for (float y = EPS; y < VIEW_HEIGHT/2; y += 1.f * H/1.5){
		for (float x = 2.f * EPS; x <= VIEW_WIDTH - 2.f * EPS; x += 0.5 * H){
			if (particles.size() < MAX_PARTICLES)
			{
				float jitter = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (particles.size()%2==0){
				      particles.push_back(Particle(x + jitter, y + 0.1 * VIEW_HEIGHT * sin(2 * M_PI * (x+jitter)/(VIEW_WIDTH - 2.f * EPS)), MASS, REST_DENS, "blue"));
        } else {
				      particles.push_back(Particle(x + jitter, y + 0.1 * VIEW_HEIGHT * sin(2 * M_PI * (x+jitter)/(VIEW_WIDTH - 2.f * EPS)) + VIEW_HEIGHT/2, MASS2, REST_DENS2, "red"));
          }
			}
    }
    }
  }


void ComputeDensityPressure(void)
{
	for (auto &pi : particles)
	{
		pi.rho = 0.f;
		for (auto &pj : particles)
		{
			Vector2d rij = pj.x - pi.x;
			float r2 = rij.squaredNorm();

			if (r2 < HSQ)
			{
				// this computation is symmetric
				pi.rho += pi.m * POLY6 * pow(HSQ - r2, 3.f);
			}
		}
		pi.p = GAS_CONST * pow(pi.rho - pi.rd, GAMMA);
	}
}

void ComputeForces(void)
{
	for (auto &pi : particles)
	{
		Vector2d fpress(0.f, 0.f);
		Vector2d fvisc(0.f, 0.f);
		for (auto &pj : particles)
		{
			if (&pi == &pj)
				continue;

			Vector2d rij = pj.x - pi.x;
			float r = rij.norm();

			if (r < H)
			{
				// compute pressure force contribution
				fpress += -rij.normalized() * pj.m * (pi.p + pj.p) / (2.f * pj.rho) * SPIKY_GRAD * pow(H - r, 2.f);
				// compute viscosity force contribution
				fvisc += VISC * pj.m * (pj.v - pi.v) / pj.rho * VISC_LAP * (H - r);
			}
		}
		Vector2d fgrav = G * pi.rho;
		pi.f = fpress + fvisc + fgrav;
	}
}


void Integrate(void)
{
	if (first == true){
	for (auto &p : particles){
		p.v += DT/2 * p.f / p.rho;
		first = false;
		}
	}
	for (auto &p : particles)
	{
		// forward Euler integration, now leapfrog
		p.x += DT * p.v;

		// enforce boundary conditions
		if (p.x(0) - EPS < 0.0f)
		{
			p.v(0) *= BOUND_DAMPING;
			p.x(0) = EPS;
		}
		if (p.x(0) + EPS > VIEW_WIDTH)
		{
			p.v(0) *= BOUND_DAMPING;
			p.x(0) = VIEW_WIDTH - EPS;
		}
		if (p.x(1) - EPS < 0.0f)
		{
			p.v(1) *= BOUND_DAMPING;
			p.x(1) = EPS;
		}
		if (p.x(1) + EPS > VIEW_HEIGHT)
		{
			p.v(1) *= BOUND_DAMPING;
			p.x(1) = VIEW_HEIGHT - EPS;
		}
	}
	ComputeDensityPressure();
	ComputeForces();
	for (auto &p : particles){
		p.v += DT * p.f / p.rho;
	}

}

void Update(void)
{
	ComputeDensityPressure();
	ComputeForces();
	Integrate();

	glutPostRedisplay();
}

void InitGL(void)
{
	glClearColor(0.9f, 0.9f, 0.9f, 1);
	glEnable(GL_POINT_SMOOTH);
	glPointSize(H / 2.f);
	glMatrixMode(GL_PROJECTION);
}

void Render(void)
{
	glClear(GL_COLOR_BUFFER_BIT);

	glLoadIdentity();
	glOrtho(0, VIEW_WIDTH, 0, VIEW_HEIGHT, 0, 1);

	glColor4f(0.2f, 0.6f, 1.0f, 1);
	glBegin(GL_POINTS);
	for (auto &p : particles){
  //auto n = particles.size();
  //for (decltype(n) i = 0; i < n; i++){
    if(p.color=="blue"){
      glVertex2f(p.x(0), p.x(1));
    }
  }
	glEnd();

  glColor4f(1.0f, 0.0f, 0.0f, 0.0f);
	glBegin(GL_POINTS);
  for (auto &p : particles){
  //for (decltype(n) i = 0; i < n; i++){
    if(p.color=="red"){
      glVertex2f(p.x(0), p.x(1));
    }
  }
  glEnd();

	glutSwapBuffers();
}

void Keyboard(unsigned char c, __attribute__((unused)) int x, __attribute__((unused)) int y)
{
	switch (c)
	{
	case ' ':
		//if (particles.size() >= MAX_PARTICLES)
			//std::cout << "maximum number of particles reached" << std::endl;
		//else
		//{
		//	unsigned int placed = 0;
		//	for (float y = VIEW_HEIGHT / 1.5f - VIEW_HEIGHT / 5.f; y < VIEW_HEIGHT / 1.5f + VIEW_HEIGHT / 5.f; y += H * 0.95f)
		//		for (float x = VIEW_WIDTH / 2.f - VIEW_HEIGHT / 5.f; x <= VIEW_WIDTH / 2.f + VIEW_HEIGHT / 5.f; x += H * 0.95f)
		//			if (placed++ < BLOCK_PARTICLES && particles.size() < MAX_PARTICLES)
		//				particles.push_back(Particle(x, y, MASS, REST_DENS, "blue"));
		//}
		for(auto &p: particles){
			if (p.m == MASS){
				p.m = MASS2;
				p.rd = REST_DENS2;
			} else {
				p.m = MASS;
				p.rd = REST_DENS;
			}
		}
		//G[1] = 5000.f * -9.8f;
		break;
	case 'r':
	case 'R':
		particles.clear();
		InitSPH();
		break;
	}
}

int main(int argc, char **argv)
{
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInit(&argc, argv);
	glutCreateWindow("Müller SPH");
	glutDisplayFunc(Render);
	glutIdleFunc(Update);
	glutKeyboardFunc(Keyboard);

	InitGL();
	InitSPH();

	glutMainLoop();
	return 0;
}
