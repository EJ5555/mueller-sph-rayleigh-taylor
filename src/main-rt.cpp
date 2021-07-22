#if __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;


// interaction
const static int MAX_PARTICLES = 4000;
const static int DAM_PARTICLES = 40000;
const static int BLOCK_PARTICLES = 250;

// rendering projection parameters
const static int WINDOW_WIDTH = 800;
const static int WINDOW_HEIGHT = 600;
const static double VIEW_WIDTH = 1.5 * 800.f;
const static double VIEW_HEIGHT = 1.5 * 600.f;


// "Particle-Based Fluid Simulation for Interactive Applications"
// solver parameters
//const static float M_PI = 3.14159265359;
static Vector2d G(0.f, -9.8f); // external (gravitational) forces
const static float REST_DENS = 0.f;//1000.f;		 // rest density
const static float REST_DENS2 = 0.f;
const static float MASS = 60.f;				 // assume all particles have the same mass
const static float MASS2 = 300.f;
const static float GAS_CONST = 10000.f;		 // const for equation of state
static float sigma1 = GAS_CONST / ((-1.f) * G[1]);
const static float GAS_CONST2 = GAS_CONST * MASS / MASS2;
static float sigma2 = GAS_CONST2 / ((-1.f) * G[1]);
const static float GAMMA = 1.085f; 	// Inkompressibilität
const static float H = 64.f;				 // kernel radius
const static float HSQ = H * H;				 // radius^2 for optimization
const static float VISC = 0.1f;			 // viscosity constant
const static float VISC2 = 0.1f;
const static float DT = 0.005f;			 // integration timestep
static float T = 0;

// smoothing kernels defined in Müller and their gradients
const static float POLY6 = 315.f / (65.f * M_PI * pow(H, 9.f));
const static float SPIKY_GRAD = -45.f / (M_PI * pow(H, 6.f));
const static float VISC_LAP = 45.f / (M_PI * pow(H, 6.f));

// simulation parameters
const static float EPS = 16.f; //H; // boundary epsilon
const static float BOUND_DAMPING = -.95f;

//leapfrog
static bool first = true;


// particle data structure
// stores position, velocity, and force for integration
// stores density (rho) and pressure values for SPH
struct Particle
{
	Particle(float _x, float _y, float _m, float _rd, float _gc, float _vs, string _color) : x(_x, _y), v(0.f, 0.f), f(0.f, 0.f), rho(0), p(0.f), m(_m), rd(_rd), gc(_gc), vs(_vs), color(_color) {}
	Vector2d x, v, f;
	float rho, p;
  float m;
	float rd;
	float gc;
	float vs;
  const string color;
};

// solver data
static vector<Particle> particles;
//static vector<Particle> particles2;



//Grenzfläche
static int border = 0;
static int move_border = 0;
static float border_height = VIEW_HEIGHT /2.f;
static float border_v = 0;
static float force_up = 0;
static float force_down = 0;

static float zufall() {
	double rdn = (float)rand() / RAND_MAX;
	if (rdn == 1)
	{
		rdn = zufall();
	}
	return  rdn;
}

void InitSPH(void)
{
	cout << "initialize " << MAX_PARTICLES << " particles" << endl;

	for (int i = 0; i < MAX_PARTICLES; i++)
	{
		float a = EPS;
		float b = border_height - EPS;
		float y = -sigma1 * log(exp(-a / sigma1) - zufall() * (exp(-a / sigma1) - exp(-b / sigma1)));
		float x = zufall() * VIEW_WIDTH;
		Particle p = Particle(x, y, MASS, REST_DENS, GAS_CONST, VISC, "blue");

		particles.push_back(p);
	}

	cout << "initialize " << MAX_PARTICLES << " heavy particles" << endl;
	for (int i = 0; i < MAX_PARTICLES; i++)
	{
		float a = border_height + EPS;
		float b = VIEW_HEIGHT - EPS;
		float y = -sigma2 * log(exp(-a / sigma2) - zufall() * (exp(-a / sigma2) - exp(-b / sigma2)));
		float x = zufall() * VIEW_WIDTH;
		Particle p = Particle(x, y, MASS2, REST_DENS2, GAS_CONST2, VISC2, "red");

		particles.push_back(p);
	}
  }


void ComputeDensityPressure(void)
{
#pragma omp parallel for
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
		pi.p = pi.gc * (pi.rho - pi.rd)/abs(pi.rho - pi.rd) * pow(abs(pi.rho - pi.rd), GAMMA);
	}
}

void ComputeForces(void)
{
#pragma omp parallel for
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
				fpress += rij.normalized() * pj.m * (pi.p / pow(pi.rho,2) + pj.p / pow(pj.rho,2)) * SPIKY_GRAD * pow(H - r, 2.f);
				// compute viscosity force contribution
				fvisc += pi.vs * pj.m * (pj.v - pi.v) / pj.rho * VISC_LAP * (H - r);
			}
		}
		Vector2d fgrav = G * pi.rho;
		pi.f = fpress + fvisc + fgrav;
	}
}


void Integrate(void)
{
	if (first == true) {
#pragma omp parallel for
		for (auto& p : particles) {
			p.v += DT / 2 * p.f / p.rho;
			first = false;
		}
	}
	force_up = 0;
	force_down = 0;

	T += DT;
	for (auto& p : particles)
	{
		// forward Euler integration, now leapfrog
		p.x += DT * p.v;
		//cout << T << endl;

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

		if ((border == 100) && (abs(p.x(1) - border_height) < EPS))
		{
			if (p.x(1) > border_height)
			{
				force_down += p.m * abs(p.v(1)) / DT + abs(p.f(1));
				p.v(1) *= BOUND_DAMPING;
				p.x(1) = border_height + EPS;
			}
			else
			{
				force_up += p.m * abs(p.v(1)) / DT + abs(p.f(1));
				p.v(1) *= BOUND_DAMPING;
				p.x(1) = border_height - EPS;
			}

		}
	}
	if (move_border == 1)
	{
		border_v += (force_up - force_down) * DT / 1000 - border_v;
		border_height += border_v * DT;
	}

		ComputeDensityPressure();
		ComputeForces();
#pragma omp parallel for
		for (auto& p : particles) {
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

void renderBitmapString(float x, float y, void* font, string str)
{
	glRasterPos2f(x, y);
	for (string::iterator c = (&str)->begin(); c != (&str)->end(); ++c)
	{
		glutBitmapCharacter(font, *c);
	}
}


void InitGL(void)
{
	glClearColor(0.9f, 0.9f, 0.9f, 1);
	glEnable(GL_POINT_SMOOTH);
	glPointSize(H / 20.f);
	glMatrixMode(GL_PROJECTION);
}

void renderBitmapString(float x, float y, void *font, string str)
{
  glRasterPos2f(x,y);
  for (string::iterator c = (&str)->begin(); c != (&str)->end(); ++c)
  {
    glutBitmapCharacter(font, *c);
  }
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

  glColor4f(0.5f, 0.5f, 0.5f, 0.0f);
	glBegin(GL_POINTS);
	for (float x = 0; x < VIEW_WIDTH; x += VIEW_WIDTH/10)
	{
		glVertex2f(x, border_height);
	}
	glEnd();

	glColor4f(0.1f, 0.1f, 0.1f, 0.0f);
	std::ostringstream stream;
	stream << "t = " << T;
	std::string text = stream.str();
	renderBitmapString(5, VIEW_HEIGHT - 25, GLUT_BITMAP_TIMES_ROMAN_24, text);


	glutSwapBuffers();
}


void Keyboard(unsigned char c, __attribute__((unused)) int x, __attribute__((unused)) int y)
{
	switch (c)
	{
	case ' ':
		border += 1;
		break;
	case 'm':
		move_border += 1;
		break;
	case 'r':
	case 'R':
		cout << "reset system" << endl;
		particles.clear();
		InitSPH();
		border_height = VIEW_HEIGHT / 2.f;
		T = 0;
		move_border = 0;
		border = 0;
		break;
	}
}

int main(int argc, char **argv)
{
	omp_set_num_threads(4);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInit(&argc, argv);
	glutCreateWindow("Müller SPH - Rayleigh-Taylor Instabilität");
	glutDisplayFunc(Render);
	glutIdleFunc(Update);
	glutKeyboardFunc(Keyboard);

	InitGL();
	InitSPH();

	glutMainLoop();
	return 0;
}
