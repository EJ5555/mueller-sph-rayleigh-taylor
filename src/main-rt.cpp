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

// "Particle-Based Fluid Simulation for Interactive Applications"
// solver parameters
const static float M_PI = 3.14159265359;
static Vector2d G(0.f, 12000.f * -9.8f); // external (gravitational) forces
const static float REST_DENS = 1000.f;		 // rest density
const static float REST_DENS2 = 1000.f;
const static float GAS_CONST = 1800.f;		 // const for equation of state
const static float GAS_CONST2 = 2500.f;
const static float GAMMA = 1.085f; 	// Inkompressibilität
const static float H = 16.f;				 // kernel radius
const static float HSQ = H * H;				 // radius^2 for optimization
const static float MASS = 60.f;				 // assume all particles have the same mass
const static float MASS2 = 90.f;
const static float VISC = 200.f;			 // viscosity constant
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
	Particle(float _x, float _y, float _m, float _rd, float _gc, string _color) : x(_x, _y), v(0.f, 0.f), f(0.f, 0.f), rho(0), p(0.f), m(_m), rd(_rd), gc(_gc), color(_color) {}
	Vector2d x, v, f;
	float rho, p;
  float m;
	float rd;
	float gc;
  const string color;
};

// solver data
static vector<Particle> particles;
//static vector<Particle> particles2;

// interaction
const static int MAX_PARTICLES = 2500;
const static int DAM_PARTICLES = 40000;
const static int BLOCK_PARTICLES = 250;

// rendering projection parameters
const static int WINDOW_WIDTH = 800;
const static int WINDOW_HEIGHT = 600;
const static double VIEW_WIDTH = 1.5 * 800.f;
const static double VIEW_HEIGHT = 1.5 * 600.f;


//Grenzfläche
static int border = 0;
static float border_hight = VIEW_HEIGHT / 2 + 2* EPS;
static float impuls_unten = 0;
static float impuls_oben = 0;

//Obere Grenzfläche
static float upper_border = VIEW_HEIGHT;

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
	cout << "initializing " << MAX_PARTICLES << " particles" << endl;

	for (int i = 0; i < MAX_PARTICLES; i++)
	{
		float sigma = MASS * MAX_PARTICLES / (VIEW_WIDTH * REST_DENS); //GAS_CONST / (MASS * (-1.f) * G[1]) * 350000.f;
		float y = -sigma *1000* log(1 - zufall());
		float x = zufall() * VIEW_WIDTH;
	
		particles.push_back(Particle(x, y, MASS, REST_DENS, GAS_CONST, "blue"));
	}
	

	//for (auto& p : particles) {
	//	p.v(0) = 5000.f;
	//}
	//for (float y = EPS; y < VIEW_HEIGHT / 2;)
	//{
	//	float dh = H/2;
	//	for (float x = 2.f * EPS; x <= VIEW_WIDTH - 2.f * EPS; x += dh)
	//	{
	//		particles.push_back(Particle(x, y, MASS, REST_DENS, "blue"));
	//	
	//	}
	//	y += dh;
    //}
	//for (float y = VIEW_HEIGHT / 2 + EPS; y < VIEW_HEIGHT;)
	//{
	//	float dh = H/2;
	//	for (float x = 1.f * EPS; x <= VIEW_WIDTH - 1.f * EPS; x += dh)
	//	{
	//		particles.push_back(Particle(x, y, MASS2, REST_DENS2, "red"));
	//	}
	//	y += dh;
	//}
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
#pragma omp parallel for
	for (auto &p : particles){
		p.v += DT/2 * p.f / p.rho;
		first = false;
		}
	}
	impuls_oben = 0;
	impuls_unten = 0;
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
		if ((border == 1 || border == 2) && abs(p.x(1) - border_hight) < EPS)
		{
			if (p.x(1) > border_hight)
			{
				impuls_unten += abs(p.v(1)) * p.m;
				p.v(1) *= BOUND_DAMPING;
				p.x(1) = border_hight + EPS;
			}
			else
			{
				impuls_oben += abs(p.v(1)) * p.m;
				p.v(1) *= BOUND_DAMPING;
				p.x(1) = border_hight - EPS;
			}

		}
		if (abs(p.x(1) - upper_border) < EPS)
		{
			if (p.x(1) < upper_border)
			{
				p.v(1) *= BOUND_DAMPING;
				p.x(1) = upper_border - EPS;
			}
		}


		border_hight += (impuls_oben - impuls_unten) * DT / 2000000;


		//if (p.x(1) + EPS > VIEW_HEIGHT)
		//{
		//	p.v(1) *= -1.f;
		//	p.x(1) = VIEW_HEIGHT - EPS - (p.x(1) + EPS - VIEW_HEIGHT);
		//}
	}
	ComputeDensityPressure();
	ComputeForces();
#pragma omp parallel for
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

  glColor4f(0.5f, 0.5f, 0.5f, 0.0f);
	glBegin(GL_POINTS);
	for (float x = 0; x < VIEW_WIDTH; x += VIEW_WIDTH/10)
	{
		glVertex2f(x, border_hight);
		glVertex2f(x, upper_border);
	}
	glEnd();


	glutSwapBuffers();
}

void Keyboard(unsigned char c, __attribute__((unused)) int x, __attribute__((unused)) int y)
{
	switch (c)
	{
	case ' ':
		if (border == 0)
		{
			for (int i = 0; i < MAX_PARTICLES/2.f; i++)
			{
				float sigma = MASS2 * MAX_PARTICLES / (VIEW_WIDTH * REST_DENS2); //GAS_CONST / (MASS * (-1.f) * G[1]) * 350000.f;
				float y = -sigma * 1000 * log(1 - zufall()) + border_hight+3*EPS;
				float x = zufall() * VIEW_WIDTH;

				particles.push_back(Particle(x, y, MASS2, REST_DENS2, GAS_CONST2, "red"));
			}
			border += 1;
		}
		else
		{
			border += 1;
		}

		//for(auto &p: particles){
		//	if (p.m == MASS){
		//		p.m = MASS2;
		//		p.rd = REST_DENS2;
		//	} else {
		//		p.m = MASS;
		//		p.rd = REST_DENS;
		//	}
		//}
		//G[1] = 5000.f * -9.8f;
		break;
	case 'w':
			upper_border += 10;
			break;
		case 's':
				upper_border -= 10;
				break;
	case 'r':
	case 'R':
		particles.clear();
		InitSPH();
		border_hight = VIEW_HEIGHT / 2;
		border = 0;
		break;
	}
}

int main(int argc, char **argv)
{
	omp_set_num_threads(4);
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
