#include "lab2.h"
#include <ctime>
#include <time.h>
#include <stdlib.h>  
#include <iostream>
using namespace std;
const int FIREWORKS = 20;
const int FIREWORK_PARTICLES = 50;

static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 480;



class Firework
{

	public:
		float x[FIREWORK_PARTICLES];
		float y[FIREWORK_PARTICLES];
		float xSpeed[FIREWORK_PARTICLES];
		float ySpeed[FIREWORK_PARTICLES];

		float R;
		float G;
		float B;
		int Y;
		int U;
		int V;
		float alpha;
		int particleSize;
		bool hasExploded;
		int framesUntilLaunch;
		static const float baselineYSpeed;
		static const float maxYSpeed;
		static const float GRAVITY;
 
		Firework(); // Constructor declaration
		void initialise();
		void move();
		void explode();
		void updateYUV();
};

__global__ void draw(uint8_t* Y,uint8_t* U,uint8_t* V,int fireworkNum,int offsetx,int offsety,Firework* fw){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<FIREWORK_PARTICLES){
		if(fw[fireworkNum].x[idx]>=0 && fw[fireworkNum].x[idx]+offsetx<W && fw[fireworkNum].y[idx]+offsety>=0 && fw[fireworkNum].y[idx]+offsety<H){
			int particlePos = (int)fw[fireworkNum].x[idx]+offsetx+W*((int)fw[fireworkNum].y[idx]+offsety);
			Y[particlePos] = fw[fireworkNum].Y;
			int choPos = particlePos/4;
			U[choPos] = fw[fireworkNum].U;
			V[choPos] = fw[fireworkNum].V;
		}
	}
}

const float Firework::GRAVITY = 0.1;
const float Firework::baselineYSpeed = -4.0f;
const float Firework::maxYSpeed = -4.0f;

Firework::Firework(){
	initialise();
}
void Firework::initialise()
{
    // Pick an initial x location and  random x/y speeds
    float xLoc = (rand() / (float)RAND_MAX) * W;
    float xSpeedVal = -2 + (rand() / (float)RAND_MAX) * 4.0;
    float ySpeedVal = baselineYSpeed + ((float)rand() / (float)RAND_MAX) * maxYSpeed;
 
    // Set initial x/y location and speeds
    for (int loop = 0; loop < FIREWORK_PARTICLES; loop++)
    {
        x[loop] = xLoc;
        y[loop] = H; // Push the particle location down off the bottom of the screen
        xSpeed[loop] = xSpeedVal;
        ySpeed[loop] = ySpeedVal;
    }
 
    // Assign a random colour and full alpha (i.e. particle is completely opaque)
    R = (rand() / (float)RAND_MAX)*255;
    G = (rand() / (float)RAND_MAX)*255;
    B = (rand() / (float)RAND_MAX)*255;
    Y = 0.299*R+0.587*G+0.114*B;
    U = -0.169*R+-0.331*G+0.5*B+128;
    V = 0.5*R-0.419*G-0.081*B+128;
    alpha = 1.0f;
 
    // Firework will launch after a random amount of frames between 0 and 400
    framesUntilLaunch = 2;//((int)rand() % NFRAME);
 
    // Size of the particle (as thrown to glPointSize) - range is 1.0f to 4.0f
    particleSize = 2;//1.0f + ((float)rand() / (float)RAND_MAX) * 3.0f;


 
    // Flag to keep trackof whether the firework has exploded or not
    hasExploded = false;
 
    cout << "Initialised a firework." << endl;
}
 
void Firework::move()
{

    for (int loop = 0; loop < FIREWORK_PARTICLES; loop++)
    {
        // Once the firework is ready to launch start moving the particles
        if (framesUntilLaunch <= 0)
        {
            x[loop] += xSpeed[loop];
 
            y[loop] += ySpeed[loop];
 
            ySpeed[loop] += GRAVITY;
        }

    }
    framesUntilLaunch--;
 
    // Once a fireworks speed turns positive (i.e. at top of arc) - blow it up!
    if (ySpeed[0] > 0.0f)
    {
        for (int loop2 = 0; loop2 < FIREWORK_PARTICLES; loop2++)
        {
            // Set a random x and y speed beteen -4 and + 4
            xSpeed[loop2] = -4 + (rand() / (float)RAND_MAX) * 8;
            cout<<"boom velX "<<xSpeed[loop2]<<endl;
            ySpeed[loop2] = -4 + (rand() / (float)RAND_MAX) * 8;
            cout<<"boom velY "<<ySpeed[loop2]<<endl;	
        }
 		particleSize =1;
        cout << "Boom!" << endl;
        hasExploded = true;
    }
}
 

void Firework::updateYUV(){
	R = alpha*R;
	G = alpha*G;
	B = alpha*B;
	Y = 0.299*R+0.587*G+0.114*B;
    U = -0.169*R+-0.331*G+0.5*B+128;
    V = 0.5*R-0.419*G-0.081*B+128;
}

void Firework::explode()
{
    for (int loop = 0; loop < FIREWORK_PARTICLES; loop++)
    {
        // Dampen the horizontal speed by 1% per frame
        xSpeed[loop] *= 0.99;
 
        // Move the particle
        x[loop] += xSpeed[loop];
        y[loop] += ySpeed[loop];
 
        // Apply gravity to the particle's speed
        ySpeed[loop] += GRAVITY;
    }
 
    // Fade out the particles (alpha is stored per firework, not per particle)
    if (alpha > 0.0f)
    {
        alpha -= 0.01f;
        updateYUV();
    }
    else // Once the alpha hits zero reset the firework
    {
    	cout<<"explode init"<<endl;
        initialise();
    }
}
struct Lab2VideoGenerator::Impl {
	int t = 0;
	Firework* fw_cpu = new Firework[FIREWORKS];
};

Lab2VideoGenerator::Lab2VideoGenerator(): impl(new Impl) {
	srand(time(NULL));
}

Lab2VideoGenerator::~Lab2VideoGenerator() {}

void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};



void Lab2VideoGenerator::Generate(uint8_t *y) {
	
	
	cudaMemset(y,0, W*H);
	uint8_t* u = y+W*H;
	uint8_t* v = y+W*H+W*H/4;
	cudaMemset(u,128, W*H/4);
	cudaMemset(v,128, W*H/4);
	int particleSize = (impl->fw_cpu[0]).particleSize;
	cout<<"particleSize:  "<<particleSize<<endl;
	cout<<"xposiotion  "<<(impl->fw_cpu[0]).x[0]<<"   yposiotion  "<<(impl->fw_cpu[0]).y[0]<<endl;	
	for(int loop = 0; loop < FIREWORKS; loop++)
    {
            // Draw the point
           	Firework* fw_gpu = NULL;
	        cudaMalloc(&fw_gpu, FIREWORKS*sizeof(Firework));
	    	cudaMemcpy(fw_gpu, impl->fw_cpu, FIREWORKS*sizeof(Firework), cudaMemcpyHostToDevice);            
	       	for(int offsetx=-particleSize;offsetx<particleSize;offsetx++){
		       	for(int offsety=-particleSize;offsety<particleSize;offsety++){
		       		draw<<<FIREWORK_PARTICLES/128+1,128>>>(y,u,v,loop,offsetx,offsety,fw_gpu);
		        }
	        }
	        cudaMemcpy(impl->fw_cpu,fw_gpu,FIREWORKS*sizeof(Firework), cudaMemcpyDeviceToHost);
	        cudaFree(fw_gpu); 
	        if ((impl->fw_cpu[loop]).hasExploded == false){
	        	(impl->fw_cpu[loop]).move();
	        }
	        else{
	            (impl->fw_cpu[loop]).explode();
	        }   
    }
	++(impl->t);
	cout<<"t   "<<(impl->t)<<endl;
}
