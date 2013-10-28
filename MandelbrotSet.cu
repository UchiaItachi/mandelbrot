// To build this code, use
// +-------------------------------------------------------------+
// | nvcc -ccbin=g++-4.4 -arch=sm_13 MandelbrotSet.cu -lGL -lglut|
// +-------------------------------------------------------------+

// OpenGL specific headers

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

// the usual gang of C++ headers

#include <iostream>
#include <complex>
#include <cmath>
#include <cstdlib>

// window size for displaying graphics

#define WIDTH  1024
#define HEIGHT 1024

// define a class for complex numbers and their operations

class dcmplx
{
public:
    double re;   // real component
    double im;   // imaginary component

// function to calculate the magnitude or absolute value of the complex number
// this function is called from and executes on the device (GPU) 

__device__
double magnitude()
{
    return pow((re*re + im*im),0.5);
}

};

// kernel to check all points inside the specified window for membership in the set
// this function is called from the host (CPU) but executes on the device (GPU)

__global__ void Mandelbrot(double xmin, 
                           double xmax, 
                           double ymin, 
                           double ymax, 
                           int *dev_color, 
                           const int MAX_ITER)
{
    double dx = (xmax - xmin)/WIDTH;  // grid spacing along X
    double dy = (ymax - ymin)/HEIGHT; // grid spacing along Y

    // global (i,j) location
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;

    double x = xmin + (double) i*dx;   // actual x coordinate (real component)
    double y = ymin + (double) j*dy;   // actual y coordinate (imaginary component)

    dcmplx c;
    c.re = x;
    c.im = y;

    dcmplx z;
    z.re = 0.0;
    z.im = 0.0;

    // ---------------
    // z <---- z*z + c

    int iter = 0;

    while(iter<MAX_ITER)
    {
        iter++;
        dcmplx temp = z;
        z.re = temp.re*temp.re - temp.im*temp.im  +  c.re;
        z.im = 2.0*temp.re*temp.im                +  c.im;
        
        if (z.magnitude() > 2.0) break;
    }

    // the 2D array "dev_color" stores how many iterations were required for divergence
    // for points outside the Mandelbrot set, this is typically a small number
    // points inside the set do not diverge and thus iter is a large number for such points

    dev_color[i*WIDTH + j] = iter;
}

// calculate pixel colors for the current graphics window, defined by the
// minimum and maximum X and Y coordinates

void showMandelbrot(double xmin, double xmax, double ymin, double ymax)
{
    //----------------------------------------------------------------
    //  Use GPU for calculating the 2D array of "iterations to escape"
    //----------------------------------------------------------------

    // problem parameters
    const int MAX_ITER = 200;

    // 2D array size on the host is identical to the window size in pixels
    const int NX = WIDTH;
    const int NY = HEIGHT;

    int *iters = new int[WIDTH*HEIGHT];

    double dx = (xmax - xmin)/NX; // grid spacing along X
    double dy = (ymax - ymin)/NY; // grid spacing along Y

    // allocate device pointers of the same size as the host
    int *d_color;
    cudaMalloc((void **) &d_color, WIDTH * HEIGHT * sizeof(int));

    const int tx = 16;
    const int ty = 16;

    dim3 threads(tx, ty, 1);

    const int bx = ceil( (float) WIDTH  / (float) tx );
    const int by = ceil( (float) HEIGHT / (float) ty );

    dim3 blocks( bx, by, 1);

    Mandelbrot<<<blocks,threads>>>(xmin, xmax, ymin, ymax, d_color, MAX_ITER);

    // copy results from the GPU to the CPU for plotting
    cudaMemcpy(iters,d_color,WIDTH*HEIGHT*sizeof(int),cudaMemcpyDeviceToHost);

    //--------------------------------
    //  Render the image using OpenGL
    //--------------------------------

    // select background color to be white
    // R = 1, G = 1, B = 1, alpha = 0
    glClearColor (1.0, 1.0, 1.0, 0.0);

    // initialize viewing values
    glMatrixMode(GL_PROJECTION);

    // replace current matrix with the identity matrix
    glLoadIdentity();

    // set clipping planes in the X-Y-Z coordinate system
    glOrtho(xmin,xmax,ymin,ymax, -1.0, 1.0);

    // clear all pixels
    glClear (GL_COLOR_BUFFER_BIT);

    // assign pixel color based on the number of iterations - Red Green Blue (RGB) 

    for(int i = 0; i < NX; i++) {
        for(int j = 0; j < NY; j++) {

            double x = xmin + i*dx;   // actual x coordinate
            double y = ymin + j*dy;   // actual y coordinate

            int VAL = iters[i*WIDTH + j];

            if(VAL==MAX_ITER)
            {
                glColor3f(0,0,0);   // black
            }
            else
            {
                // ratio of iterations required to escape
                // the higher this value, the closer the point is to the set
                float frac = (float) VAL / MAX_ITER;

                if(frac<=0.5)
                {
                    // yellow to blue transition
                    glColor3f(2*frac,2*frac,1-2*frac);
                }
                else
                {
                    // red to yellow transition
                    glColor3f(1,2-2*frac,0);
                }
            }
            glRectf (x, y,x+dx,y+dy);
      }
    }

    glColor3f(1,1,1);
    glRectf(xmin+(xmax-xmin)/2-1.95*dx,ymin,xmin+(xmax-xmin)/2+1.95*dx,ymax);
    glRectf(xmin,ymin+(ymax-ymin)/2-1.95*dy,xmax,ymin+(ymax-ymin)/2+1.95*dy);

    glFlush ();
}

// Entry point for the display routine

void display(void)
{
    // specify initial window size in the X-Y plane
    double xmin = -2, xmax = 1, ymin = -1.5, ymax = 1.5;

    int choice;  // user selection of appropriate quadrant
                 //
                 //        |
                 //    1   |   2
                 //        |
                 // -------+-------
                 //        |
                 //    3   |   4
                 //        |

    // infinite loop until user kills this process
    while(true)
    {
        // display the Mandelbrot set in (xmin,ymin)-(xmax,ymax)
        showMandelbrot(xmin,xmax,ymin,ymax);

        // ask user for selecting a region for further zoom-in
        std::cout << "Zoom in to <1,2,3,4>: ";
        std::cin >> choice;

        // update display limits based on user choice
        switch (choice) {
            case 0:
                return;
            case 1:
                xmax = xmin + (xmax - xmin)/2; 
                ymin = ymin + (ymax - ymin)/2;
                break;
            case 2:
                xmin = xmin + (xmax - xmin)/2; 
                ymin = ymin + (ymax - ymin)/2;
                break;
            case 3:
                xmax = xmin + (xmax - xmin)/2; 
                ymax = ymin + (ymax - ymin)/2;
                break;
            case 4:
                xmin = xmin + (xmax - xmin)/2; 
                ymax = ymin + (ymax - ymin)/2;
                break;
        }
    }
}

int main(int argc, char* argv[])
{
    //--------------------------------
    //   Create a WINDOW using GLUT
    //--------------------------------

    // launch the GLUT runtime
    glutInit(&argc, argv);

    // set the window's display mode
    glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB);

    // set the windows width and height
    glutInitWindowSize (WIDTH, HEIGHT);

    // location of top left corner of window
    glutInitWindowPosition (0, 0);      

    // create a window with the specified title
    glutCreateWindow ("Mandelbrot Set");

    //---------------------------------------------
    // Display something in the window using OpenGL
    //---------------------------------------------

    // pass a function pointer
    glutDisplayFunc(display);

    // GLUT processing loop continues until the application terminates
    glutMainLoop();

    return 0;
}
