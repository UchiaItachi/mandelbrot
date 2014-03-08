// GLFW header

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLFW/glfw3.h>

// CUDA - OpenGL interoperability

#define GL_GLEXT_PROTOTYPES
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// the usual gang of C++ headers

#include <iostream>
#include <complex>
#include <cmath>
#include <cstdlib>

// define a class for complex numbers and their operations

class dcmplx
{
public:
    double re;   // real component
    double im;   // imaginary component

// calculate the magnitude or absolute value of the complex number 
// this function is called from and executes on the device (GPU) 

__device__ double magnitude()
{
    return pow((re*re + im*im),0.5);
}

};

// kernel to check all points inside the specified window for 
// membership in the set and calculate an appropriate pixel color 
// for each point

__global__ void calculateMandelbrot(const int WIDTH,
                                    const int HEIGHT,
                                    double xmin,
                                    double xmax,
                                    double ymin,
                                    double ymax,
                                    uchar4* ptr,
                                    const int MAX_ITER)
{
    double dx = (xmax - xmin)/WIDTH;  // grid spacing along X
    double dy = (ymax - ymin)/HEIGHT; // grid spacing along Y

    // global (i,j) location handled by this thread
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;

    // out-of-bounds threads return without doing enything
    if ((i >= WIDTH) || (j >= HEIGHT)) return;

    // offset using row-major ordering
    int offset = i + WIDTH*j;

    // calculate (x,y) potition
    double x = xmin + (double) i*dx;   // actual x coordinate (real part)
    double y = ymin + (double) j*dy;   // actual y coordinate (imaginary part)

    // carry out the iterative check
    // z <---- z*z + c
    dcmplx c;
    c.re = x;
    c.im = y;

    dcmplx z;
    z.re = 0.0;
    z.im = 0.0;

    int iter = 0;

    while(iter<MAX_ITER)
    {
        iter++;
        dcmplx temp = z;
        z.re = temp.re*temp.re - temp.im*temp.im  +  c.re;
        z.im = 2.0*temp.re*temp.im                +  c.im;
        
        if (z.magnitude() > 2.0) break;
    }

    // "iter" now stores how many iterations were required for divergence
    // for points outside the Mandelbrot set, this is typically a small number
    // points inside the set do not diverge and thus iter is a large number
    // assign pixel color based on the number of iterations

    float R, G, B;

    if(iter==MAX_ITER)
    {
        // this point is inside the Mandelbrot set. Paint it black.
        R = 0;
        G = 0;
        B = 0;
    }
    else
    {
        // ratio of iterations required to escape
        // the higher this value, the closer the point is to the set
        float frac = (float) iter / MAX_ITER;

        if(frac<=0.5)
        {
            // yellow to blue transition
            R = 2*frac;
            G = 2*frac;
            B = 1 - 2*frac;
        }
        else
        {
            // red to yellow transition
            R = 1;
            G = 2 - 2*frac;
            B = 0;
        }
    }

    // convert pixel color from float(0-1) to int(0-255)
    // (unsigned char is an eight bit integer)
    //
    // 0000 0000      0
    // 0000 0001      1
    // 0000 0010      2
    // 0000 0011      3
    // 0000 0100      4
    //     .          .
    //     .          .
    //     .          .
    // 1111 1111    255

    ptr[offset].x = (int) 255*R;
    ptr[offset].y = (int) 255*G;
    ptr[offset].z = (int) 255*B;
}

// minimum and maximum X and Y coordinates

void showMandelbrot(GLFWwindow *window,
                           int WIDTH,
                           int HEIGHT,
                           double xmin,
                           double xmax,
                           double ymin,
                           double ymax)
{
    //----------------------------------------------------------------
    //  Use GPU for calculating the 2D array of "iterations to escape"
    //----------------------------------------------------------------

    // problem parameters
    const int MAX_ITER = 200;

    // create variables that will be shared between OpenGL and CUDA device
    GLuint bufferObj;
    cudaGraphicsResource *resource;

    cudaDeviceProp prop;
    int device;

    memset(&prop, 0, sizeof(cudaDeviceProp));

    // choose a CUDA capable device that is at least compute 1.3
    prop.major = 1;
    prop.minor = 3;
    cudaChooseDevice( &device, &prop);

    // tell the runtime that we intend to use this device for CUDA and OpenGL
    cudaGLSetGLDevice(device);

    // generate a pixel buffer object (PBO)
    glGenBuffers(1, &bufferObj);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, 
                 WIDTH * HEIGHT * 4, 
                 NULL, 
                 GL_DYNAMIC_DRAW_ARB);

    // register "bufferObj" with the CUDA runtime as a graphics resource
    cudaGraphicsGLRegisterBuffer(&resource, 
                                 bufferObj, 
                                 cudaGraphicsMapFlagsNone);

    // create a pointer in device memory for CUDA to use the buffer object
    uchar4* devPtr;
    size_t size;

    cudaGraphicsMapResources(1, &resource, NULL);
    cudaGraphicsResourceGetMappedPointer( (void**) &devPtr, &size, resource);

    // launch CUDA kernel to populate the device buffer
    const int tx = 32;
    const int ty = 32;

    dim3 threads(tx, ty, 1);

    const int bx = ceil( (float) WIDTH  / (float) tx );
    const int by = ceil( (float) HEIGHT / (float) ty );

    dim3 blocks( bx, by, 1);

    // invoke CUDA kernel to calculate pixel colors
    calculateMandelbrot<<<blocks,threads>>>(WIDTH, HEIGHT, 
                                            xmin, xmax, 
                                            ymin, ymax, 
                                            devPtr, MAX_ITER);

    // make sure CUDA kernel is finished before plotting the results
    cudaGraphicsUnmapResources(1, &resource, NULL);

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

    // render pixel data from buffer already in GPU memory
    glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    // divide rendered image into 4 quads separated by thick white lines
    glColor3f(255,255,255); // white
    double dx = (xmax - xmin)/WIDTH;  // grid spacing along X
    double dy = (ymax - ymin)/HEIGHT; // grid spacing along Y
    glRectf(xmin+(xmax-xmin)/2-1.95*dx,ymin,xmin+(xmax-xmin)/2+1.95*dx,ymax);
    glRectf(xmin,ymin+(ymax-ymin)/2-1.95*dy,xmax,ymin+(ymax-ymin)/2+1.95*dy);
}

int main(int argc, char* argv[])
{
    //--------------------------------
    //   Create a WINDOW using GLFW
    //--------------------------------

    GLFWwindow *window;

    // initialize the library
    if(!glfwInit())
        return -1;

    // window size for displaying graphics
    int WIDTH  = 800;
    int HEIGHT = 800;

    // set the window's display mode
    window = glfwCreateWindow(WIDTH, HEIGHT, "Mandelbrot Set", NULL, NULL);
    if(!window) 
    {
        glfwTerminate();
	return -1;
    }

    // make the windows context current
    glfwMakeContextCurrent(window);

    // user selection of appropriate quadrant for zooming in
    int choice;  

    std::cout << " +------+------+ " << std::endl;
    std::cout << " |      |      | " << std::endl;
    std::cout << " |  1   |   2  | " << std::endl;
    std::cout << " |      |      | " << std::endl;
    std::cout << " +------+------+ " << std::endl;
    std::cout << " |      |      | " << std::endl;
    std::cout << " |  3   |   4  | " << std::endl;
    std::cout << " |      |      | " << std::endl;
    std::cout << " +------+------+ " << std::endl;

    // specify initial window size in the X-Y plane
    double xmin = -2, xmax = 1, ymin = -1.5, ymax = 1.5;

    //---------------------------------------
    // Loop until the user closes the window
    //---------------------------------------

    while(!glfwWindowShouldClose(window))
    {
        // display the Mandelbrot set in (xmin,ymin)-(xmax,ymax)
        showMandelbrot(window, WIDTH, HEIGHT, xmin,xmax,ymin,ymax);

        // swap front and back buffers
        glfwSwapBuffers(window);

        // poll for and processs events
        glfwPollEvents();

        // ask user for selecting a region for further zoom-in
        std::cout << "Zoom in to <1, 2, 3, 4> [0 to quit]:";
        std::cin >> choice;

        // update display limits based on user choice
        switch (choice) {
            case 0:
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
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

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
