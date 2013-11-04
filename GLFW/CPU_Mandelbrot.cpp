// OpenGL specific headers

#include <GLFW/glfw3.h>

// the usual gang of C++ headers

#include <iostream>
#include <complex>
#include <cmath>
#include <cstdlib>

// this function checks if a point (x,y) is a member of the Mandelbrot set
// it returns the number of iterations it takes for this point to escape from the set
// if (x,y) is inside the set, it will not escape even after the maximum number of iterations
// and this function will take a long time to compute this and return the maximum iterations

int Mandelbrot_Member(double x, double y, const int MAX_ITER)
{
   typedef std::complex <double> dcmplx;   // define a new data type, the double-precision complex number

   dcmplx c(x,y);
   dcmplx z(0.0,0.0);

   int iter = 0;

   while(iter<MAX_ITER)
   {
      iter++;
      z = z*z + c;
      if (abs(z) > 2.0) break;   // (x,y) is outside the set, quick exit from this loop
   }
   return iter;
}
 
// calculate pixel colors for the current graphics window, defined by the
// minimum and maximum X and Y coordinates

void showMandelbrot(GLFWwindow *window, int WIDTH, int HEIGHT, double xmin, double xmax, double ymin, double ymax)
{
    //--------------------------------
    //  OpenGL initialization stuff 
    //--------------------------------
    glfwGetFramebufferSize(window, &WIDTH, &HEIGHT);

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

    // problem parameters
    const int MAX_ITER = 200;

    // 2D array size is identical to the window size in pixels
    const int NX = WIDTH;
    const int NY = HEIGHT;

    int *iters = new int[WIDTH*HEIGHT];

    double dx = (xmax - xmin)/NX; // grid spacing along X
    double dy = (ymax - ymin)/NY; // grid spacing along Y

    // fill the 2D array with the "iter" parameter, which 
    // represents the number of iterations it takes for the
    // point to escape from the set

    for(int i = 0; i < NX; i++) {
      for(int j = 0; j < NY; j++) {

        double x = xmin + i*dx;   // actual x coordinate
        double y = ymin + j*dy;   // actual y coordinate

        // calculate iterations to escape
        iters[i*WIDTH + j] = Mandelbrot_Member(x,y,MAX_ITER);
      }
    }
    
    // assign color based on the number of iterations - Red Green Blue (RGB) 

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

    // swap front and back buffers
    glfwSwapBuffers(window);

    // poll for and processs events
    glfwPollEvents();
}

// Entry point for the display routine

void display(GLFWwindow *window, int WIDTH, int HEIGHT)
{
    // specify initial window size in the X-Y plane
    double xmin = -2, xmax = 1, ymin = -1.5, ymax = 1.5;

    int choice;  // user selection of appropriate quadrant
                 //
                 //        |
                 //    1   |   2
                 //        |
                 //   -----+-----
                 //        |
                 //    3   |   4
                 //        |

    // infinite loop until user kills this process
    while(true)
    {
        // display the Mandelbrot set in (xmin,ymin)-(xmax,ymax)
        showMandelbrot(window, WIDTH, HEIGHT, xmin,xmax,ymin,ymax);

        // ask user for selecting a region for further zoom-in
        std::cout << "Zoom in to <1,2,3,4>: ";
        std::cin >> choice;

        // update display limits based on user choice
        switch (choice) {
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

    //---------------------------------------
    // Loop until the user closes the window
    //---------------------------------------

    while(!glfwWindowShouldClose(window))
    {
        // render function
        display(window, WIDTH, HEIGHT);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
