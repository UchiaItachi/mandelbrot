all:
	g++ -I /usr/local/include/ -L /usr/local/lib -o cpu.x -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo CPU_Mandelbrot.cpp -lglfw
	nvcc -I /usr/local/include/ -L /usr/local/lib/ -arch=sm_30 GPUrendering.cu -o gpu.x -Xlinker -framework,OpenGL,-framework,Cocoa,-framework,IOKit,-framework,CoreVideo -lglfw

clean:
	rm *.x
