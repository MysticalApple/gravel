NVCC = nvcc
NVCCFLAGS = -arch=sm_86

OBJ = main.obj rendering.obj

gravel: $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o gravel $(OBJ) -cudart static

main.obj: main.cu rendering.h
	$(NVCC) $(NVCCFLAGS) -c main.cu

rendering.obj: rendering.cu rendering.h
	$(NVCC) $(NVCCFLAGS) -c rendering.cu

clean:
	del *.obj *.lib *.exp