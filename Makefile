NVCC = nvcc
NVCCFLAGS = -arch=sm_86 -diag-suppress 177

OBJ = main.obj rendering.obj logic.obj parsing.obj

gravel: $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o gravel $(OBJ) -cudart static -luser32 -lgdi32

main.obj: main.cu rendering.h logic.h parsing.h
	$(NVCC) $(NVCCFLAGS) -c main.cu

rendering.obj: rendering.cu rendering.h
	$(NVCC) $(NVCCFLAGS) -c rendering.cu

logic.obj: logic.cu logic.h
	$(NVCC) $(NVCCFLAGS) -c logic.cu

parsing.obj: parsing.cu parsing.h
	$(NVCC) $(NVCCFLAGS) -c parsing.cu

clean:
	del *.obj *.lib *.exp
	