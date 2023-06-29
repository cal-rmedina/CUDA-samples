export PATH := /usr/local/cuda/bin:$(PATH)
CXX = nvcc
#TFLAG=-DPRINT_PI
NVFLAGS = -Wno-deprecated-declarations -gencode arch=compute_61,code=compute_61
NAME =main

$(NAME).exe: $(NAME).cu
	$(CXX) $(TFLAG) $(NVFLAGS) $< -o $@

.PHONY: clean

clean:
	rm -f $(NAME).exe

.PHONY: run

run: $(NAME).exe
	./$<

profile: $(NAME).exe
	nsys profile --trace=nvtx,cuda,osrt ./$<
