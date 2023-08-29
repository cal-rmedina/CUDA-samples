export PATH := /usr/local/cuda/bin:$(PATH)
CXX = nvcc
NVFLAGS = -Wno-deprecated-declarations -gencode arch=compute_61,code=compute_61
TFLAG=-DPRINT_1ST_STREAM
STREAM_FLAG=--default-stream per-thread
NAME =main

$(NAME).exe: $(NAME).cu
	$(CXX) $(TFLAG) $(STREAM_FLAG) $(NVFLAGS) $< -o $@

.PHONY: clean

clean:
	rm -f $(NAME).exe

.PHONY: run

run: $(NAME).exe
	./$<
profile: $(NAME).exe
	nsys profile --trace=cuda,osrt ./$<
