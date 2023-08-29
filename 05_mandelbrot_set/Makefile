CXX = nvcc
#TFLAG = -DPRINT_SET_DETAILS
NVFLAGS = -Wno-deprecated-declarations -gencode arch=compute_61,code=compute_61
NAME = main

$(NAME).exe: $(NAME).cu
	$(CXX) $(TFLAG) $(NVFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -f $(NAME).exe

.PHONY: run
run: $(NAME).exe
	./$<

profile: $(NAME).exe
	nsys profile --trace=cuda,osrt ./$<
