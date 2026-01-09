CC = clang
CFLAGS = -O3 -march=native -Wall -Wextra -fopenmp
LDFLAGS = -lm -lwebp -lwebpmux -fopenmp

splatter.out: splatter.c
	$(CC) $(CFLAGS) splatter.c $(LDFLAGS) -o splatter.out

run: splatter.out
	@time ./splatter.out
clean:
	rm -f splatter.out *_splat.webp

stage0.out: stage0_gradient_test.c
	$(CC) $(CFLAGS) -fplugin=$(HOME)/Documents/Enzyme/enzyme/build/Enzyme/ClangEnzyme-18.so \
	    stage0_gradient_test.c -lm -o stage0.out

test_gradient: stage0.out
	./stage0.out

stage1.out: stage1_optimize.c
	$(CC) $(CFLAGS) -fplugin=$(HOME)/Documents/Enzyme/enzyme/build/Enzyme/ClangEnzyme-18.so \
	    stage1_optimize.c -lm -o stage1.out

optimize: stage1.out
	./stage1.out

stage2.out: stage2_multiple_pixels.c
	$(CC) $(CFLAGS) -fplugin=$(HOME)/Documents/Enzyme/enzyme/build/Enzyme/ClangEnzyme-18.so \
	    stage2_multiple_pixels.c -lm -o stage2.out

fit_patch: stage2.out
	./stage2.out

stage3.out: stage3_real_image.c
	$(CC) $(CFLAGS) -fplugin=$(HOME)/Documents/Enzyme/enzyme/build/Enzyme/ClangEnzyme-18.so \
	    stage3_real_image.c -lm -lpng -o stage3.out

fit_real: stage3.out
	./stage3.out

stage4.out: stage4_rgb_color.c
	$(CC) $(CFLAGS) -fplugin=$(HOME)/Documents/Enzyme/enzyme/build/Enzyme/ClangEnzyme-18.so \
	    stage4_rgb_color.c -lm -lpng -o stage4.out

fit_rgb: stage4.out
	./stage4.out

stage5.out: stage5_multiple_gaussians.c
	$(CC) $(CFLAGS) -fplugin=$(HOME)/Documents/Enzyme/enzyme/build/Enzyme/ClangEnzyme-18.so \
	    stage5_multiple_gaussians.c -lm -lpng -o stage5.out

fit_multi: stage5.out
	./stage5.out

stage6.out: stage6_two_views.c
	$(CC) $(CFLAGS) -fplugin=$(HOME)/Documents/Enzyme/enzyme/build/Enzyme/ClangEnzyme-18.so \
	    stage6_two_views.c -lm -lpng -ljson-c -o stage6.out

fit_3d: stage6.out
	./stage6.out

stage7.out: stage7_manual_gradients.c
	$(CC) $(CFLAGS) stage7_manual_gradients.c -lm -lpng -ljson-c -o stage7.out

manual_grad: stage7.out
	./stage7.out

stage8.out: stage8_multiple_pixels.c
	$(CC) $(CFLAGS) stage8_multiple_pixels.c -lm -lpng -ljson-c -o stage8.out

multi_pixel: stage8.out
	./stage8.out

stage9.out: stage9_multiple_gaussians.c
	$(CC) $(CFLAGS) stage9_multiple_gaussians.c -lm -lpng -ljson-c -o stage9.out

multi_gauss: stage9.out
	./stage9.out

.PHONY: run clean