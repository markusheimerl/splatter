CC = clang
CFLAGS = -O3 -march=native -Wall -Wextra -fopenmp
LDFLAGS = -lm -lwebp -lwebpmux -fopenmp

splatter.out: splatter.c
	$(CC) $(CFLAGS) splatter.c $(LDFLAGS) -o splatter.out

run: splatter.out
	@time ./splatter.out
clean:
	rm -f splatter.out *_splat.webp

.PHONY: run clean