CC = clang
CFLAGS = -O3 -march=native -Wall -Wextra -fopenmp
LDFLAGS = -lm -lwebp -lwebpmux -fopenmp

splatter: splatter.c
	$(CC) $(CFLAGS) splatter.c $(LDFLAGS) -o splatter

run: splatter
	@time ./splatter

clean:
	rm -f splatter *.webp

.PHONY: run clean