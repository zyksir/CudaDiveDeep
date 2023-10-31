all: sgemm

sgemm: 
	cd sgemm
	make
	cd ..

clean:
	cd sgemm
	make clean