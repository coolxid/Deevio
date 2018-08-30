# Deevio
# --- To Train Deep Learning Model ---
1) Download Data
2) Folder Structure to follow:
	Data
	|_______train
	|		|___________bad
	|		|___good	|___image_bad.jpeg		
	|		|___image_good.jpeg
	|
	|___val
		|___________bad
		|___good	|___image_bad.jpeg		
			|___image_good.jpeg	

3) python main_CNN.py

# --- To Train Deep Learning Model ---
1) Download Data
2) Folder Structure to follow:
	Data
	|___________bad
	|___good	|___image_bad.jpeg		
	|___image_good.jpeg


3) python main_hog.py


# --- To Execute Hand-Crafted Feature Machine Learning Model ---
cd Deevio_ML
make build	"To build the docker"
make run	"To run the docker"

# --- To Execute Deep Learning Model ---
cd Deevio_DL
make build	"To build the docker"
make run	"To run the docker"

# --- To Test an example ---
docker ip: 0.0.0.0
curl http://0.0.0.0:5000/predict?image=https://st3.depositphotos.com/5376016/12932/i/950/depositphotos_129320790-stock-photo-single-crooked-and-rusty-nail.jpg



