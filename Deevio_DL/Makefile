DOCKER=docker
DATA_DIR=/workspace
VOLUMES=-v $(PWD):/workspace/src
DGX_OPTS=--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864

build:
	$(DOCKER) build --rm -t deevio -f Dockerfile .

run:
	$(DOCKER) run --rm -it -p 5000:5000 --name deevio $(VOLUMES) $(DGX_OPTS) deevio
