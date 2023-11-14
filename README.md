# NIF-only branch!

This branch is nothing to do with path tracing: it strips out everything except the neural image field rendering to build a remote real-time NIF rendering/inference demo. This probably belongs in its own repository but it was easier to rapidly prototype by branchign from a working program.

The demo consists of two parts: a cloud based render server and a local remote viewer. The instructions below are a quick start guide to build and run this demo.

## Graphcloud instructions

These instructions are tested on Graphcloud but should work on other cloud services where IPUs are available and you are free to tunnel/forward ports from the container.

### Build a Docker image for Poplar SDK 3.3.0

Setup will be much easier if you build and launch the tested docker image:

```
git clone https://github.com/markp-gc/docker-files.git

export CONTAINER_SSH_PORT=2023

docker build -t $USER/poplar_3.3_dev --build-arg UNAME=$USER  \
--build-arg UID=$(id -u) --build-arg GID=$(id -g) \
--build-arg CUSTOM_SSH_PORT=$CONTAINER_SSH_PORT  \
docker-files/graphcore/poplar_dev
```

### Launch the container

Source your Poplar SDK so that you can use gc-docker command to launch your container (within the container you will be using its SDK which is fixed at 3.3.0 to ensure compatibility if someone reconfigures the shared host system). The SDK should be pre-installed in /opt but check the path for your system:

```
source /opt/gc/poplar_sdk-ubuntu_20_04-3.2.0+1277-7cd8ade3cd/enable
gc-docker -- --detach -it --rm --name "$USER"_docker -v/nethome/$USER:/home/$USER --tmpfs /tmp/exec $USER/poplar_3.3_dev
```

You should now see the running container listed when you run the command docker ps and you should be able to attach to it to get a shell in the container: `docker attach "$USER"_docker`

:warning: Note that the gc-docker command above has mounted your home directory as the home directory in your container. This could break your home in the base system in theory (but in practice the convenience outweighs the small risk). Just be aware that changes to your home directory in the container will be reflected in the base home directory (which might have a different Ubuntu version e.g. 18 instead of 22).

### Clone and build the prototype render server code

The render server performs Neural Image Field (NIF) inference with the ability to connect a real-time viewer over the netowrk. It was hacked on top of the old path trace render preview user-interface in the nif_only branch (in case you are wonderign why the repo name does not match the demo):

```
git clone --recursive --branch nif_only https://github.com/markp-gc/ipu_path_trace.git
mkdir -p ipu_path_trace/build
cd ipu_path_trace/build/
cmake -GNinja ..
ninja -j100
```

This should configure and build successfully as the container already has everything you need installed.

### Build the remote user interface application

The remote-UI runs on your local laptop or workstation. (The setup here will be less straightforward than for the server depending on your machine’s configuration). There is a list of dependencies in that repo's [README](https://github.com/markp-gc/remote_render_ui#dependencies)

It has been tested on Ubuntu 18 and Mac OSX 11.6 it should be possible to build on other systems if you can satisfy the dependencies.

Clone the repo (we also checkout a known good commit here):

```
git clone --recursive  https://github.com/markp-gc/remote_render_ui.git
mkdir -p remote_render_ui/build
cd remote_render_ui/build/
git checkout 3f1dc9a
cmake -GNinja ..
ninja -j100
```

Unless your machine happened to be configured perfectly already then this will probably not configure or build first time so please carefully review all the dependencies, especially for the videolib submodule as the FFMPEG version must be 4.4: [videolib instructions](https://github.com/markp-gc/videolib#installing-dependencies).

### Create a tunnel into your container

The most reliable way to achieve this is to use VS Code to forward the port for you. Setup VS code for remote development in your container as follows:
- Install VS Code on your local laptop/workstation: [get VS Code](https://code.visualstudio.com)
- Install the remote-development pack (follow this link and click install) [get the remote development extension pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
- Connect VS Code to your cloud server via SSH following the instructions here: [connect VS code via SSH](https://code.visualstudio.com/docs/remote/ssh)

The VS Code remote-development package supports development in containers: once you are connected to the remote machine via SSH you can see and attach to running containers by clicking the Remote Explorer icon in the left panel, then select Containers from the drop down list at the top. You should be able to see the container that you created is in the list and in the running state. Click the arrow icon to attach to your container in the current window:

![image](https://github.com/markp-gc/ipu_path_trace/assets/65598182/d7813be1-d72a-4482-ba51-b338a77276c8)

Once you are in the container find the ports panel and forward a port number of your choice (> 1024):

![image](https://github.com/markp-gc/ipu_path_trace/assets/65598182/7405e3bc-5f39-4775-9b12-f1fdf7f031df)

### Download/upload a NIF asset to your cloud machine

Whilst you can train your own NIF for use in the demo we recommend you check it works with a pre-trained asset first. There is a trained neural image field included in the git repo. This model is quite large for this type of neural network: 8 layers and hidden size 320 and requires approximately 1M FLOPs per sample. The network architecture is the one from: [Towards Neural path Tracing in SRAM](https://arxiv.org/abs/2305.20061).

MLPs used in neural real-time rendering on desktop GPUs typically have a FLOP rates less than 100K FLOPS per sample (this keeps models small enough that their weights can remain in cache for longer).

### Run the demo

On the cloud machine in your docker container launch the server:

```
./ipu_trace --assets ../nif_models/urban_alley_01_4k_fp16_yuv/assets.extra/ -w 1104 -h 1008 -o out.pgm --ui-port 9000 --save-exe test
```

This resolution was chosen to utilise all IPU threads (the number of pixels is a multiple of 6 x 1472). On subsequent runs you can replace --save-exe with --load-exe in order to skip compilation (unless you modify and need to recompile the Poplar compute graph).

On your local laptop or workstation launch the remote user interface:

```
./remote-ui -w 1600 -h 1100 --port 9000
```

:warning: The above commands assume that you set up your SSH tunnel so that localhost is directed into the container, and that the locally forwarded port is the same as the remote port (VS Code might choose another local port if the one you asked for was in use).

The user interface should launch and connect to render the server. You should be able to rotate the view using the control wheel in the top right of the control panel, zoom using the FOV slider, and change exposure and gamma with their sliders (this is a high dynamic range neural image field).

With a stable internet connection you should see a frame rate of about 50 frames per second and latency that is good enough for smooth interaction. Rendering one frame here involves performing neural network inference on a batch of 1104 x 1008 inputs which at 50 frames per second is a throughput of 55.6 million pixels per second (or 57.1 TFLOPs/sec).

### Train your own neural image field

The neural environment light uses a neural image field (NIF) network. These are MLP based image approximators and are trained using Graphcore's NIF implementation: [NIF Training Scripts](https://github.com/graphcore/examples/tree/master/vision/neural_image_fields/tensorflow2). Before you start a training run you will need to source an equirectangular-projection HDRI image (e.g. those found here are suitable: [HDRIs](https://polyhaven.com/hdris)). Download a 2k or 4k image and pass it to the NIF traning script('--input'). You can train a smaller NIF than we used above (i.e. "GPU sized") and see how the framerate improves:

```
git clone https://github.com/graphcore/examples.git
cd examples/vision/neural_image_fields/tensorflow2
pip install -r requirements.txt
python3 train_nif.py --train-samples 8000000 --epochs 1000 --callback-period 100 --fp16 --loss-scale 16384 --color-space yuv --layer-count 2 --layer-size 128 --batch-size 1024 --callback-period 100 --embedding-dimension 12 --input input_hdri.exr --model nif_models/output_nif_fp16_yuv
```

The trained keras model contains a subfolder called assets.extra, give that path to the server using the --assets command line option.
