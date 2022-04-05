## Using NVIDIA GPU in Docker

Python deep learning packages (like PyTorch and TensorFlow) can leverage GPU to accelerate computation. It would be greatly beneficial for you to configure GPU in those scenarios. 

GPU acceleration is more mature on NVIDIA GPU models. The GPU hardware is exposed to your operating system by GPU driver, which only provides the basic interface to the hardware. Deep learning packages and GPU/drivers are further interleaved with **CUDA** to acceleration computation.

As our use case of GPU will be to use CUDA, **ensure that your GPU model supports CUDA** before you continue. You can check from [NVIDIA's website](https://developer.nvidia.com/cuda-gpus).

### Guideline

The normal workflow for using CUDA in operating system would be to **install GPU driver**, **install correct version of CUDA**, **install correct version of packages**. You may have been using CUDA well in your host operating system well. However, things are a little different to use CUDA in Docker.

You have to first expose the GPU to Docker.  Then, well-configure Docker images will ship with CUDA and packages themselves to leverage the GPU exposed. This means other than exposing GPU to Docker, you don't have to bother anything if you are only a Docker image user.

This tutorial covers how to expose GPU to Docker, on NVIDIA's certain GPU models. "GPU" below will specifically refer to these certain NVIDIA GPU models.

### How to expose GPU to Docker

 The fundamental will be **install the driver**. Below is the guide to expose GPU to Docker on Linux and Windows, respectively. Note that these two guides are separate.

#### On Linux

Exposing GPU to Docker is natively supported in Linux distributions. Please first check [Pre-Requisites](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#pre-requisites) before you do the following.

1. Install NVIDIA driver

   Generally you should go to [NVIDIA's download page](https://www.nvidia.com/download/index.aspx) to find proper version of driver for your system, download and run to install the driver.

   There may be other convenient ways to install driver for different Linux distributions. Try to search the tutorial for your own distribution on the Internet.

2. Install Docker

   Go to [Installation per distro](https://docs.docker.com/engine/install/) and follow your distribution's guide to install Docker on Linux.

   You may want [Manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user) so that you don't have to always run Docker as root.

3. Install `nvidia-docker2`

   Go to `nvidia-container-toolkit`'s [ Docker > Getting Started](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) and follow your distribution's guide to install `nvidia-docker2` on Linux.

4. Restart and test

   You may need to restart to let changes take effect. After that,  in your terminal,

   ```sh
   sudo docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

   should return you the driver information. Then the whole installation is successful.

#### On Windows

The mechanism of Docker on Windows is to run Docker in a backend Linux virtual machine. This backend can be either Hyper-V or WSL2. It is recommended to choose WSL2 as the backend, which is covered as below.

1. Install Windows 10 21H2 or Windows 11

   Currently the GPU passthrough is only supported on Windows 10 21H2 (or higher) or Windows 11. Make sure you have these two installed if you want to use GPU in Docker.

   Note that Windows 10 21H2 won't be delivered automatically. You may have to open **Settings > Check for Updates** to manually install the upgrade.

2. Install NVIDIA driver on Windows

   Usually you don't have to do anything since Windows has installed the NVIDIA driver automatically. 

   It seems that the driver version number of Windows 10 should be **>=470.25** (not sure). In case you may have to install or update the driver, go to [NVIDIA's download page](https://www.nvidia.com/download/index.aspx) to find proper version of driver for your system, download and run to install the driver.

   Also remember that **there is no sense to install GPU driver or CUDA in virtual machine like WSL2**. Follow on to know what to do.

3. Install WSL2

   Microsoft's webpage has illustrated how to manually install the WSL2. Go to [Manual install steps for older version](https://docs.microsoft.com/en-us/windows/wsl/install-manual) and refer to the **Install > Manual install steps for older versions** section. Its sibling section **Install > Install WSL** is also a good tutorial but is more of in a command-line way. You may refer to either as you like

   There may be many Linux distributions you can choose to install as WSL2. But ensure you install **Ubuntu or Debian** if you want to use GPU in Docker.

4. Upgrade Linux kernel of WSL2

   The WSL2 kernel version has to be **5.10.43** or higher to support GPU usage. The kernel version of the WSL2 you installed in previous step may not meet the requirement.

   Although we are to upgrade WSL2's kernel, operation is done on Windows side. Go to [Microsoft's Update Catalog](https://www.catalog.update.microsoft.com/Search.aspx?q=wsl) to find proper upgrade package. Download, unpack and execute.

5. Set up your Linux account.

   Open the WSL2 you have just installed and then it will require you to create an account.

Steps above are summarized in [Enable NVIDIA CUDA on WSL2](https://docs.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl). Please follow on. Next we will install Docker and GPU supporting package.

1. Install Docker on Windows

   It is not recommended to directly install Docker in WSL2 (though you can do so). Instead, follow [Install Docker Desktop for Windows](https://docs.docker.com/desktop/windows/install/) to install the Docker Desktop for Windows. Follow [Install Docker Desktop](https://docs.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers#install-docker-desktop) to configure your Docker's setting. Then you can run `docker` command in either Windows PowerShell or WSL2 terminal.

2. Install `nvidia-docker2` on WSL2

   Now you have to open the WSL2 distribution you have installed.  By following `nvidia-container-toolkit`'s [Docker > Getting Started > Installing on Ubuntu and Debian](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian), which are the recommended distributions to install as WSL2, type in below commands one by one in the WSL2 terminal:

   ```shell
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
      && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
      
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   ```

3. Restart and test

   You may need to restart Windows to let changes take effect. After thast, either in Windows PowerShell or WSL2 terminal,

   ```sh
   docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

   should return you the driver information. Then the whole installation is successful.

#### Troubleshooting

Using GPU in Docker involves **operating system/virtual machine**, **Docker** and [`nvidia-docker` project](https://github.com/NVIDIA/nvidia-docker). Out of these three sources that may provide you the guide, plus other legacy blogs, please refer to `nvidia-docker` project's as the most authoritative and updated.
