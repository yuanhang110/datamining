#!/usr/bin/env python
# coding: utf-8

# # Preliminaries

# ```{tip}
# The preliminaries section can be used to explain the notations and background information needed to understand the main results. Readers already have the relevant background may also jump directly to the main results.
# ```

# **Where is the book published?**

# The book is published by [GitLab pages](https://docs.gitlab.com/ee/user/project/pages/) at  
# > <https://ccha23.gitlab1.pages.cs.cityu.edu.hk/cs5483jupyter>

# The code is available at the repository  
# > <https://gitlab1.cs.cityu.edu.hk/ccha23/cs5483jupyter.git>  
# 
# The book is compiled from the jupyter notebooks under the folder [`docs`](https://gitlab1.cs.cityu.edu.hk/ccha23/cs5483jupyter/-/tree/main/docs).

# ```{seealso}
# The jupyter notebooks are paired with MyST markdown files for better version control.  See [jupytext](https://github.com/mwouts/jupytext) for details.
# ```

# **How to obtain and run the notebooks easily?**

# The simplest way is launch to the course JupyterHub server. You can click on the launch badge  
# > [![CS5483](badge.dio.svg)](https://cs5483.cs.cityu.edu.hk/hub/user-redirect/git-pull?repo=https://gitlab1.cs.cityu.edu.hk/ccha23/cs5483jupyter&urlpath=lab/tree/cs5483jupyter/docs/Abstract.ipynb&branch=main)
# 
# To launch a specific notebook page, click on the top of the notebook page:  
# > <i class="fas fa-rocket"></i> > JupyterHub

# ```{caution}
# For the first time you launch the notebooks, there should not be any existing folder `~/cs5483jupyter`, or the repository cannot be pulled to that location.
# ```

# For others with no access to the jupyterhub server, the notebooks can be opened to the Binder service:  
# > [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fgitlab1.cs.cityu.edu.hk%2Fccha23%2Fcs5483jupyter/main?urlpath=git-pull?repo%3Dhttps%3A%2F%2Fgitlab1.cs.cityu.edu.hk%2Fccha23%2Fcs5483jupyter%26branch%3Dmain%26urlpath%3Dlab%2Ftree%2Fcs5483jupyter%2Fdocs%2FAbstract.ipynb)
# 
# Specific notebook page can also be opened with the top bar:
# > <i class="fas fa-rocket"></i> > Binder

# It may take a while to load especially after the repository is updated because Binder needs to build an image of the repository before running it.

# ```{caution}
# The storage on Binder is temporary and so, to save your changes to a notebook, you should download the notebook. The server has limited computing power, and is culled after a certain period of idle time.
# ```

# **How to run the notebooks locally with more control on the resources?**

# The notebooks can also run locally (or anywhere) in a docker container:
# 
# 1. Install [docker](https://docs.docker.com/get-started/#download-and-install-docker) and run it with your [configuration](https://docs.docker.com/config/daemon/).
# 2. Install [Visual Studio Code (VSCode)](https://code.visualstudio.com/) and the extension [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
# 3. Run VSCode and click on the top bar `View > Command Palette`.
# 4. Enter `Remote-Containers: Clone Repository in Container Volume...`
# 5. Enter the repository url: `https://gitlab1.cs.cityu.edu.hk/ccha23/cs5483jupyter.git`

# Your web browser should open in a moment with a Jupyter server running at <http://localhost:8888>, where you can interact with the notebooks saved to a container volume. For the first time, it may take a while to load as docker needs to pull a docker image to run.

# ```{caution}
# Ensure port 8888 is free for use by the jupyter server.
# ```

# ```{tip}
# In case of error, you may rebuild the container with `View > Command Palette > Remote-Containers: Rebuild Container`.
# ```
