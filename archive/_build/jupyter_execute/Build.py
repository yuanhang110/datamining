#!/usr/bin/env python
# coding: utf-8

# # Building the book

# In[1]:


from IPython.display import Code


# ## Compilation

# In the Jupyter interface, open the terminal (`File > New > Terminal`) and run  
# ```bash
# ./build.sh
# ```
# under the *repository root*. Answer `y` or simply press enter when prompted.

# To preview the book website,
# - start the HTTP server (`File > New Launcher > www`) and 
# - navigate to `www/cs5483jupyter`.

# The source code of `build.sh` is shown below:

# In[2]:


Code('../build.sh')


# - `jupyter-book clean docs` to empty previously built book.
# - `jupyter-book build docs` to build the book.
# - `rm -rf \"${JUPYTER_SERVER_ROOT}/www/cs5483jupyter\" && mkdir --parents \$_ && cp -rf docs/_build/html/* \$_` moves the book to the folder `www/cs5483jupyter` under the Jupyter Server Root for the HTTP server to serve the book website.

# ```{tip}
# - You can run `yes | ./build.sh` to automatically answer `y` when prompted.
# - After rebuilding the book, you can refresh an existing preview page to update its content.
# ```

# ```{seealso}
# - [JupyterBook docs on CLI reference](https://jupyterbook.org/reference/cli.html)
# ```

# ## Configuration

# If you use this book as a template in your own repository to host the book, you should edit the configuration file `_config.yml` under the `docs` folder:

# In[3]:


Code('_config.yml')


# In particular, you should change the repository url is specified by the lines:
# 
# ```yaml
# repository:
#   url: "https://github.com/ccha23/cs5483jupyter"  # Online location of your book
# ```
# 
# to your repository while retaining `github.com` as the base url, i.e.,
# 
# ```yaml
# repository:
#   url: "https://github.com/{repo path}"  # Online location of your book
# ```
# 
# where the repository is actually hosted at `https://gitlab1.cs.cityu.edu.hk/{repo path}`.

# ```{important}
# This is a hack to [support GitLab pages](https://github.com/executablebooks/jupyter-book/issues/1416):
# - Using `github.com` tricks `jupyter-book` to enable the launch buttons and other repository-related buttons.
# - The javascript `github2gitlab.js` under the folder `_static` then replaces `github.com` in the buttons by `gitlab1.cs.cityu.edu.hk`.
# 
# If you host your repository in other git servers, you should also edit `github2gitlab.js` to substitute `gitlab1.cs.cityu.edu.hk` with the appropriate server name.
# ```

# In[4]:


Code('_static/github2gitlab.js')


# ```{seealso}
# - [JupyterBook docs on Config reference](https://jupyterbook.org/customize/config.html)
# ```
