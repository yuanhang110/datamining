---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"tags": []}

# Modifying Contents

+++

```{tip}
JupyterBook uses the MyST syntax which is nicely summarized by [this cheat sheet](https://jupyterbook.org/reference/cheatsheet.html).
```

```{code-cell} ipython3
:tags: [remove-cell]

from IPython.display import Code
```

In the Jupyter interface, open the terminal (`File > New > Terminal`) and run  
```bash
./build.sh
```
under the repository root. Answer `y` or simply press enter when prompted.

+++

To preview the book website,
- start the http server (`File > New Launcher > www`) and 
- navigate to `www/cs5483jupyter`.

+++

```{tip}
After rebuilding the book, you can refresh an existing preview page to update its content.
```

+++

## Citations

+++

To cite some references such as the main textbooks {cite}`Han11, Witten11`, write
```markdown
  {cite}`Han11, Witten11`
```
and add the details of the references to the BibTeX file `ref.bib`:

```{code-cell} ipython3
:tags: [remove-input]

Code('ref.bib')
```

`````{note}

For the above to work, the bibliography is added to the [References](References.ipynb) notebook with

````markdown
```{bibliography}
```
````

`````

+++

```{seealso}
- [JupyterBook docs](https://jupyterbook.org/content/citations.html)
- [sphinxcontrib-bibtex docs](https://sphinxcontrib-bibtex.readthedocs.io)
```

+++ {"tags": []}

## Equations

+++

Eqn {eq}`MI` below defines the mutual information

$$
\begin{align}
I(\R{X} \wedge \R{Y}) &\coloneqq D(P_{\R{X},\R{Y}}\| P_{\R{X}}\times P_{\R{Y}})\\
&=  \int_{\mc{X}\times \mc{Y}} d P_{\R{X},\R{Y}} \,\log \frac{d P_{\R{X},\R{Y}}}{d(P_{\R{X}}\times P_{\R{Y}})}.
\end{align}
$$ (MI)

It uses LaTeX commands supported by Mathjax3:

```
$$
\begin{align}
I(\R{X} \wedge \R{Y}) &\coloneqq D(P_{\R{X},\R{Y}}\| P_{\R{X}}\times P_{\R{Y}})\\
&= \int_{\mc{X}\times \mc{Y}} d P_{\R{X},\R{Y}} \,\log \frac{d P_{\R{X},\R{Y}}}{d(P_{\R{X}}\times P_{\R{Y}})}.
\end{align}
$$ (MI)
```

+++

```{seealso}
- [JupyterBook docs](https://jupyterbook.org/content/math.html)
- [MathJax3 docs](https://docs.mathjax.org/en/latest/input/tex/macros/index.html)
```

+++

## Diagrams

+++

{numref}`fig:DT` below shows a decision tree that corresponds to an AND Gate:

+++

```{figure} images/DT.dio.svg 
---
name: fig:DT
---
Decision Tree
```

The figure is create with the directive `figure`:

````
```{figure} images/DT.dio.svg 
---
name: fig:DT
---
Decision Tree **test**
```
````

+++

The decision tree image <images/DT.dio.svg> is an svg file editable by the Diagram app. To create such a diagram:  
- Click `File > New Launcher > Diagram` to create a `.dio` diagram file to draw in the Diagram app.
- Rename the file to `.dio.svg` so that is can be included into a notebook as an `svg` image.

+++

```{tip}
You can import [mermaid diagrams](https://mermaid-js.github.io/mermaid/) in the Diagram app by `Insert > Advanced >  Mermaid...`.
```
