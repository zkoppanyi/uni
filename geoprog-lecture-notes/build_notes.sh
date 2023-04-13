# jupyter books link: https://jupyterbook.org/en/stable/start/publish.html

source /home/zoltan/env/photo/bin/activate

# # install
# pip install -U jupyter-book
# sudo apt-get install texlive-latex-extra texlive-fonts-extra texlive-xetex latexmk

# build
jupyter-book build geoprog-lecture-notes/

# generate pdf: https://jupyterbook.org/en/stable/advanced/pdf.html
# #jupyter-book build geoprog-lecture-notes/ --builder pdfhtml
# jupyter-book build mybookname/ --builder pdflatex # preferred