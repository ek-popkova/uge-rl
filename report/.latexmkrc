# latexmk configuration: `latexmk` runs pdflatex + bibtex + pdflatex as many times as needed.
$pdf_mode = 1;                       # produce PDF with pdflatex
$out_dir  = 'build';                 # all generated files go to build/
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
$bibtex_use = 2;                     # run bibtex when needed, remove .bbl on clean
