pandoc -f markdown_github+yaml_metadata_block+citations -t html -o D4128.html --filter pandoc-citeproc --csl=acm-sig-proceedings.csl --number-sections --toc -s -S --template=pandoc-template --include-before-body=header.html D4128.md 
