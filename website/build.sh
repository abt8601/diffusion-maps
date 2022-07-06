#!/bin/sh -e

branch_name=main

mkdir -p "website/doc/$branch_name/cpp/pdf" "website/doc/$branch_name/py"

make doc

mv build/cppdoc/html "website/doc/$branch_name/cpp"
mv build/cppdoc/latex/refman.pdf \
    "website/doc/$branch_name/cpp/pdf/diffusion-maps.pdf"
mv build/pydoc "website/doc/$branch_name/py/html"
