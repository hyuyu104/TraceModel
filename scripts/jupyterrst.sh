#!/bin/bash

out_path="docs/source/jupyter/"
if [ -e "$out_path" ]; then
    rm -r "$out_path"
    mkdir "$out_path"
fi

for c in notebooks/*; do
    if [[ "$c" == *.ipynb ]]; then
        jupyter nbconvert --to rst $c
        mv "${c:0:${#c}-6}_files" "$out_path"
        mv "${c:0:${#c}-6}.rst" "$out_path"
    else
        cp -r $c "$out_path"
    fi
done