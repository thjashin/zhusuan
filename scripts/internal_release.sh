#!/bin/bash

# internal doc release
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MAIN_DIR="$(dirname $SCRIPT_DIR)"
DOC_DIR=$MAIN_DIR/docs
cd $DOC_DIR
make clean
make api
make html
scp -r ./_build/html/* juncluster3:~/public_html/zhusuan/
cd $MAIN_DIR

# internal source release
python ./setup.py sdist
scp ./dist/ZhuSuan-0.3.0.tar.gz juncluster3:~/public_html/

# internal wheel release
python ./setup.py bdist_wheel --universal
scp ./dist/ZhuSuan-0.3.0-py2.py3-none-any.whl juncluster3:~/public_html/
