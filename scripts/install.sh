#!/bin/bash

set -e


conda_url=https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
venv_dir=$PWD/venv

mark=.done-venv
if [ ! -f $mark ]; then
  echo 'Making python virtual environment'
  name=$(basename $conda_url)
  if [ ! -f $name ]; then
    wget $conda_url || exit 1
  fi
  [ ! -f $name ] && echo "File $name does not exist" && exit 1
  [ -d $venv_dir ] && rm -r $venv_dir
  sh $name -b -p $venv_dir || exit 1
  . $venv_dir/bin/activate
  echo 'Installing python dependencies'
  pip install -r requirements.txt || exit 1
  touch $mark
fi
echo "if [ \"\$(which python)\" != \"$venv_dir/bin/python\" ]; then source $venv_dir/bin/activate; fi" > env.sh

sv56=https://github.com/openitu/STL/archive/refs/tags/v2009.tar.gz
sv56_dir=$PWD/STL-2009
mark=.done-sv56
home=$PWD
if [ ! -f $mark ]; then
  if [ -z "$(which sv56demo)" ]; then
    if [ ! -f $(basename $sv56) ]; then
      wget -q $sv56 || exit 1
    fi
    echo 'Unpacking sv56 source files'
    [ -d $sv56_dir ] && rm -r $sv56_dir
    tar -xf $(basename $sv56) || exit 1
    echo 'Building sv56'
    cd $sv56_dir/src/sv56
    make -f makefile.unx || exit 1
    if [ ! -e $sv56_dir/src/sv56/sv56demo ]; then
	echo 'fail to compile sv56'
	exit 1
    fi
  fi
  cd $home
  touch $mark
fi

echo "export PATH=$sv56_dir/src/sv56:\$PATH" >> env.sh
