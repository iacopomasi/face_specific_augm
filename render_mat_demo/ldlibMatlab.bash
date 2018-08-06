#/bin/sh
LIBS_PREFIX=/home/iac/Code/face_specific_augm/render_mat_demo/local/
env LD_LIBRARY_PATH=$LIBS_PREFIX/Mesa-7.0.3/lib64/:$LIBS_PREFIX/lib64/osgPlugins-2.8.1/:$LIBS_PREFIX/lib64:$LIBS_PREFIX/lib /usr/local/MATLAB/R2013b/bin/matlab -desktop
