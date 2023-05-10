export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
cd src
####all 2.0
nohup python -u train.py ../config/awa1.yaml >./outlog/awa1.log &
nohup python -u train.py ../config/awa2.yaml >./outlog/awa2.log &
nohup python -u train.py ../config/sun.yaml >./outlog/sun.log &
nohup python -u train.py ../config/flo.yaml >./outlog/flo.log &
nohup python -u train.py ../config/cub.yaml >./outlog/cub.log &

###===================================================================
###ablation -cc 20
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa120.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo20.log &
nohup python -u train.py ../config/cub.yaml >./aoutlog/cub20.log &
nohup python -u train.py ../config/sun.yaml >./outlog/sun20.log &
nohup python -u train.py ../config/awa2.yaml >./outlog/awa220.log &
###===================================================================
###ablation -cls 3.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo.log &
nohup python -u train.py ../config/cub.yaml >./aoutlog/cub.log &
nohup python -u train.py ../config/sun.yaml >./outlog/sun.log &
nohup python -u train.py ../config/awa2.yaml >./outlog/awa2.log &
#
nohup python -u test.py ../config/awa1.yaml >./abl/awa1-cls.log &
nohup python -u test.py ../config/flo.yaml >./abl/flo-cls.log &
nohup python -u test.py ../config/cub.yaml >./abl/cub-cls.log &
nohup python -u test.py ../config/sun.yaml >./abl/sun-cls.log &
nohup python -u test.py ../config/awa2.yaml >./abl/awa2-cls.log &
###===================================================================
###ablation -w 10.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-10.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-10.log &
nohup python -u train.py ../config/cub.yaml >./aoutlog/cub-10.log &
nohup python -u train.py ../config/sun.yaml >./outlog/sun-10.log &
nohup python -u train.py ../config/awa2.yaml >./outlog/awa2-10.log &
#
nohup python -u test.py ../config/awa1.yaml >./abl/awa1-w.log &
nohup python -u test.py ../config/flo.yaml >./abl/flo-w.log &
nohup python -u test.py ../config/cub.yaml >./abl/cub-w.log &
nohup python -u test.py ../config/sun.yaml >./abl/sun-w.log &
nohup python -u test.py ../config/awa2.yaml >./abl/awa2-w.log &
###===================================================================
###ablation -cr 4.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-2.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-2.log &
nohup python -u train.py ../config/cub.yaml >./aoutlog/cub-2.log &
nohup python -u train.py ../config/sun.yaml >./outlog/sun-2.log &
nohup python -u train.py ../config/awa2.yaml >./outlog/awa2-2.log &
#
nohup python -u test.py ../config/awa1.yaml >./abl/awa1-cr.log &
nohup python -u test.py ../config/flo.yaml >./abl/flo-cr.log &
nohup python -u test.py ../config/cub.yaml >./abl/cub-cr.log &
nohup python -u test.py ../config/sun.yaml >./abl/sun-cr.log &
nohup python -u test.py ../config/awa2.yaml >./abl/awa2-cr.log &

###===================================================================
###ablation +vae 5.0 
# nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-3.log &
# nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-3.log &
# nohup python -u train.py ../config/cub.yaml >./aoutlog/cub-3.log &
# nohup python -u train.py ../config/sun.yaml >./outlog/sun.log &
# nohup python -u train.py ../config/awa2.yaml >./outlog/awa2.log &
# #
# nohup python -u test.py ../config/awa1.yaml >./abl/awa1-vae.log &
# nohup python -u test.py ../config/flo.yaml >./abl/flo-vae.log &
# nohup python -u test.py ../config/cub.yaml >./abl/cub-vae.log &

###===================================================================
###para -dim 32 6.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-4.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-4.log &
nohup python -u train.py ../config/cub.yaml >./aoutlog/cub-4.log &
#
nohup python -u test.py ../config/awa1.yaml >./dim/awa1-32.log &
nohup python -u test.py ../config/flo.yaml >./dim/flo-32.log &
nohup python -u test.py ../config/cub.yaml >./dim/cub-32.log &

###===================================================================
###para -dim 128 6.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-5.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-5.log &
nohup python -u train.py ../config/cub.yaml >./aoutlog/cub-5.log &
#
nohup python -u test.py ../config/awa1.yaml >./dim/awa1-128.log &
nohup python -u test.py ../config/flo.yaml >./dim/flo-128.log &
nohup python -u test.py ../config/cub.yaml >./dim/cub-128.log &
###===================================================================
###para -dim 16 6.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-6.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-6.log &
nohup python -u train.py ../config/cub.yaml >./aoutlog/cub-6.log &
#
nohup python -u test.py ../config/awa1.yaml >./dim/awa1-16.log &
nohup python -u test.py ../config/flo.yaml >./dim/flo-16.log &
nohup python -u test.py ../config/cub.yaml >./dim/cub-16.log &
###===================================================================
###para -dim 256 6.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-7.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-7.log &
nohup python -u train.py ../config/cub.yaml >./aoutlog/cub-7.log &
#
nohup python -u test.py ../config/awa1.yaml >./dim/awa1-256.log &
nohup python -u test.py ../config/flo.yaml >./dim/flo-256.log &
nohup python -u test.py ../config/cub.yaml >./dim/cub-256.log &
###===================================================================
###para -cls 0.01 7.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-cls-0.01.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-cls-0.01.log &
#
nohup python -u test.py ../config/awa1.yaml >./cls/awa1-0.01.log &
nohup python -u test.py ../config/flo.yaml >./cls/flo-0.01.log &
###===================================================================

###para -cls 0.1 7.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-cls-0.1.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-cls-0.1.log &
#
nohup python -u test.py ../config/awa1.yaml >./cls/awa1-0.1.log &
nohup python -u test.py ../config/flo.yaml >./cls/flo-0.1.log &
###===================================================================
###para - cls 10 7.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-cls-10.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-cls-10.log &
#
nohup python -u test.py ../config/awa1.yaml >./cls/awa1-10.log &
nohup python -u test.py ../config/flo.yaml >./cls/flo-10.log &
###===================================================================
###para - cls 5 7.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-cls-5.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-cls-5.log &
#
nohup python -u test.py ../config/awa1.yaml >./cls/awa1-5.log &
nohup python -u test.py ../config/flo.yaml >./cls/flo-5.log &
###===================================================================
###para - cls 0.5 7.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-cls-0.5.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-cls-0.5.log &
#
nohup python -u test.py ../config/awa1.yaml >./cls/awa1-0.5.log &
nohup python -u test.py ../config/flo.yaml >./cls/flo-0.5.log &
###===================================================================
###para - cls 0.05 7.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-cls-0.05.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-cls-0.05.log &
#
nohup python -u test.py ../config/awa1.yaml >./cls/awa1-0.05.log &
nohup python -u test.py ../config/flo.yaml >./cls/flo-0.05.log &

###===================================================================
###para -w 0.05 8.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-w-0.05.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-w-0.05.log &
#
nohup python -u test.py ../config/awa1.yaml >./ww/awa1-0.05.log &
nohup python -u test.py ../config/flo.yaml >./ww/flo-0.05.log &
###===================================================================
###para -w 0.01 8.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-w-0.01.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-w-0.01.log &
#
nohup python -u test.py ../config/awa1.yaml >./ww/awa1-0.01.log &
nohup python -u test.py ../config/flo.yaml >./ww/flo-0.01.log &
###===================================================================
###para -w 0.5 8.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-w-0.5.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-w-0.5.log &
#
nohup python -u test.py ../config/awa1.yaml >./ww/awa1-0.5.log &
nohup python -u test.py ../config/flo.yaml >./ww/flo-0.5.log &
###===================================================================
###para -w 1 8.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-w-1.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-w-1.log &
#
nohup python -u test.py ../config/awa1.yaml >./ww/awa1-1.log &
nohup python -u test.py ../config/flo.yaml >./ww/flo-1.log &
###===================================================================
###para -w 5 8.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-w-5.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-w-5.log &
#
nohup python -u test.py ../config/awa1.yaml >./ww/awa1-5.log &
nohup python -u test.py ../config/flo.yaml >./ww/flo-5.log &
###===================================================================
###para -w 10 8.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-w-10.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-w-10.log &
#
nohup python -u test.py ../config/awa1.yaml >./ww/awa1-10.log &
nohup python -u test.py ../config/flo.yaml >./ww/flo-10.log &
###===================================================================
###para -cr 10 9.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-cr-10.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-cr-10.log &
#
nohup python -u test.py ../config/awa1.yaml >./cr/awa1-10.log &
nohup python -u test.py ../config/flo.yaml >./cr/flo-10.log &
###===================================================================
###para -cr 5 9.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-cr-5.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-cr-5.log &
#
nohup python -u test.py ../config/awa1.yaml >./cr/awa1-5.log &
nohup python -u test.py ../config/flo.yaml >./cr/flo-5.log &
###===================================================================
###para -cr 0.5 9.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-cr-0.5.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-cr-0.5.log &
#
nohup python -u test.py ../config/awa1.yaml >./cr/awa1-0.5.log &
nohup python -u test.py ../config/flo.yaml >./cr/flo-0.5.log &
###===================================================================
###para -cr 0.1 9.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-cr-0.1.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-cr-0.1.log &
#
nohup python -u test.py ../config/awa1.yaml >./cr/awa1-0.1.log &
nohup python -u test.py ../config/flo.yaml >./cr/flo-0.1.log &
###===================================================================
###para -cr 0.05 9.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-cr-0.05.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-cr-0.05.log &
#
nohup python -u test.py ../config/awa1.yaml >./cr/awa1-0.05.log &
nohup python -u test.py ../config/flo.yaml >./cr/flo-0.05.log &
###===================================================================
###para -cr 0.01 9.0 
nohup python -u train.py ../config/awa1.yaml >./aoutlog/awa1-cr-0.01.log &
nohup python -u train.py ../config/flo.yaml >./aoutlog/flo-cr-0.01.log &
#
nohup python -u test.py ../config/awa1.yaml >./cr/awa1-0.01.log &
nohup python -u test.py ../config/flo.yaml >./cr/flo-0.01.log &



###===================================================================
###f-clswgan
nohup python -u test.py ../config/awa1.yaml >./testlog/awa1.log &
nohup python -u test.py ../config/awa2.yaml >./testlog/awa2.log &
nohup python -u test.py ../config/sun.yaml >./testlog/sun.log &
nohup python -u test.py ../config/flo.yaml >./testlog/flo.log &
nohup python -u test.py ../config/cub.yaml >./testlog/cub.log &
###ours
nohup python -u test.py ../config/awa1.yaml >./ourtestlog/awa1.log &
nohup python -u test.py ../config/awa2.yaml >./ourtestlog/awa2.log &
nohup python -u test.py ../config/sun.yaml >./ourtestlog/sun.log &
nohup python -u test.py ../config/flo.yaml >./ourtestlog/flo.log &
nohup python -u test.py ../config/cub.yaml >./ourtestlog/cub.log &
