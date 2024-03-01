1.datagen.sh is for datageneration

2.deepdecoders.py is the arch of deepdecoder

3.earlystopping.py is the class for implementation of EMV

4.run_deepdecoder_div.py is for using deepdecoder with unshared weights between emitters

5.run_deepdecoder_reg.py is for shared weights between emitters and add a distance from current params to the ininitialization regularization using deepdecoder

6.run_non_deepdecoder.py is for shared weights between emitters using deepdecoder

7.run_non_deepdecoder_nonlog.py the same as run_non_deepdecoder.py, but not transformed to log domain 

8.run_non_dip.py is for for shared weights between emitters using DIP(skip) model, which is saved in model folder

9.run_non_dip_nonlog.py is same as run_non_dip.py, but not transformed to log domain.
