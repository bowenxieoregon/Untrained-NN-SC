datagen.sh is for datageneration
deepdecoders.py is the arch of deepdecoder
earlystopping.py is the class for implementation of EMV
run_deepdecoder_div.py is for using deepdecoder with unshared weights between emitters
run_deepdecoder_reg.py is for shared weights between emitters and add a distance from current params to the ininitialization regularization using deepdecoder
run_non_deepdecoder.py is for shared weights between emitters using deepdecoder
run_non_deepdecoder_nonlog.py the same as run_non_deepdecoder.py, but not transformed to log domain 
run_non_dip.py is for for shared weights between emitters using DIP(skip) model
run_non_dip_nonlog.py is same as run_non_dip.py, but not transformed to log domain.
